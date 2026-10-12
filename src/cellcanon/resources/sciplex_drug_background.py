from __future__ import annotations
from functools import lru_cache
import json
import pandas as pd
from rdkit import Chem
from cellcanon.model.chembl_binding_annotations import TYPES, curate, measurement, threshold_evidence
from cellcanon.model.target_pathway_features import annotation_key
from cellcanon.resources.sciplex_knowledge import OUTPUT, sha256, write_json, declared_drug_names
from cellcanon.resources.sciplex_public_background import PublicSnapshot

@lru_cache(maxsize=10000)
def standard_parent_key(smiles):
    parent = annotation_key(smiles)
    return Chem.MolToInchiKey(Chem.MolFromSmiles(parent)) if parent else None

def same_drug(local_smiles, remote_smiles):
    a, b = (standard_parent_key(local_smiles), standard_parent_key(remote_smiles))
    return bool(a and b and (a == b))

def batches(items, size=25):
    items = list(items)
    return [items[i:i + size] for i in range(0, len(items), size)]

def run(output=OUTPUT):
    root = output / 'drug_background'
    root.mkdir(parents=True, exist_ok=True)
    client = PublicSnapshot(output / 'public_background/http')
    drugs = pd.read_parquet(output / 'local_audit/drug_coverage.parquet').to_dict('records')
    drugmap = {r['drug']: r for r in drugs}
    status = client.json('https://www.ebi.ac.uk/chembl/api/data/status.json')
    assert status['chembl_db_version'] == 'ChEMBL_37', 'Keep existing and new assay release comparable.'
    molecules = {}
    priorids = sorted({v for r in drugs for v in r['chembl_ids']})
    for batch in batches(priorids, 50):
        for mol in client.chembl_query('molecule', {'molecule_chembl_id__in': ','.join(batch), 'limit': 1000}, 'molecules'):
            molecules[mol['molecule_chembl_id']] = mol
    mapping_rows, name_audit = ([], [])
    missing = [r for r in drugs if len(r['chembl_ids']) == 0]
    for batch in batches(missing, 15):
        keys = sorted({standard_parent_key(r['canonical_smiles']) for r in batch})
        for mol in client.chembl_query('molecule', {'molecule_structures__standard_inchi_key__in': ','.join(keys), 'limit': 1000}, 'molecules'):
            molecules[mol['molecule_chembl_id']] = mol
    for r in drugs:
        matches = [m for m in molecules.values() if same_drug(r['canonical_smiles'], (m.get('molecule_structures') or {}).get('canonical_smiles'))]
        if not matches:
            for name in [str(r['legacy_drug']), *declared_drug_names(r)]:
                found = client.chembl_query('molecule', {'molecule_synonyms__molecule_synonym__iexact': name, 'limit': 1000}, 'molecules')
                accepted = [m for m in found if same_drug(r['canonical_smiles'], (m.get('molecule_structures') or {}).get('canonical_smiles'))]
                name_audit.append({'drug': r['drug'], 'name': name, 'candidate_ids': [m['molecule_chembl_id'] for m in found], 'accepted_ids': [m['molecule_chembl_id'] for m in accepted]})
                for m in found:
                    molecules[m['molecule_chembl_id']] = m
                if accepted:
                    matches = accepted
                    break
        for mol in matches:
            remote = mol['molecule_structures']['canonical_smiles']
            mapping_rows.append({'drug': r['drug'], 'split': r['split'], 'molecule_chembl_id': mol['molecule_chembl_id'], 'identity_status': 'exact_standardized_parent_smiles' if annotation_key(remote) == annotation_key(r['canonical_smiles']) else 'same_full_standard_parent_inchi_tautomer_normalized', 'standard_parent_inchi_key': standard_parent_key(remote), 'canonical_smiles': remote, 'preferred_name': mol.get('pref_name'), 'source_release': status['chembl_db_version']})
    mapping = pd.DataFrame(mapping_rows).drop_duplicates(['drug', 'molecule_chembl_id'])
    mapping.to_parquet(root / 'chembl_identity.parquet', index=False)
    write_json(root / 'name_candidate_audit.json', name_audit)
    write_json(root / 'chembl_molecules.json', list(molecules.values()))
    print(json.dumps({'event': 'chembl_identity_complete', 'drugs': mapping.drug.nunique(), 'test': mapping.query("split == 'test'").drug.nunique(), 'http_attempts': client.calls}), flush=True)
    molecule_drugs = mapping.groupby('molecule_chembl_id').drug.agg(list).to_dict()
    mechanisms = []
    for batch in batches(sorted(molecule_drugs)):
        mechanisms.extend(client.chembl_query('mechanism', {'molecule_chembl_id__in': ','.join(batch), 'limit': 1000}, 'mechanisms'))
    write_json(root / 'chembl_mechanisms.json', mechanisms)
    targets = []
    for batch in batches(sorted({m['target_chembl_id'] for m in mechanisms if m.get('target_chembl_id')}), 75):
        targets.extend(client.chembl_query('target', {'target_chembl_id__in': ','.join(batch), 'limit': 1000}, 'targets'))
    write_json(root / 'chembl_mechanism_targets.json', targets)
    newids = sorted(set(molecule_drugs) - set(priorids))
    activities = []
    for batch in batches(newids):
        activities.extend(client.chembl_query('activity', {'molecule_chembl_id__in': ','.join(batch), 'assay_type': 'B', 'target_organism': 'Homo sapiens', 'standard_type__in': ','.join(TYPES), 'limit': 1000}, 'activities'))
    preliminary = [a for a in activities if measurement(a)[1] is None]
    assays = []
    for batch in batches(sorted({a['assay_chembl_id'] for a in preliminary}), 75):
        assays.extend(client.chembl_query('assay', {'assay_chembl_id__in': ','.join(batch), 'limit': 1000}, 'assays'))
    known = {t['target_chembl_id']: t for t in targets}
    for batch in batches(sorted({a['target_chembl_id'] for a in preliminary} - set(known)), 75):
        for t in client.chembl_query('target', {'target_chembl_id__in': ','.join(batch), 'limit': 1000}, 'targets'):
            known[t['target_chembl_id']] = t
    assaymap = {a['assay_chembl_id']: a for a in assays}
    accepted, assay_audit = ([], [])
    for activity in activities:
        record, reason = curate(activity, assaymap.get(activity['assay_chembl_id']), known.get(activity['target_chembl_id']))
        for drug in molecule_drugs[activity['molecule_chembl_id']]:
            assay_audit.append({'drug': drug, 'activity_id': activity['activity_id'], 'status': reason or 'accepted'})
            if record:
                accepted.append({'drug': drug, 'split': drugmap[drug]['split'], **record})
    old = pd.read_parquet(output / 'local_audit/binding_evidence.parquet')
    combined = pd.concat([old, pd.DataFrame(accepted)], ignore_index=True).drop_duplicates(['drug', 'activity_id'])
    combined.to_parquet(root / 'binding_evidence.parquet', index=False)
    pd.DataFrame(assay_audit).to_parquet(root / 'new_activity_audit.parquet', index=False)
    print(json.dumps({'event': 'binding_complete', 'rows': len(combined), 'drugs': combined.drug.nunique(), 'test': combined.query("split == 'test'").drug.nunique(), 'http_attempts': client.calls}), flush=True)
    from .sciplex_pubchem_background import collect_pubchem
    pubchem_rows = collect_pubchem(drugs, client, root, same_drug, standard_parent_key)
    summary = {}
    for split in ['support', 'validation', 'test']:
        ids = {r['drug'] for r in drugs if r['split'] == split}
        group = combined[combined.drug.isin(ids)]
        active = {r.drug for r in group.itertuples() if threshold_evidence(r.value_nM, r.relation, 1000.0)[0]}
        pub = [r for r in pubchem_rows if r['drug'] in ids]
        summary[split] = {'drugs': len(ids), 'chembl_identity': len(set(mapping.drug) & ids), 'binding_measured': group.drug.nunique(), 'binding_active_1uM': len(active), 'pubchem_identity': sum((r['cid'] is not None for r in pub)), 'pubchem_description': sum((any((d.get('Description') for d in r['descriptions'])) for r in pub))}
    receipt = {'status': 'DRUG_PUBLIC_BACKGROUND_RETRIEVED', 'coverage': summary, 'new_assay_records': len(accepted), 'new_molecule_ids': newids, 'chembl_service_status': status, 'http_attempts_this_invocation': client.calls, 'llm_calls': 0, 'response_expression_queries': 0, 'identity_rule': 'Full standard parent InChIKey equality; standard tautomer normalization disclosed separately from literal canonical SMILES.', 'output_sha256': {p.name: sha256(p) for p in root.iterdir() if p.suffix in {'.json', '.parquet'} and p.name != 'manifest.json'}}
    write_json(root / 'manifest.json', receipt)
    print(json.dumps({'event': 'drug_background_complete', 'coverage': summary}), flush=True)
