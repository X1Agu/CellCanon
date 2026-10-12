from __future__ import annotations
import json
import pandas as pd
from cellcanon.resources.sciplex_knowledge import OUTPUT, sha256, write_json
from cellcanon.resources.sciplex_public_background import PublicSnapshot
from cellcanon.resources.sciplex_drug_background import batches
from cellcanon.resources.sciplex_pubchem_background import collect_pubchem
from cellcanon.resources.sciplex_drug_background import same_drug, standard_parent_key
from cellcanon.model.chembl_binding_annotations import TYPES, measurement, curate, threshold_evidence

def run(output=OUTPUT):
    root = output / 'catalog_annotations'
    root.mkdir(parents=True, exist_ok=True)
    resolution_path = output / 'catalog_identity/resolution.json'
    resolution = [r for r in json.loads(resolution_path.read_text()) if r['accepted']]
    client = PublicSnapshot(output / 'public_background/http')
    status = client.json('https://www.ebi.ac.uk/chembl/api/data/status.json')
    assert status['chembl_db_version'] == 'ChEMBL_37'
    bymol = {mid: r for r in resolution for mid in r['molecule_chembl_ids']}
    assert len(bymol) == sum((len(r['molecule_chembl_ids']) for r in resolution)), 'A catalog molecule cannot silently overwrite another drug mapping.'
    ids = sorted(bymol)
    mechanisms = []
    activities = []
    for batch in batches(ids):
        mechanisms.extend(client.chembl_query('mechanism', {'molecule_chembl_id__in': ','.join(batch), 'limit': 1000}, 'mechanisms'))
        activities.extend(client.chembl_query('activity', {'molecule_chembl_id__in': ','.join(batch), 'assay_type': 'B', 'target_organism': 'Homo sapiens', 'standard_type__in': ','.join(TYPES), 'limit': 1000}, 'activities'))
    preliminary = [a for a in activities if measurement(a)[1] is None]
    targetids = {m['target_chembl_id'] for m in mechanisms if m.get('target_chembl_id')} | {a['target_chembl_id'] for a in preliminary}
    targets = []
    assays = []
    for batch in batches(sorted(targetids), 75):
        targets.extend(client.chembl_query('target', {'target_chembl_id__in': ','.join(batch), 'limit': 1000}, 'targets'))
    for batch in batches(sorted({a['assay_chembl_id'] for a in preliminary}), 75):
        assays.extend(client.chembl_query('assay', {'assay_chembl_id__in': ','.join(batch), 'limit': 1000}, 'assays'))
    targetmap = {t['target_chembl_id']: t for t in targets}
    assaymap = {a['assay_chembl_id']: a for a in assays}
    accepted = []
    audit = []
    for a in activities:
        record, reason = curate(a, assaymap.get(a['assay_chembl_id']), targetmap.get(a['target_chembl_id']))
        r = bymol[a['molecule_chembl_id']]
        audit.append({'drug': r['drug'], 'activity_id': a['activity_id'], 'status': reason or 'accepted'})
        if record:
            accepted.append({'drug': r['drug'], 'split': r['split'], 'identity_tier': 'catalog_reconciled_not_native_full_identity', **record})
    evidence = pd.DataFrame(accepted)
    evidence.to_parquet(root / 'binding_evidence.parquet', index=False)
    pd.DataFrame(audit).to_parquet(root / 'activity_audit.parquet', index=False)
    write_json(root / 'mechanisms.json', mechanisms)
    write_json(root / 'targets.json', targets)
    descriptors = [{'drug': r['drug'], 'split': r['split'], 'canonical_smiles': r['resolved_smiles']} for r in resolution]
    pub = collect_pubchem(descriptors, client, root, same_drug, standard_parent_key)
    coverage = {}
    for split in ['support', 'validation', 'test']:
        rs = [r for r in resolution if r['split'] == split]
        es = evidence[evidence.split.eq(split)]
        coverage[split] = {'resolved_drugs': len(rs), 'with_binding_measurement': es.drug.nunique(), 'active_1uM_drugs': len({r.drug for r in es.itertuples() if threshold_evidence(r.value_nM, r.relation, 1000.0)[0]}), 'pubchem_descriptions': sum((any((d.get('Description') for d in r['descriptions'])) for r in pub if r['split'] == split))}
    receipt = {'status': 'CATALOG_BACKGROUND_PREPARED', 'coverage': coverage, 'source_sha256': {str(resolution_path): sha256(resolution_path)}, 'http_attempts_this_invocation': client.calls, 'output_sha256': {p.name: sha256(p) for p in root.iterdir() if p.suffix in {'.json', '.parquet'} and p.name != 'manifest.json'}, 'claim_boundary': 'Catalogue source annotations; chemical identity is resolved separately.'}
    write_json(root / 'manifest.json', receipt)
    print(json.dumps({'coverage': coverage, 'binding_rows': len(evidence), 'http_attempts': client.calls}), flush=True)
