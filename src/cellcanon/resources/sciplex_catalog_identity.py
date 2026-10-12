from __future__ import annotations
from collections import defaultdict
import json
import urllib.parse
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from cellcanon.resources.sciplex_knowledge import OUTPUT, declared_drug_names, normalized_name, write_json, sha256
from cellcanon.resources.sciplex_public_background import PublicSnapshot
from cellcanon.resources.sciplex_drug_background import standard_parent_key, annotation_key

def compatible_completion(native, remote):
    akey, bkey = (standard_parent_key(native), standard_parent_key(remote))
    if not akey or not bkey:
        return False
    a = Chem.MolFromSmiles(annotation_key(native))
    b = Chem.MolFromSmiles(annotation_key(remote))
    if akey.split('-')[0] != bkey.split('-')[0]:
        if Chem.FindMolChiralCenters(a, includeUnassigned=False) or Chem.FindMolChiralCenters(b, includeUnassigned=False):
            return False
        enum = rdMolStandardize.TautomerEnumerator()
        enum.SetRemoveBondStereo(False)
        return Chem.MolToSmiles(enum.Canonicalize(a)) == Chem.MolToSmiles(enum.Canonicalize(b))
    return a.GetNumAtoms() == b.GetNumAtoms() and a.GetNumBonds() == b.GetNumBonds() and b.HasSubstructMatch(a, useChirality=True)

def run(output=OUTPUT):
    root = output / 'catalog_identity'
    root.mkdir(parents=True, exist_ok=True)
    client = PublicSnapshot(output / 'public_background/http')
    drugs = pd.read_parquet(output / 'local_audit/drug_coverage.parquet')
    strict = pd.read_parquet(output / 'drug_background/chembl_identity.parquet')
    missing = drugs[~drugs.drug.isin(strict.drug)]
    molecules = {m['molecule_chembl_id']: m for m in json.loads((output / 'drug_background/chembl_molecules.json').read_text())}
    previous = json.loads((output / 'drug_background/name_candidate_audit.json').read_text())
    candidates = defaultdict(set)
    for r in previous:
        candidates[r['drug']].update(r['candidate_ids'])
    names = defaultdict(list)
    named_sources = defaultdict(list)
    for rp in client.root.glob('*.receipt.json'):
        record = json.loads(rp.read_text())
        url = record['url']
        if not url.startswith('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/') or '/property/' not in url:
            continue
        path = rp.with_name(rp.name.replace('.receipt.json', '.json'))
        if not path.exists():
            continue
        assert sha256(path) == record['sha256']
        name = normalized_name(urllib.parse.unquote(url.split('/name/', 1)[1].split('/property/', 1)[0]))
        names[name].extend(json.loads(path.read_text()).get('PropertyTable', {}).get('Properties', []))
        named_sources[name].append(url)
    results = []
    for d in missing.to_dict('records'):
        declared = declared_drug_names(d)
        if not candidates[d['drug']]:
            for name in declared:
                found = client.chembl_query('molecule', {'pref_name__iexact': name, 'limit': 1000}, 'molecules')
                for m in found:
                    molecules[m['molecule_chembl_id']] = m
                    candidates[d['drug']].add(m['molecule_chembl_id'])
        chems = defaultdict(list)
        for mid in candidates[d['drug']]:
            m = molecules[mid]
            smiles = (m.get('molecule_structures') or {}).get('canonical_smiles')
            if compatible_completion(d['canonical_smiles'], smiles):
                chems[standard_parent_key(smiles)].append(m)
        pubs = defaultdict(list)
        for name in declared:
            for p in names.get(normalized_name(name), []):
                smiles = p.get('SMILES') or p.get('IsomericSMILES')
                if compatible_completion(d['canonical_smiles'], smiles):
                    pubs[standard_parent_key(smiles)].append(p)
        common = set(chems) & set(pubs)
        agreement = 'same_full_standard_parent_inchi'
        if not common:
            refined = {}
            for ck, cm in chems.items():
                for pk, ps in pubs.items():
                    if ck.split('-')[0] != pk.split('-')[0]:
                        continue
                    cs = cm[0]['molecule_structures']['canonical_smiles']
                    agreed = [p for p in ps if compatible_completion(p.get('SMILES') or p.get('IsomericSMILES'), cs)]
                    if agreed:
                        refined.setdefault(ck, []).extend(agreed)
            if len(refined) == 1:
                common = set(refined)
                pubs.update(refined)
                agreement = 'named_pubchem_stereochemistry_refined_by_named_chembl_record'
        accepted = len(common) == 1
        key = next(iter(common)) if accepted else None
        ms = chems[key] if accepted else []
        results.append({'drug': d['drug'], 'split': d['split'], 'canonical_name': d['canonical_name'], 'native_smiles': d['canonical_smiles'], 'native_standard_parent_inchi': standard_parent_key(d['canonical_smiles']), 'accepted': accepted, 'resolved_standard_parent_inchi': key, 'registry_agreement': agreement if accepted else None, 'native_tautomer_reconciled': bool(accepted and standard_parent_key(d['canonical_smiles']).split('-')[0] != key.split('-')[0]), 'resolved_smiles': ms[0]['molecule_structures']['canonical_smiles'] if accepted else None, 'molecule_chembl_ids': sorted((m['molecule_chembl_id'] for m in ms)), 'pubchem_cids': sorted({p['CID'] for p in pubs[key]}) if accepted else [], 'status': 'catalog_name_two_registry_consistent_completion' if accepted else 'unresolved_no_unique_compatible_two_registry_agreement', 'strict_native_identity': False, 'predictor_descriptor_changed': False, 'candidate_chembl_ids': sorted(candidates[d['drug']]), 'compatible_chembl_keys': sorted(chems), 'compatible_pubchem_keys': sorted(pubs), 'pubchem_named_sources': sorted({url for name in declared for url in named_sources.get(normalized_name(name), [])})})
    write_json(root / 'resolution.json', results)
    write_json(root / 'molecule_records.json', list(molecules.values()))
    summary = {'status': 'CATALOG_IDENTITIES_RESOLVED', 'native_strict_drugs': strict.drug.nunique(), 'remaining_audited': len(results), 'accepted_completions': sum((r['accepted'] for r in results)), 'accepted_test_completions': sum((r['accepted'] and r['split'] == 'test' for r in results)), 'unresolved_drugs': [r['drug'] for r in results if not r['accepted']], 'http_attempts': client.calls, 'output_sha256': {p.name: sha256(p) for p in root.glob('*.json') if p.name != 'manifest.json'}}
    write_json(root / 'manifest.json', summary)
    print(json.dumps(summary), flush=True)
