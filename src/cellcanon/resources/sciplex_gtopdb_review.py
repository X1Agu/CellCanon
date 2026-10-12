from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlencode
import pandas as pd
from cellcanon.resources.sciplex_knowledge import OUTPUT, sha256, write_json
from cellcanon.resources.sciplex_layer5_coverage import FUNCTIONAL_SOURCE_ROOT
from cellcanon.resources.sciplex_public_background import PublicSnapshot
PARENT = OUTPUT.parent / 'layer5_audit_v1'
DESTINATION = OUTPUT.parent / 'route_completion_v1'
ACTION_PAIRS = {('Inhibitor', 'Inhibition'): -1, ('Antagonist', 'Antagonist'): -1, ('Activator', 'Activation'): 1, ('Agonist', 'Full agonist'): 1}

def reference_ids(value):
    return sorted({x.strip() for x in str(value).split('|') if re.fullmatch('[1-9][0-9]*', x.strip())})

def review_record(row, resolve, reference_metadata):
    reasons = []
    symbol = resolve(row['Target Gene Symbol'])
    if row['Target Species'] != 'Human':
        reasons.append('nonhuman_target')
    if not symbol:
        reasons.append('unresolved_or_multigene_target')
    if str(row['Target Subunit IDs']).strip():
        reasons.append('multisubunit_target_requires_typed_route')
    if str(row['Primary Target']).lower() != 'true':
        reasons.append('not_curated_primary_target')
    sign = ACTION_PAIRS.get((row['Type'], row['Action']))
    if sign is None:
        reasons.append('action_unknown_binding_only_or_context_dependent')
    pmids = reference_ids(row['PubMed ID'])
    verified = [p for p in pmids if p in reference_metadata and reference_metadata[p].get('title')]
    if not pmids:
        reasons.append('no_pubmed_reference_web_only_or_missing')
    elif not verified:
        reasons.append('reference_identity_not_resolved')
    if any((re.search('\\bretracted\\b', reference_metadata[p]['title'], re.I) for p in verified)):
        reasons.append('retracted_reference_title')
    return {'accepted': not reasons, 'reasons': reasons, 'canonical_symbol': symbol, 'action_sign': sign, 'pubmed_ids': pmids, 'resolved_pubmed_ids': verified}

def run(destination=DESTINATION):
    root = destination / 'gtopdb_review'
    root.mkdir(parents=True, exist_ok=True)
    parent = json.loads((PARENT / 'manifest.json').read_text())
    inputs = {}

    def verify(path, expected):
        assert sha256(path) == expected, str(path)
        inputs[str(path.resolve())] = expected
    for name in ['seed_evidence.json', 'gtopdb_candidate_inventory.json']:
        verify(PARENT / name, parent['output_sha256'][name])
    mp = FUNCTIONAL_SOURCE_ROOT / 'functional_manifest.json'
    m = json.loads(mp.read_text())
    inputs[str(mp)] = sha256(mp)
    for item in m['outputs']:
        if item['name'].startswith('source_'):
            verify(FUNCTIONAL_SOURCE_ROOT / Path(item['relative_path']).name, item['sha256'])
    hp = OUTPUT / 'public_background/http' / (hashlib.sha256(b'https://rest.genenames.org/fetch/status/Approved').hexdigest() + '.json')
    verify(hp, json.loads(hp.with_suffix('.receipt.json').read_text())['sha256'])
    records = json.loads(hp.read_text())['response']['docs']
    primary = {h['symbol'] for h in records}
    aliases = {}
    for h in records:
        for a in [*h.get('alias_symbol', []), *h.get('prev_symbol', [])]:
            aliases.setdefault(a, set()).add(h['symbol'])

    def resolve(symbol):
        if symbol in primary:
            return symbol
        values = aliases.get(symbol, set())
        return next(iter(values)) if len(values) == 1 else None
    mapping = json.loads((PARENT / 'gtopdb_candidate_inventory.json').read_text())['mapping']
    raw_path = FUNCTIONAL_SOURCE_ROOT / 'gtopdb_2026.2_interactions.csv'
    raw = pd.read_csv(raw_path, skiprows=1, keep_default_na=False, low_memory=False)
    matched = raw.loc[raw['Ligand ID'].isin({r['candidate_ligand_ids'][0] for r in mapping if r['exact_unique']})]
    eligible_for_lookup = matched.loc[matched['Target Species'].eq('Human') & matched['Primary Target'].astype(str).str.lower().eq('true')]
    pmids = sorted({p for x in eligible_for_lookup['PubMed ID'] for p in reference_ids(x)})
    client = PublicSnapshot(root / 'http')
    metadata = {}
    for offset in range(0, len(pmids), 25):
        batch = pmids[offset:offset + 25]
        query = '(' + ' OR '.join(('EXT_ID:' + p for p in batch)) + ') AND SRC:MED'
        url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urlencode({'query': query, 'format': 'json', 'pageSize': 100, 'resultType': 'lite'})
        body = client.json(url)
        for r in body.get('resultList', {}).get('result', []):
            if r.get('source') == 'MED' and str(r.get('id')) in batch:
                metadata[str(r['id'])] = {k: r.get(k) for k in ['id', 'source', 'title', 'authorString', 'journalTitle', 'pubYear', 'doi', 'pubType']}
        print(json.dumps({'event': 'reference_metadata', 'requested': min(offset + 25, len(pmids)), 'resolved': len(metadata)}), flush=True)
    write_json(root / 'reference_metadata.json', metadata)
    old = json.loads((PARENT / 'seed_evidence.json').read_text())
    oldsigns = {(s['drug'], s['symbol'], s['action_sign']) for s in old if s['action_sign'] is not None}
    reviews = []
    seeds = []
    for d in mapping:
        if not d['exact_unique']:
            continue
        for index, r in raw.loc[raw['Ligand ID'].eq(d['candidate_ligand_ids'][0])].iterrows():
            record = r.to_dict()
            decision = review_record(record, resolve, metadata)
            eid = 'gtopdb:' + hashlib.sha256((inputs[str(raw_path)] + ':' + str(index)).encode()).hexdigest()[:24]
            review = {'drug': d['drug'], 'split': d['split'], 'identity_tier': d['identity_tier'], 'evidence_id': eid, 'source_row': int(index), 'source_url': 'https://www.guidetopharmacology.org/GRAC/LigandDisplayForward?ligandId=' + str(record['Ligand ID']), 'raw_record': record, **decision}
            reviews.append(review)
            if decision['accepted']:
                symbol = decision['canonical_symbol']
                seeds.append({'drug': d['drug'], 'symbol': symbol, 'node': 'GENE:' + symbol, 'tier': 'curated_moa', 'evidence_id': eid, 'source_url': review['source_url'], 'action': record['Type'].upper(), 'action_sign': decision['action_sign'], 'identity_tier': d['identity_tier'], 'source_kind': 'gtopdb_primary_functional_action', 'pubmed_ids': decision['resolved_pubmed_ids'], 'source_row': int(index), 'direct_physical_binding_verified': False, 'condition_causal_mechanism_verified': False, 'new_drug_gene_sign': (d['drug'], symbol, decision['action_sign']) not in oldsigns})
    write_json(root / 'review_records.json', reviews)
    write_json(root / 'accepted_seeds.json', seeds)
    write_json(root / 'mapping.json', mapping)
    summary = {}
    for split in ['support', 'validation', 'test']:
        drugs = {d['drug'] for d in mapping if d['split'] == split}
        ss = [s for s in seeds if s['drug'] in drugs]
        olddrugs = {d for d, _, _ in oldsigns}
        summary[split] = {'drug_denominator': len(drugs), 'accepted_rows': len(ss), 'accepted_drugs': len({s['drug'] for s in ss}), 'new_drug_gene_signs': len({(s['drug'], s['symbol'], s['action_sign']) for s in ss if s['new_drug_gene_sign']}), 'new_signed_seed_drugs': len({s['drug'] for s in ss} - olddrugs)}
    for p in (root / 'http').iterdir():
        if p.is_file():
            inputs[str(p.resolve())] = sha256(p)
    result = {'status': 'GTOPDB_LOCAL_PRIMARY_ACTION_REVIEW_COMPLETE', 'summary': summary, 'reviewed_rows': len(reviews), 'accepted_rows': len(seeds), 'excluded_reason_counts': dict(Counter((reason for r in reviews for reason in r['reasons']))), 'reference_ids_requested': len(pmids), 'reference_ids_resolved': len(metadata), 'public_metadata_requests': client.calls, 'reference_validation': 'bibliographic_identity_only; no abstract/full-text validation or condition-effect validation', 'source_sha256': inputs, 'source_code_sha256': sha256(Path(__file__)), 'output_sha256': {p.name: sha256(p) for p in root.iterdir() if p.is_file() and p.name != 'manifest.json'}, 'llm_calls': 0, 'training_runs': 0, 'perturbation_response_label_reads': 0}
    write_json(root / 'manifest.json', result)
    print(json.dumps({k: result[k] for k in ['status', 'summary', 'reviewed_rows', 'accepted_rows']}))
    return root
