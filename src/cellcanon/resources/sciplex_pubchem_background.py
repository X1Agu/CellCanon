from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from cellcanon.resources.sciplex_knowledge import sha256, write_json

def collect_pubchem(drugs, client, root, same_drug, standard_parent_key):
    properties = {}
    descriptions = defaultdict(list)
    failures = []
    for receiptpath in client.root.glob('*.receipt.json'):
        receipt = json.loads(receiptpath.read_text())
        url = receipt['url']
        if not url.startswith('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/'):
            continue
        key = hashlib.sha256(url.encode()).hexdigest()
        path = client.root / (key + '.json')
        if not path.exists():
            continue
        assert sha256(path) == receipt['sha256']
        value = json.loads(path.read_text())
        if '/property/' in url:
            for p in value.get('PropertyTable', {}).get('Properties', []):
                properties[p['CID']] = p
        elif '/description/' in url:
            for d in value.get('InformationList', {}).get('Information', []):
                if d not in descriptions[d['CID']]:
                    descriptions[d['CID']].append(d)

    def lookup(items, kind):
        if not items:
            return
        suffix = 'property/InChIKey,IsomericSMILES' if kind == 'inchikey' else 'description'
        url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/' + kind + '/' + ','.join(map(str, items)) + '/' + suffix + '/JSON'
        try:
            value = client.json(url)
        except (OSError, RuntimeError, ValueError) as error:
            if len(items) > 1:
                midpoint = len(items) // 2
                lookup(items[:midpoint], kind)
                lookup(items[midpoint:], kind)
            else:
                failures.append({'kind': kind, 'identifier': items[0], 'url': url, 'error': str(error)[:300]})
            return
        if kind == 'inchikey':
            records = value.get('PropertyTable', {}).get('Properties', [])
            for p in records:
                properties[p['CID']] = p
            returned = {p.get('InChIKey') for p in records}
        else:
            records = value.get('InformationList', {}).get('Information', [])
            for d in records:
                if d not in descriptions[d['CID']]:
                    descriptions[d['CID']].append(d)
            returned = {d['CID'] for d in records}
        missing = [item for item in items if item not in returned]
        if missing and len(items) > 1:
            for item in missing:
                lookup([item], kind)
        elif missing:
            failures.append({'kind': kind, 'identifier': items[0], 'url': url, 'error': 'successful_response_without_requested_identifier'})
    known = {standard_parent_key(p.get('SMILES') or p.get('IsomericSMILES')) for p in properties.values()}
    missing = sorted({standard_parent_key(r['canonical_smiles']) for r in drugs} - known - {None})
    for start in range(0, len(missing), 25):
        lookup(missing[start:start + 25], 'inchikey')
    bykey = defaultdict(list)
    for p in properties.values():
        bykey[standard_parent_key(p.get('SMILES') or p.get('IsomericSMILES'))].append(p)
    needed = sorted({p['CID'] for d in drugs for p in bykey.get(standard_parent_key(d['canonical_smiles']), []) if p['CID'] not in descriptions})
    for start in range(0, len(needed), 40):
        lookup(needed[start:start + 40], 'cid')
    rows = []
    for d in drugs:
        matched = bykey.get(standard_parent_key(d['canonical_smiles']), [])
        if matched:
            selected = min(matched, key=lambda p: (-sum((bool(v.get('Description')) for v in descriptions[p['CID']])), p['CID']))
            assert same_drug(d['canonical_smiles'], selected.get('SMILES') or selected.get('IsomericSMILES'))
            cid = selected['CID']
            rows.append({'drug': d['drug'], 'split': d['split'], 'cid': cid, 'identity_status': 'same_full_standard_parent_inchi', 'descriptions': descriptions[cid], 'description_url': f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/description/JSON', 'equivalent_cids': sorted((p['CID'] for p in matched)), 'errors': []})
        else:
            rows.append({'drug': d['drug'], 'split': d['split'], 'cid': None, 'identity_status': 'no_verified_pubchem_candidate', 'descriptions': [], 'errors': []})
    write_json(root / 'pubchem_drug_background.json', rows)
    write_json(root / 'pubchem_retrieval_failures.json', failures)
    write_json(root / 'pubchem_property_index.json', list(properties.values()))
    print(json.dumps({'event': 'pubchem_batch_complete', 'drugs': len(rows), 'identities': sum((r['cid'] is not None for r in rows)), 'descriptions': sum((any((v.get('Description') for v in r['descriptions'])) for r in rows)), 'retrieval_failures': len(failures)}), flush=True)
    return rows
