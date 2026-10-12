from __future__ import annotations
from collections import defaultdict
import json
import re
import urllib.parse
import pandas as pd
from cellcanon.resources.sciplex_knowledge import OUTPUT, SCIPLEX_ROOT, ENTITY_SOURCE_ROOT, sha256, write_json, nonempty, storage_strings
from cellcanon.resources.sciplex_public_background import PublicSnapshot

def split_terms(text):
    return [v for v in re.split('[;|\\s]+', str(text)) if v and v != '-']

def run(output=OUTPUT):
    root = output / 'gene_background'
    root.mkdir(parents=True, exist_ok=True)
    client = PublicSnapshot(output / 'public_background/http')
    hurl = 'https://rest.genenames.org/fetch/status/Approved'
    hpath = client.fetch(hurl)
    hdata = json.loads(hpath.read_text())['response']
    hgnc = hdata['docs']
    assert len(hgnc) == hdata['numFound'] and len(hgnc) > 40000
    byens_h, bysymbol_h = (defaultdict(list), defaultdict(list))
    for h in hgnc:
        assert h['status'] == 'Approved'
        if h.get('ensembl_gene_id'):
            byens_h[h['ensembl_gene_id']].append(h)
        bysymbol_h[h['symbol']].append(h)
        for sym in h.get('alias_symbol', []) + h.get('prev_symbol', []):
            bysymbol_h[sym].append(h)
    npath = SCIPLEX_ROOT.parent / 'reference/ncbi_gene/Homo_sapiens.gene_info.gz'
    ncbi = pd.read_csv(npath, sep='\t', dtype=str, keep_default_na=False)
    excluded_taxa = ncbi.loc[~ncbi['#tax_id'].eq('9606'), '#tax_id'].value_counts().to_dict()
    ncbi = ncbi.loc[ncbi['#tax_id'].eq('9606')].copy()
    assert len(ncbi) > 20000
    byens_n, bysymbol_n = (defaultdict(list), defaultdict(list))
    byid_n = {}
    for n in ncbi.to_dict('records'):
        byid_n[n['GeneID']] = n
        for en in re.findall('Ensembl:(ENSG\\d+)', n['dbXrefs']):
            byens_n[en].append(n)
        bysymbol_n[n['Symbol']].append(n)
        for sym in n['Synonyms'].split('|'):
            if sym != '-':
                bysymbol_n[sym].append(n)
    upath = output / 'public_background/uniprot_human_reviewed.parquet'
    uni = pd.read_parquet(upath)
    assert uni['Organism (ID)'].eq('9606').all()
    byid_u, bysym_u = (defaultdict(list), defaultdict(list))
    byaccession = uni.set_index('Entry').to_dict('index')
    for u in uni.to_dict('records'):
        for gid in split_terms(u['GeneID']):
            byid_u[gid].append(u)
        for sym in split_terms(u['Gene Names (primary)']):
            bysym_u[sym].append(u)
    panel = pd.read_parquet(output / 'local_audit/gene_coverage.parquet')
    rows = []
    for g in panel.to_dict('records'):
        en, symbol = (g['gene'], g['gene_symbol'])
        hs = byens_h.get(en, [])
        ns = byens_n.get(en, [])
        hids = {str(h['entrez_id']) for h in hs if h.get('entrez_id')}
        nids = {n['GeneID'] for n in ns}
        candidates = hids | nids
        status = 'exact_ensembl_to_human_gene' if len(candidates) == 1 else 'ambiguous_ensembl' if candidates else 'unresolved_external_identity'
        if not candidates:
            hs = list({h['hgnc_id']: h for h in bysymbol_h.get(symbol, [])}.values())
            ns = list({n['GeneID']: n for n in bysymbol_n.get(symbol, [])}.values())
            candidates = {str(h['entrez_id']) for h in hs if h.get('entrez_id')} | {n['GeneID'] for n in ns}
            if len(candidates) == 1:
                status = 'unique_human_symbol_or_declared_alias'
        gid = next(iter(candidates)) if len(candidates) == 1 else None
        n = byid_n.get(gid, {})
        proteins = byid_u.get(gid, []) if gid else []
        if not proteins and len(hs) == 1 and (len(candidates) <= 1):
            proteins = [{'Entry': a, **byaccession[a]} for a in hs[0].get('uniprot_ids', []) if a in byaccession]
        if not proteins and (not gid) and (not candidates):
            proteins = bysym_u.get(symbol, [])
            if len(proteins) == 1:
                status = 'unique_reviewed_human_primary_symbol'
        proteins = list({u['Entry']: u for u in proteins}.values())
        rows.append({'gene': en, 'gene_symbol': symbol, 'role': g['role'], 'gene_index': g['gene_index'], 'ncbi_gene_id': gid, 'hgnc_ids': sorted({h['hgnc_id'] for h in hs}), 'identity_status': status, 'candidate_ncbi_ids': sorted(candidates), 'canonical_symbol': n.get('Symbol') or (hs[0]['symbol'] if len(hs) == 1 else symbol), 'gene_type': n.get('type_of_gene') or (hs[0].get('locus_type') if len(hs) == 1 else None), 'gene_description': n.get('description') or (hs[0].get('name') if len(hs) == 1 else None), 'uniprot_accessions': [u['Entry'] for u in proteins], 'uniprot_function_texts': [{'accession': u['Entry'], 'text': u['Function [CC]'], 'url': 'https://www.uniprot.org/uniprotkb/' + u['Entry'] + '/entry'} for u in proteins if nonempty(u['Function [CC]'])], 'go_biological_process': sorted({term.strip() for u in proteins for term in u['Gene Ontology (biological process)'].split(';') if term.strip()}), 'protein_names': sorted({u['Protein names'] for u in proteins if nonempty(u['Protein names'])}), 'taxon': 9606, 'perturbation_response_access': False})
    write_json(root / 'gene_identities.json', rows)
    print(json.dumps({'event': 'gene_identity', 'genes': len(rows), 'ncbi_ids': sum((r['ncbi_gene_id'] is not None for r in rows)), 'human_uniprot_function': sum((bool(r['uniprot_function_texts']) for r in rows))}), flush=True)
    ids = sorted({r['ncbi_gene_id'] for r in rows if r['ncbi_gene_id']}, key=int)
    summaries = {}
    retrieval_failures = []

    def fetch_batch(batch):
        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?' + urllib.parse.urlencode({'db': 'gene', 'id': ','.join(batch), 'retmode': 'json'})
        try:
            data = client.json(url)
        except (OSError, RuntimeError, ValueError) as error:
            if len(batch) > 1:
                midpoint = len(batch) // 2
                fetch_batch(batch[:midpoint])
                fetch_batch(batch[midpoint:])
            else:
                retrieval_failures.append({'gene_id': batch[0], 'url': url, 'error': str(error)[:300]})
            return
        assert 'result' in data and 'error' not in data
        result = data['result']
        assert set(result['uids']) == set(batch)
        for gid in batch:
            value = result[gid]
            assert str(value['organism']['taxid']) == '9606'
            summaries[gid] = {'gene_id': gid, 'name': value.get('name'), 'description': value.get('description'), 'summary': value.get('summary', ''), 'status': value.get('status'), 'current_id': value.get('currentid'), 'source_url': 'https://www.ncbi.nlm.nih.gov/gene/' + gid, 'snapshot_url': url}
    for start in range(0, len(ids), 100):
        fetch_batch(ids[start:start + 100])
        if start % 500 == 0:
            print(json.dumps({'event': 'ncbi_gene_summary', 'done': min(start + 100, len(ids)), 'total': len(ids)}), flush=True)
    write_json(root / 'ncbi_gene_summaries.json', summaries)
    write_json(root / 'ncbi_retrieval_failures.json', retrieval_failures)
    legacy = pd.read_parquet(ENTITY_SOURCE_ROOT / 'entity_registry.parquet')
    legacy = legacy[legacy.entity_type.eq('gene')].copy()
    legacy['ensembl_id'] = legacy.ensembl_id.map(storage_strings)
    legacy = legacy.explode('ensembl_id').set_index('ensembl_id')
    mismatches = []
    for r in rows:
        s = summaries.get(r['ncbi_gene_id'], {})
        r['ncbi_summary'] = s.get('summary') or None
        r['ncbi_summary_url'] = s.get('source_url')
        r['functional_text_available'] = bool(r['uniprot_function_texts'] or nonempty(r['ncbi_summary']))
        r['descriptive_text_available'] = bool(r['functional_text_available'] or nonempty(r['gene_description']))
        r['functional_text_status'] = 'human_source_functional_description' if r['functional_text_available'] else 'identity_description_only' if r['descriptive_text_available'] else 'no_curated_external_description_found'
        if r['gene'] in legacy.index:
            group = legacy.loc[[r['gene']]]
            legacyacc = {a for value in group.uniprot_accession for a in storage_strings(value)}
            accepted = set(r['uniprot_accessions'])
            if accepted and legacyacc and (not accepted & legacyacc):
                mismatches.append({'gene': r['gene'], 'symbol': r['gene_symbol'], 'legacy_accessions': sorted(legacyacc), 'human_verified_accessions': sorted(accepted), 'status': 'legacy_accession_not_the_curated_human_gene_accession'})
    write_json(root / 'gene_cards.json', rows)
    write_json(root / 'legacy_accession_audit.json', mismatches)
    coverage = {}
    for role in ['core', 'extra']:
        group = [r for r in rows if r['role'] == role]
        coverage[role] = {'genes': len(group), 'ncbi_identity': sum((r['ncbi_gene_id'] is not None for r in group)), 'human_uniprot_function': sum((bool(r['uniprot_function_texts']) for r in group)), 'ncbi_summary': sum((nonempty(r['ncbi_summary']) for r in group)), 'functional_text': sum((r['functional_text_available'] for r in group)), 'descriptive_text': sum((r['descriptive_text_available'] for r in group))}
    receipt = {'status': 'HUMAN_GENE_BACKGROUND_COMPILED', 'coverage': coverage, 'legacy_accession_mismatches': len(mismatches), 'hgnc_approved_records': len(hgnc), 'excluded_non_9606_ncbi_rows': excluded_taxa, 'ncbi_retrieval_failed_ids': len(retrieval_failures), 'http_attempts_this_invocation': client.calls, 'llm_calls': 0, 'expression_response_queries': 0, 'source_sha256': {str(p): sha256(p) for p in [npath, hpath, upath, ENTITY_SOURCE_ROOT / 'entity_registry.parquet']}, 'output_sha256': {name: sha256(root / name) for name in ['gene_identities.json', 'gene_cards.json', 'ncbi_gene_summaries.json', 'ncbi_retrieval_failures.json', 'legacy_accession_audit.json']}}
    write_json(root / 'manifest.json', receipt)
    print(json.dumps({'event': 'gene_background_complete', 'coverage': coverage, 'legacy_accession_mismatches': len(mismatches)}), flush=True)
