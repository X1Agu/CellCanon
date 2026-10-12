from __future__ import annotations
import json
from cellcanon.resources.sciplex_knowledge import OUTPUT, nonempty, sha256, write_json
from cellcanon.resources.sciplex_public_background import PublicSnapshot

def run(output=OUTPUT):
    root = output / 'ensembl_background'
    root.mkdir(parents=True, exist_ok=True)
    client = PublicSnapshot(output / 'public_background/http')
    source = output / 'gene_background/gene_cards.json'
    genes = json.loads(source.read_text())
    url = 'https://rest.ensembl.org/lookup/id?species=homo_sapiens;object_type=gene;expand=0'
    records = {}
    failures = []

    def lookup(ids):
        try:
            values = client.post_json(url, {'ids': ids})
        except (OSError, RuntimeError, ValueError) as error:
            if len(ids) > 50:
                middle = len(ids) // 2
                lookup(ids[:middle])
                lookup(ids[middle:])
            else:
                failures.append({'ids': ids, 'error': str(error)[:300]})
            return
        assert set(values) == set(ids)
        for gid, value in values.items():
            if value:
                assert value['id'] == gid and value['species'] == 'homo_sapiens' and (value['object_type'] == 'Gene')
            records[gid] = value
    for start in range(0, len(genes), 500):
        lookup([g['gene'] for g in genes[start:start + 500]])
        print(json.dumps({'event': 'ensembl_metadata', 'queried': min(start + 500, len(genes)), 'total': len(genes)}), flush=True)
    cards = []
    for g in genes:
        r = records.get(g['gene']) or {}
        cards.append({'gene': g['gene'], 'gene_symbol': g['gene_symbol'], 'role': g['role'], 'lookup_status': 'exact_current_human_gene' if r else 'not_returned_by_current_lookup' if g['gene'] in records else 'retrieval_failed', 'description': r.get('description'), 'display_name': r.get('display_name'), 'biotype': r.get('biotype'), 'assembly_name': r.get('assembly_name'), 'version': r.get('version'), 'species': r.get('species'), 'source_url': 'https://www.ensembl.org/Homo_sapiens/Gene/Summary?g=' + g['gene'], 'adds_descriptive_coverage': bool(nonempty(r.get('description')) and (not g['descriptive_text_available'])), 'counted_as_functional_mechanism': False, 'output_gene_identity_changed': False})
    write_json(root / 'lookup_records.json', records)
    write_json(root / 'gene_metadata.json', cards)
    write_json(root / 'retrieval_failures.json', failures)
    coverage = {}
    for role in ['core', 'extra']:
        rows = [r for r in cards if r['role'] == role]
        coverage[role] = {'genes': len(rows), 'exact_current_identity': sum((r['lookup_status'] == 'exact_current_human_gene' for r in rows)), 'descriptions': sum((nonempty(r['description']) for r in rows)), 'additional_descriptive_genes': sum((r['adds_descriptive_coverage'] for r in rows))}
    write_json(root / 'manifest.json', {'status': 'ENSEMBL_METADATA_COMPLETE' if not failures else 'ENSEMBL_METADATA_WITH_RETRIEVAL_FAILURES', 'coverage': coverage, 'failed_batches': len(failures), 'http_attempts': client.calls, 'source_sha256': {str(source): sha256(source)}, 'documentation': 'https://rest.ensembl.org/documentation/info/lookup_post', 'snapshot_release': 'API_release_unverified', 'output_sha256': {name: sha256(root / name) for name in ['lookup_records.json', 'gene_metadata.json', 'retrieval_failures.json']}})
    print(json.dumps({'coverage': coverage, 'failed_batches': len(failures)}), flush=True)
