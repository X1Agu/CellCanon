from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import re
import pandas as pd
from cellcanon.resources.sciplex_knowledge import LEAF, OUTPUT, NETWORK_SOURCE_ROOT, sha256, write_json
from cellcanon.resources.sciplex_public_background import PublicSnapshot
from cellcanon.resources.sciplex_drug_background import standard_parent_key
DRUGCENTRAL_ROOT = LEAF / 'drugcentral'

def parse_go_term(term):
    return re.fullmatch(r'(.*?)\s*\[(GO:\d+)\]', term.strip())

def run(output=OUTPUT):
    root = output / 'network_background'
    root.mkdir(parents=True, exist_ok=True)
    client = PublicSnapshot(output / 'public_background/http')
    hp = client.fetch('https://rest.genenames.org/fetch/status/Approved')
    hs = json.loads(hp.read_text())['response']['docs']
    primary = {h['symbol'] for h in hs}
    aliases = defaultdict(set)
    for h in hs:
        for alias in [h['symbol'], *h.get('alias_symbol', []), *h.get('prev_symbol', [])]:
            aliases[alias].add(h['symbol'])

    def resolve(symbol):
        if symbol in primary:
            return symbol
        a = aliases.get(symbol, set())
        return next(iter(a)) if len(a) == 1 else None
    upath = output / 'public_background/uniprot_human_reviewed.parquet'
    uniprot = pd.read_parquet(upath)
    upmap = {r['Entry']: r['Gene Names (primary)'] for r in uniprot.to_dict('records') if len(r['Gene Names (primary)'].split()) == 1}
    manifest = json.loads((NETWORK_SOURCE_ROOT / 'network_manifest.json').read_text())
    inputs = {str(hp): sha256(hp), str(upath): sha256(upath)}
    tables = {}
    for name in ['reactome_memberships', 'signed_network']:
        item = next((v for v in manifest['outputs'] if v['name'] == name))
        path = NETWORK_SOURCE_ROOT / Path(item['relative_path']).name
        assert sha256(path) == item['sha256']
        inputs[str(path)] = item['sha256']
        tables[name] = pd.read_parquet(path)
    members = tables['reactome_memberships'].copy()
    members['canonical_symbol'] = members.gene_symbol.map(resolve)
    members['mapping_status'] = members.canonical_symbol.notna().map({True: 'approved_human_symbol_or_unique_alias', False: 'unresolved_symbol'})
    members['relation_semantics'] = 'pathway_membership_not_direction_or_activation'
    members.to_parquet(root / 'reactome_memberships.parquet', index=False)
    network = tables['signed_network'].copy()
    for end in ['source', 'target']:
        symbols = network[end + '_gene_symbol'].map(resolve)
        accessions = network[end + '_uniprot'].map(upmap).map(resolve)
        conflict = symbols.notna() & accessions.notna() & symbols.ne(accessions)
        network[end + '_canonical_symbol'] = accessions.fillna(symbols).mask(conflict)
        network[end + '_identity_conflict'] = conflict
    network['human_symbols_resolved'] = network.source_canonical_symbol.notna() & network.target_canonical_symbol.notna()
    network['directed_usable'] = network.human_symbols_resolved & network.consensus_direction
    network['signed_usable'] = network.directed_usable & network.sign.isin([-1, 1])
    complex_rows = []
    for end in ['source', 'target']:
        node_ids = []
        node_types = []
        for record in network[[end + '_canonical_symbol', end + '_gene_symbol', end + '_uniprot']].to_dict('records'):
            symbol = record[end + '_canonical_symbol']
            accession = record[end + '_uniprot']
            if pd.notna(symbol):
                node_ids.append('GENE:' + symbol)
                node_types.append('gene')
                continue
            raw_symbol = record[end + '_gene_symbol']
            parts = [resolve(s) for s in str(raw_symbol).split('_')]
            if str(accession).startswith('COMPLEX:') and len(parts) > 1 and all(parts):
                node_ids.append(accession)
                node_types.append('complex')
                complex_rows.extend(({'complex_id': accession, 'canonical_symbol': s, 'semantics': 'component_of_complex_not_pairwise_regulatory_edge'} for s in sorted(set(parts))))
            else:
                node_ids.append(None)
                node_types.append('unresolved')
        network[end + '_node_id'] = node_ids
        network[end + '_node_type'] = node_types
    network['typed_nodes_resolved'] = network.source_node_id.notna() & network.target_node_id.notna()
    network['typed_directed_usable'] = network.typed_nodes_resolved & network.consensus_direction
    network['typed_signed_usable'] = network.typed_directed_usable & network.sign.isin([-1, 1])
    network['condition_effect_status'] = 'not_observed_in_sciplex_condition'
    network.to_parquet(root / 'regulatory_network.parquet', index=False)
    complexes = pd.DataFrame(complex_rows).drop_duplicates()
    complexes.to_parquet(root / 'complex_components.parquet', index=False)
    eurl = 'https://reactome.org/download/current/Ensembl2Reactome.txt'
    ep = client.fetch(eurl, suffix='tsv', timeout=60)
    inputs[str(ep)] = sha256(ep)
    ens_frames = []
    columns = ['ensembl_id', 'pathway_id', 'source_url', 'pathway_name', 'evidence_code', 'species']
    for chunk in pd.read_csv(ep, sep='\t', header=None, names=columns, dtype=str, chunksize=200000):
        ens_frames.append(chunk[chunk.species.eq('Homo sapiens') & chunk.ensembl_id.str.startswith('ENSG')])
    ens_members = pd.concat(ens_frames, ignore_index=True).drop_duplicates()
    assert len(ens_members) > 10000
    ens_members['source_snapshot'] = 'Reactome_API_release_unverified'
    ens_members['semantics'] = 'pathway_membership_not_direction_or_activation'
    ens_members.to_parquet(root / 'reactome_ensembl_memberships.parquet', index=False)
    go = []
    for u in uniprot.to_dict('records'):
        symbol = resolve(u['Gene Names (primary)'])
        if not symbol:
            continue
        for column, aspect in [('Gene Ontology (biological process)', 'biological_process'), ('Gene Ontology (molecular function)', 'molecular_function'), ('Gene Ontology (cellular component)', 'cellular_component')]:
            for term in u[column].split(';'):
                match = parse_go_term(term)
                if match:
                    go.append({'canonical_symbol': symbol, 'uniprot_accession': u['Entry'], 'go_id': match[2], 'term_name': match[1], 'aspect': aspect, 'source_url': 'https://www.uniprot.org/uniprotkb/' + u['Entry'] + '/entry', 'semantics': 'functional_annotation_not_measured_condition_response', 'taxon': 9606})
    pd.DataFrame(go).drop_duplicates().to_parquet(root / 'go_memberships.parquet', index=False)
    dcroot = DRUGCENTRAL_ROOT
    dcm = json.loads((dcroot / 'drugcentral_manifest.json').read_text())
    dc = {}
    for name in ['source_structures', 'source_interactions']:
        item = next((v for v in dcm['outputs'] if v['name'] == name))
        path = dcroot / Path(item['relative_path']).name
        assert sha256(path) == item['sha256']
        dc[name] = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
        inputs[str(path)] = item['sha256']
    structures = dc['source_structures']
    bykey = defaultdict(list)
    for row in structures.to_dict('records'):
        key = standard_parent_key(row['SMILES'])
        if key:
            bykey[key].append(row)
    drugs = pd.read_parquet(output / 'local_audit/drug_coverage.parquet')
    maps = []
    for d in drugs.to_dict('records'):
        matched = bykey.get(standard_parent_key(d['canonical_smiles']), [])
        if len(matched) == 1:
            maps.append({'drug': d['drug'], 'split': d['split'], 'STRUCT_ID': matched[0]['ID'], 'identity_status': 'same_full_standard_parent_inchi', 'drugcentral_name': matched[0]['INN']})
    mapping = pd.DataFrame(maps)
    interactions = dc['source_interactions']
    moa = interactions[interactions.ORGANISM.eq('Homo sapiens') & interactions.MOA.eq('1') & interactions.GENE.ne('')]
    joined = mapping.merge(moa, on='STRUCT_ID', how='inner', validate='many_to_many')
    joined['source_release'] = 'DrugCentral_2021_09_01'
    joined['canonical_gene_symbol'] = joined.GENE.map(resolve)
    joined['evidence_semantics'] = 'curated_drug_mechanism_target_not_sciplex_response'
    mapping.to_parquet(root / 'drugcentral_identity.parquet', index=False)
    joined.to_parquet(root / 'drugcentral_moa.parquet', index=False)
    summary = {'status': 'PATHWAY_NETWORK_BACKGROUND_COMPILED', 'reactome_rows': len(members), 'reactome_resolved_rows': int(members.canonical_symbol.notna().sum()), 'network_rows': len(network), 'network_human_resolved_rows': int(network.human_symbols_resolved.sum()), 'network_directed_rows': int(network.directed_usable.sum()), 'network_signed_rows': int(network.signed_usable.sum()), 'network_identity_conflict_rows': int((network.source_identity_conflict | network.target_identity_conflict).sum()), 'network_typed_resolved_rows': int(network.typed_nodes_resolved.sum()), 'network_typed_directed_rows': int(network.typed_directed_usable.sum()), 'network_typed_signed_rows': int(network.typed_signed_usable.sum()), 'complex_nodes': complexes.complex_id.nunique(), 'complex_memberships': len(complexes), 'reactome_exact_ensembl_rows': len(ens_members), 'go_rows': len(go), 'drugcentral_identity_drugs': mapping.drug.nunique(), 'drugcentral_moa_drugs': joined.drug.nunique(), 'drugcentral_moa_test_drugs': joined.query("split == 'test'").drug.nunique(), 'source_sha256': inputs, 'output_sha256': {p.name: sha256(p) for p in root.glob('*.parquet')}, 'llm_calls': 0, 'response_expression_queries': 0, 'claim_boundary': 'Shared biological components, not complete causal drug-cell-gene paths.'}
    write_json(root / 'manifest.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ['source_sha256', 'output_sha256']}), flush=True)
