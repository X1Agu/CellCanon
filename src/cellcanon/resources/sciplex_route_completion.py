from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import pandas as pd
from cellcanon.resources.sciplex_knowledge import OUTPUT, sha256, write_json
from cellcanon.resources.sciplex_gtopdb_review import PARENT, DESTINATION
from cellcanon.resources.sciplex_layer5_coverage import TIERS, validate_witness
CELLS = ('A549', 'K562', 'MCF7')
UNSIGNED = ('direct_target', 'directed', 'shared_pathway', 'shared_pathway_max200')

def identifier(prefix, value):
    return prefix + ':' + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:24]

def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value

def bounded_paths(start, adjacency, *, signed=False, allowed=None):
    if allowed is not None and start not in allowed:
        return {}
    found = {}
    frontier = [(start, 1, (start,), ())]
    shortest = {(start, 1): 0}
    for depth in range(1, 4):
        next_states = defaultdict(list)
        for node, sign, nodes, edges in frontier:
            for edge in adjacency.get(node, []):
                other = edge['target_node_id']
                if other in nodes or (allowed is not None and other not in allowed):
                    continue
                if signed and edge['sign'] not in {-1, 1}:
                    continue
                net = sign * int(edge['sign']) if signed else 1
                ns = (*nodes, other)
                es = (*edges, edge['edge_id'])
                found.setdefault((other, net, edge['interaction_type'] == 'transcriptional'), {'nodes': ns, 'edge_ids': es, 'sign': net if signed else None})
                key = (other, net)
                if depth == 3 or shortest.get(key, 4) < depth:
                    continue
                values = next_states[key]
                if not values or (depth == 2 and len(values) < 2 and (values[0][2][1] != ns[1])):
                    values.append((other, net, ns, es))
                    shortest[key] = depth
        frontier = [entry for values in next_states.values() for entry in values]
    return found

def signed_seeds(seeds):
    groups = defaultdict(list)
    for s in seeds:
        if s['action_sign'] in {-1, 1}:
            groups[s['drug'], s['symbol']].append(s)
    accepted = []
    conflicts = []
    for (drug, symbol), values in sorted(groups.items()):
        signs = {s['action_sign'] for s in values}
        if len(signs) > 1:
            conflicts.append({'drug': drug, 'symbol': symbol, 'signs': sorted(signs), 'evidence_ids': sorted((s['evidence_id'] for s in values))})
        else:
            accepted.append(sorted(values, key=lambda s: (s.get('source_kind') != 'gtopdb_primary_functional_action', s['evidence_id']))[0])
    return (accepted, conflicts)

class Context:

    def __init__(self):
        con = sqlite3.connect('file:' + str(OUTPUT / 'store/knowledge.sqlite') + '?mode=ro', uri=True)
        self.cards = [json.loads(r[0]) for r in con.execute('SELECT payload FROM drugs ORDER BY drug')]
        self.genes = [json.loads(r[0]) for r in con.execute('SELECT payload FROM genes ORDER BY role,gene')]
        self.network = pd.read_sql_query('SELECT * FROM regulatory_network', con)
        complexes = pd.read_sql_query('SELECT * FROM complex_components', con)
        baseline = pd.read_sql_query('SELECT * FROM cell_baseline', con)
        con.close()
        hp = OUTPUT / 'public_background/http' / (hashlib.sha256(b'https://rest.genenames.org/fetch/status/Approved').hexdigest() + '.json')
        hgnc = json.loads(hp.read_text())['response']['docs']
        primary = {h['symbol'] for h in hgnc}
        aliases = defaultdict(set)
        by_ens = defaultdict(set)
        for h in hgnc:
            if h.get('ensembl_gene_id'):
                by_ens[h['ensembl_gene_id']].add(h['symbol'])
            for sym in [*h.get('alias_symbol', []), *h.get('prev_symbol', [])]:
                aliases[sym].add(h['symbol'])

        def resolve(symbol):
            if symbol in primary:
                return symbol
            values = aliases.get(symbol, set())
            return next(iter(values)) if len(values) == 1 else None
        self.roles = {g['gene']: g['role'] for g in self.genes}
        self.symbols = {}
        self.queries = defaultdict(set)
        for g in self.genes:
            exact = by_ens.get(g['gene'], set())
            symbol = next(iter(exact)) if len(exact) == 1 else None
            if symbol is None and len(g['candidate_ncbi_ids']) <= 1 and (g['identity_status'] != 'unresolved_external_identity'):
                symbol = resolve(g['canonical_symbol'])
            if symbol:
                self.symbols[g['gene']] = symbol
                self.queries[symbol].add(g['gene'])
        eligible = self.network.loc[self.network.typed_directed_usable.eq(1) & self.network.reference_count.gt(0)]
        self.edges = eligible.set_index('edge_id').to_dict('index')
        self.adjacency = defaultdict(list)
        for edge in eligible.sort_values(['source_node_id', 'target_node_id', 'reference_count', 'edge_id'], ascending=[True, True, False, True]).to_dict('records'):
            self.adjacency[edge['source_node_id']].append(edge)
        detections = defaultdict(lambda: defaultdict(list))
        for r in baseline.to_dict('records'):
            exact = by_ens.get(r['ensembl_id'], set())
            symbol = next(iter(exact)) if len(exact) == 1 else resolve(r['gene_symbol'])
            if symbol:
                detections[symbol][r['cell_line']].append(r)
        self.components = complexes.groupby('complex_id').canonical_symbol.agg(list).to_dict()
        nodes = set(eligible.source_node_id) | set(eligible.target_node_id) | {'GENE:' + s for s in detections}
        self.node_context = {}
        self.allowed = {cell: set() for cell in CELLS}
        for node in sorted(nodes):
            parts = [node[5:]] if node.startswith('GENE:') else self.components.get(node, [])
            for cell in CELLS:
                known = bool(parts) and all((len(detections.get(p, {}).get(cell, [])) == 1 for p in parts))
                records = [detections[p][cell][0] for p in parts] if known else []
                value = min((float(r['control_detection_fraction']) for r in records)) if records else None
                self.node_context[node, cell] = {'node': node, 'cell': cell, 'components': parts, 'minimum_detection': value, 'status': 'eligible' if value is not None and value >= 0.01 else 'below_threshold' if value is not None else 'unknown_identity_or_measurement', 'measurements': records, 'semantics': 'vehicle_expression_compatibility_not_target_engagement'}
                if value is not None and value >= 0.01:
                    self.allowed[cell].add(node)
        self.bysymbol = defaultdict(dict)
        self.bypath = defaultdict(dict)
        self.pathsize = defaultdict(set)
        member = pd.read_parquet(OUTPUT / 'network_background/reactome_memberships.parquet')
        ensmember = pd.read_parquet(OUTPUT / 'network_background/reactome_ensembl_memberships.parquet')
        for r in member.loc[member.canonical_symbol.notna()].itertuples():
            ref = {'pathway_id': r.pathway_id, 'source_url': 'https://reactome.org/PathwayBrowser/#/' + r.pathway_id, 'member': r.canonical_symbol, 'snapshot': 'Reactome97'}
            self.bysymbol[r.canonical_symbol].setdefault(r.pathway_id, ref)
            self.pathsize[r.pathway_id].add(r.canonical_symbol)
            for gene in sorted(self.queries.get(r.canonical_symbol, [])):
                self.bypath[r.pathway_id].setdefault(gene, ref)
        for r in ensmember.itertuples():
            ref = {'pathway_id': r.pathway_id, 'source_url': r.source_url, 'member': r.ensembl_id, 'snapshot': r.source_snapshot}
            if r.ensembl_id in self.roles:
                self.bypath[r.pathway_id].setdefault(r.ensembl_id, ref)
            exact = by_ens.get(r.ensembl_id, set())
            if len(exact) == 1:
                symbol = next(iter(exact))
                self.bysymbol[symbol].setdefault(r.pathway_id, ref)
                self.pathsize[r.pathway_id].add(symbol)
        self.path_cache = {}

    def paths(self, symbol, *, signed=False, cell=''):
        key = (symbol, signed, cell)
        if key not in self.path_cache:
            self.path_cache[key] = bounded_paths('GENE:' + symbol, self.adjacency, signed=signed, allowed=self.allowed[cell] if cell else None)
        return self.path_cache[key]

def witness(seed, gene, tier, nodes, edges=(), references=(), sign=None, cell=''):
    return {'drug': seed['drug'], 'gene': gene, 'tier': tier, 'seed_evidence_id': seed['evidence_id'], 'seed_tier': seed['tier'], 'seed_source_url': seed['source_url'], 'identity_tier': seed['identity_tier'], 'nodes': list(nodes), 'network_edge_ids': list(edges), 'pathway_references': list(references), 'net_sign': sign, 'cell': cell, 'condition_effect_status': 'hypothesis'}

def key_for(w):
    return (w['drug'], w['gene'], w['tier'], w['net_sign'], w.get('cell', ''))

def collect_signed(ctx, seeds):
    accepted, conflicts = signed_seeds(seeds)
    routes = {}
    for i, s in enumerate(accepted):
        for cell in ('', *CELLS):
            for (node, net, transcription), path in ctx.paths(s['symbol'], signed=True, cell=cell).items():
                if not node.startswith('GENE:'):
                    continue
                for gene in sorted(ctx.queries.get(node[5:], [])):
                    sign = s['action_sign'] * net
                    if not cell:
                        w = witness(s, gene, 'signed_network', path['nodes'], path['edge_ids'], sign=sign)
                        routes.setdefault(key_for(w), w)
                    if transcription:
                        w = witness(s, gene, 'signed_transcript', path['nodes'], path['edge_ids'], sign=sign, cell=cell)
                        routes.setdefault(key_for(w), w)
        if (i + 1) % 40 == 0:
            print(json.dumps({'event': 'signed_target_contexts', 'done': i + 1, 'total': len(accepted)}), flush=True)
    return (routes, conflicts)

def summarize(ctx, routes, seeds):
    per_drug = defaultdict(lambda: defaultdict(set))
    signs = defaultdict(set)
    cellgenes = defaultdict(set)
    cellsigns = defaultdict(set)
    for w in routes.values():
        role = ctx.roles[w['gene']]
        cell = w.get('cell', '')
        if cell:
            cellgenes[w['drug'], role, cell].add(w['gene'])
            cellsigns[w['drug'], w['gene'], cell].add(w['net_sign'])
        else:
            per_drug[w['drug'], role][w['tier']].add(w['gene'])
            if w['tier'] == 'signed_transcript':
                signs[w['drug'], w['gene']].add(w['net_sign'])
    rows = []
    conditions = []
    seed_by_drug = defaultdict(list)
    signed, _ = signed_seeds(seeds)
    for s in seeds:
        seed_by_drug[s['drug']].append(s)
    signed_drugs = {s['drug'] for s in signed}
    for d in ctx.cards:
        for role in ('core', 'extra'):
            fields = per_drug[d['drug'], role]
            fields['citable_connection'] = set().union(*(fields[t] for t in ['direct_target', 'directed', 'shared_pathway']))
            denominator = sum((r == role for r in ctx.roles.values()))
            local = seed_by_drug[d['drug']]
            rows.append({'drug': d['drug'], 'split': d['split'], 'role': role, 'gene_denominator': denominator, **{t: len(fields[t]) for t in TIERS}, 'has_active_assay_seed': any((s['tier'] == 'active_assay' for s in local)), 'has_curated_moa_seed': any((s['tier'] == 'curated_moa' for s in local)), 'has_signed_curated_seed': d['drug'] in signed_drugs, 'opposing_signed_transcript_pairs': sum((len(signs[d['drug'], g]) > 1 for g in fields['signed_transcript']))})
            for cell in CELLS:
                gs = cellgenes[d['drug'], role, cell]
                conditions.append({'drug': d['drug'], 'split': d['split'], 'role': role, 'cell': cell, 'gene_denominator': denominator, 'signed_transcript_pairs': len(gs), 'opposing_signed_transcript_pairs': sum((len(cellsigns[d['drug'], g, cell]) > 1 for g in gs))})
    frame = pd.DataFrame(rows)
    cframe = pd.DataFrame(conditions)
    summary = {}
    for (split, role), group in frame.groupby(['split', 'role']):
        summary[split + ':' + role] = {'drug_denominator': len(group), 'gene_denominator': int(group.gene_denominator.iloc[0]), 'pair_denominator': int(group.gene_denominator.sum()), **{t: {'drugs': int(group[t].gt(0).sum()), 'pairs': int(group[t].sum())} for t in TIERS}, 'opposing_signed_transcript_pairs': int(group.opposing_signed_transcript_pairs.sum())}
    csummary = cframe.groupby(['split', 'role', 'cell']).agg(conditions=('drug', 'size'), pairs=('signed_transcript_pairs', 'sum'), nonempty_conditions=('signed_transcript_pairs', lambda x: int(x.gt(0).sum())), opposing_pairs=('opposing_signed_transcript_pairs', 'sum')).reset_index().to_dict('records')
    return (frame, cframe, summary, csummary, signs, cellsigns)

def build(destination=DESTINATION):
    destination.mkdir(parents=True, exist_ok=True)
    root = destination / 'routes'
    root.mkdir(parents=True, exist_ok=True)
    parent = json.loads((PARENT / 'manifest.json').read_text())
    review_root = destination / 'gtopdb_review'
    review = json.loads((review_root / 'manifest.json').read_text())
    inputs = {}
    for path, expected in parent['source_sha256'].items():
        assert sha256(Path(path)) == expected, path
        inputs[path] = expected
    for name, expected in parent['output_sha256'].items():
        assert sha256(PARENT / name) == expected, name
        inputs[str(PARENT / name)] = expected
    for name, expected in review['output_sha256'].items():
        assert sha256(review_root / name) == expected, name
        inputs[str(review_root / name)] = expected
    for mp in [PARENT / 'manifest.json', review_root / 'manifest.json']:
        inputs[str(mp)] = sha256(mp)
    ctx = Context()
    old_seeds = json.loads((PARENT / 'seed_evidence.json').read_text())
    new_seeds = json.loads((review_root / 'accepted_seeds.json').read_text())
    all_seeds = old_seeds + new_seeds
    seed_lookup = {(s['drug'], s['evidence_id']): s for s in all_seeds}
    unsigned = {}
    with (PARENT / 'route_witnesses.jsonl').open() as f:
        for line in f:
            w = json.loads(line)
            if w['tier'] in UNSIGNED:
                w['cell'] = ''
                w['condition_effect_status'] = 'hypothesis'
                unsigned[key_for(w)] = w
    base_signed, base_conflicts = collect_signed(ctx, old_seeds)
    _, _, base_summary, base_conditions, _, _ = summarize(ctx, {**unsigned, **base_signed}, old_seeds)
    del base_signed
    chosen = {}
    for s in new_seeds:
        chosen.setdefault((s['drug'], s['symbol']), s)
    for s in chosen.values():
        symbol = s['symbol']
        for gene in sorted(ctx.queries.get(symbol, [])):
            w = witness(s, gene, 'direct_target', [s['node']])
            unsigned.setdefault(key_for(w), w)
        for (node, _, _), path in ctx.paths(symbol).items():
            if node.startswith('GENE:'):
                for gene in sorted(ctx.queries.get(node[5:], [])):
                    w = witness(s, gene, 'directed', path['nodes'], path['edge_ids'])
                    unsigned.setdefault(key_for(w), w)
        for pid, ref in sorted(ctx.bysymbol.get(symbol, {}).items(), key=lambda item: (len(ctx.pathsize[item[0]]), item[0])):
            for gene, other in ctx.bypath.get(pid, {}).items():
                w = witness(s, gene, 'shared_pathway', [s['node'], pid, ctx.symbols.get(gene, gene)], references=[ref, other])
                unsigned.setdefault(key_for(w), w)
                if len(ctx.pathsize[pid]) <= 200:
                    w = {**w, 'tier': 'shared_pathway_max200'}
                    unsigned.setdefault(key_for(w), w)
    signed, conflicts = collect_signed(ctx, all_seeds)
    routes = {**unsigned, **signed}
    frame, cframe, summary, conditions, signs, cellsigns = summarize(ctx, routes, all_seeds)
    for split_role, old in parent['coverage'].items():
        for tier in UNSIGNED:
            assert base_summary[split_role][tier] == old[tier], (split_role, tier)
    write_json(root / 'base_with_search_repairs.json', {'coverage': base_summary, 'condition_coverage': base_conditions})
    write_json(root / 'seed_action_conflicts.json', conflicts)
    write_json(root / 'seed_evidence.json', all_seeds)
    frame.to_parquet(root / 'per_drug_role_coverage.parquet', index=False)
    cframe.to_parquet(root / 'per_condition_coverage.parquet', index=False)
    temp = root / 'routes.building.sqlite'
    if temp.exists():
        temp.unlink()
    con = sqlite3.connect(temp)
    con.executescript('\n      CREATE TABLE routes(route_id TEXT PRIMARY KEY,drug TEXT,gene TEXT,role TEXT,tier TEXT,net_sign INTEGER,cell TEXT,edge_count INTEGER,payload TEXT);\n      CREATE INDEX route_query ON routes(drug,role,cell,gene,tier);\n      CREATE TABLE evidence(evidence_id TEXT PRIMARY KEY,kind TEXT,drug TEXT,payload TEXT);\n      CREATE TABLE genes(gene TEXT PRIMARY KEY,symbol TEXT,role TEXT);\n      CREATE TABLE node_context(node TEXT,cell TEXT,payload TEXT,PRIMARY KEY(node,cell));\n      CREATE TABLE coverage(drug TEXT,role TEXT,payload TEXT,PRIMARY KEY(drug,role));\n      CREATE TABLE condition_coverage(drug TEXT,role TEXT,cell TEXT,payload TEXT,PRIMARY KEY(drug,role,cell));\n      CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);\n    ')
    reviews = {r['evidence_id']: r for r in json.loads((review_root / 'review_records.json').read_text())}
    background = sqlite3.connect('file:' + str(OUTPUT / 'store/knowledge.sqlite') + '?mode=ro', uri=True)
    background.row_factory = sqlite3.Row
    bindings = {r['evidence_id']: dict(r) for r in background.execute('SELECT * FROM binding_evidence')}
    drugcentral = [dict(r) for r in background.execute('SELECT * FROM drugcentral_moa')]
    control_manifest = OUTPUT / 'cell_baseline/manifest.json'
    control_hash = sha256(control_manifest)
    evidence = {}
    for s in all_seeds:
        assert s['evidence_id'] not in evidence or evidence[s['evidence_id']]['drug'] == s['drug'], 'cross-drug evidence identifier collision'
        source = reviews.get(s['evidence_id']) or bindings.get(s['evidence_id'])
        if source is None and s['evidence_id'].startswith('DrugCentral:row:'):
            source = drugcentral[int(s['evidence_id'].rsplit(':', 1)[-1])]
            assert source['drug'] == s['drug']
        if source is None:
            row = background.execute('SELECT payload FROM documents WHERE evidence_id=?', (s['evidence_id'],)).fetchone()
            assert row is not None, s['evidence_id']
            source = json.loads(row[0])
        predicate = 'measured_target_binding' if s['tier'] == 'active_assay' else 'functional_target_action' if s['action_sign'] is not None else 'curated_target_relation_unknown_action'
        evidence[s['evidence_id']] = {'evidence_id': s['evidence_id'], 'kind': s.get('source_kind', 'background_target_relation'), 'drug': s['drug'], 'subject': s['drug'], 'predicate': predicate, 'object': s['node'], 'action_sign': s['action_sign'], 'source_url': s['source_url'], 'identity_tier': s['identity_tier'], 'condition_effect_verified': False, 'source_record': source}
    background.close()
    checks = {'route_witnesses': 0, 'cell_witnesses': 0, 'opposing_polarities_preserved': 0}
    for w in routes.values():
        validate_witness(w, ctx.edges, seed_lookup, ctx.symbols)
        checks['route_witnesses'] += 1
        refs = [w['seed_evidence_id']]
        for eid in w['network_edge_ids']:
            key = 'network:' + eid
            refs.append(key)
            if key not in evidence:
                e = ctx.edges[eid]
                evidence[key] = {'evidence_id': key, 'kind': 'background_network_relation', 'drug': None, 'subject': e['source_node_id'], 'predicate': e['interaction_type'], 'object': e['target_node_id'], 'action_sign': int(e['sign']) if e['sign'] in {-1, 1} else None, 'source_url': 'https://omnipathdb.org/interactions?organisms=9606', 'pubmed_ids': [r.rsplit(':', 1)[-1] for r in json.loads(e['references_json']) if r.rsplit(':', 1)[-1].isdigit()], 'condition_effect_verified': False, 'source_record': e}
        for ref in w['pathway_references']:
            key = identifier('reactome_membership', ref)
            refs.append(key)
            evidence.setdefault(key, {'evidence_id': key, 'kind': 'pathway_membership', 'drug': None, 'subject': ref['member'], 'predicate': 'member_of', 'object': ref['pathway_id'], 'action_sign': None, 'source_url': ref['source_url'], 'condition_effect_verified': False, 'source_record': ref})
        if w['cell']:
            values = [ctx.node_context[n, w['cell']] for n in w['nodes']]
            assert all((v['status'] == 'eligible' for v in values))
            w['minimum_control_detection'] = min((v['minimum_detection'] for v in values))
            w['cell_compatibility'] = 'eligible'
            checks['cell_witnesses'] += 1
            for value in values:
                for measurement in value['measurements']:
                    key = identifier('control_baseline', [measurement['ensembl_id'], w['cell']])
                    refs.append(key)
                    evidence.setdefault(key, {'evidence_id': key, 'kind': 'vehicle_control_expression', 'drug': None, 'subject': measurement['ensembl_id'], 'predicate': 'vehicle_expression', 'object': w['cell'], 'action_sign': None, 'source_url': str(control_manifest), 'source_kind': 'local_control_artifact', 'source_sha256': control_hash, 'condition_effect_verified': False, 'source_record': measurement})
        else:
            w['cell_compatibility'] = 'not_conditioned'
        w['evidence_ids'] = list(dict.fromkeys(refs))
        w['role'] = ctx.roles[w['gene']]
        possible = cellsigns[w['drug'], w['gene'], w['cell']] if w['cell'] else signs[w['drug'], w['gene']]
        w['signed_transcript_polarities'] = sorted(possible)
        w['direction_ambiguous'] = len(possible) > 1
        w['route_id'] = identifier('route', [w['drug'], w['gene'], w['tier'], w['net_sign'], w['cell'], w['seed_evidence_id'], w['nodes'], w['network_edge_ids'], w['pathway_references']])
        con.execute('INSERT INTO routes VALUES(?,?,?,?,?,?,?,?,?)', (w['route_id'], w['drug'], w['gene'], w['role'], w['tier'], w['net_sign'], w['cell'], len(w['network_edge_ids']), json.dumps(w)))
    con.executemany('INSERT INTO evidence VALUES(?,?,?,?)', [(eid, e['kind'], e['drug'], json.dumps(clean_json(e), allow_nan=False)) for eid, e in evidence.items()])
    con.executemany('INSERT INTO genes VALUES(?,?,?)', [(g, ctx.symbols.get(g), r) for g, r in ctx.roles.items()])
    con.executemany('INSERT INTO node_context VALUES(?,?,?)', [(node, cell, json.dumps(clean_json(value), allow_nan=False)) for (node, cell), value in ctx.node_context.items()])
    con.executemany('INSERT INTO coverage VALUES(?,?,?)', [(r['drug'], r['role'], json.dumps(r)) for r in frame.to_dict('records')])
    con.executemany('INSERT INTO condition_coverage VALUES(?,?,?,?)', [(r['drug'], r['role'], r['cell'], json.dumps(r)) for r in cframe.to_dict('records')])
    info = {'version': 'sciplex-routes-v2-gtopdb-bounded3', 'source_resources': str(OUTPUT), 'parent_audit_sha256': sha256(PARENT / 'manifest.json'), 'gtopdb_review_sha256': sha256(review_root / 'manifest.json'), 'source_code_sha256': sha256(Path(__file__)), 'condition_causal_validation': 'not_evaluated', 'seed_action_conflicts': conflicts}
    con.executemany('INSERT INTO metadata VALUES(?,?)', [(k, json.dumps(v)) for k, v in info.items()])
    con.commit()
    assert con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    con.close()
    temp.replace(root / 'routes.sqlite')
    checks['opposing_polarities_preserved'] = sum((len(v) > 1 for v in signs.values())) + sum((len(v) > 1 for v in cellsigns.values()))
    write_json(root / 'checks.json', checks)
    result = {'status': 'SCIPLEX_ROUTES_COMPLETED_AND_INDEXED_BACKGROUND_HYPOTHESES', 'coverage': summary, 'condition_coverage': conditions, 'frozen_parent_coverage': parent['coverage'], 'base_with_search_repairs': base_summary, 'base_condition_coverage_with_search_repairs': base_conditions, 'gtopdb_review_summary': review['summary'], 'route_rows': len(routes), 'evidence_rows': len(evidence), 'seed_action_conflicts': len(conflicts), 'protocol': {'max_network_edges': 3, 'terminal_transcription_required_for_expression': True, 'search': 'exact_endpoint_sign_last_type_simple_paths_upto_3; alternate_middle_prefix_retained', 'cell_search': 'control_detection>=0.01_on_all_genes_and_all_complex_components_during_search', 'gtopdb': 'human_primary_single_gene_explicit_Type_Action_resolved_PubMed; functional_action_not_direct_binding_claim', 'opposing_paths': 'both signs indexed; no unique predicted direction asserted', 'complete_route': 'addressable evidence components; condition effect remains hypothesis'}, 'source_sha256': inputs, 'source_code_sha256': sha256(Path(__file__)), 'output_sha256': {p.name: sha256(p) for p in root.iterdir() if p.is_file() and p.name != 'manifest.json'}, 'llm_calls': 0, 'training_runs': 0, 'perturbation_response_label_reads': 0, 'condition_causal_validation': 'not_evaluated'}
    write_json(root / 'manifest.json', result)
    print(json.dumps({'status': result['status'], 'test_core': summary['test:core'], 'test_conditions': [r for r in conditions if r['split'] == 'test' and r['role'] == 'core']}), flush=True)
    return root
