from __future__ import annotations
import copy
import json
from pathlib import Path
import sqlite3
from cellcanon.resources.sciplex_knowledge import OUTPUT, sha256, write_json
from cellcanon.resources.sciplex_knowledge_store import LocalKnowledge
from cellcanon.resources.sciplex_gtopdb_review import DESTINATION
from cellcanon.resources.sciplex_route_completion import CELLS, identifier
SCHEMA = 'sciplex-complete-route-condition-v1'

def wire(packet):
    return json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

class RouteKnowledge:

    def __init__(self, root=DESTINATION, *, cache_root=None):
        self.root = Path(root)
        path = self.root / 'routes/routes.sqlite'
        manifest = json.loads((self.root / 'routes/manifest.json').read_text())
        if sha256(path) != manifest['output_sha256']['routes.sqlite']:
            raise ValueError('Route index hash differs from its manifest.')
        background_path = OUTPUT / 'store/knowledge.sqlite'
        sources = manifest['source_sha256']
        expected = sources.get(str(background_path))
        if expected is None:
            matches = [value for name, value in sources.items()
                       if Path(name).parts[-2:] == ('store', 'knowledge.sqlite')]
            if len(matches) != 1:
                raise ValueError('Route manifest must identify one background knowledge database.')
            expected = matches[0]
        if sha256(background_path) != expected:
            raise ValueError('Background knowledge changed after route compilation.')
        self.snapshot = sha256(self.root / 'routes/manifest.json')
        self.background = LocalKnowledge()
        self.conn = sqlite3.connect('file:' + str(path.resolve()) + '?mode=ro', uri=True)
        self.conn.row_factory = sqlite3.Row
        self.cache_root = Path(cache_root) if cache_root is not None else self.root / 'condition_cache'
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.stats = {'cache_hits': 0, 'cache_misses': 0, 'cache_corruptions': 0, 'llm_calls': 0}
        self.conflicts = json.loads(self.conn.execute("SELECT value FROM metadata WHERE key='seed_action_conflicts'").fetchone()[0])

    def close(self):
        self.conn.close()
        self.background.close()

    def drug(self, name):
        return self.background.drug(name)

    def gene(self, name):
        return self.background.gene(name)

    def search(self, *args, **kwargs):
        return self.background.search(*args, **kwargs)

    def evidence(self, eid):
        row = self.conn.execute('SELECT payload FROM evidence WHERE evidence_id=?', (eid,)).fetchone()
        if row:
            return json.loads(row[0])
        e = self.background.evidence(eid)
        return {'evidence_id': eid, 'kind': e['kind'], 'drug': e['entity_id'] if str(e['entity_id']).startswith('S') else None, 'subject': e['entity_id'], 'predicate': 'described_in_source', 'object': e['source_url'], 'action_sign': None, 'source_url': e['source_url'], 'text': e['text'], 'source_record': json.loads(e['payload']), 'condition_effect_verified': False}

    def _query_genes(self, genes, role):
        if role not in {'core', 'extra'}:
            raise ValueError('Role must be core or extra.')
        if not genes:
            return None
        result = []
        for gene in genes:
            rows = self.conn.execute('SELECT gene FROM genes WHERE role=? AND (gene=? OR symbol=?)', (role, gene, gene)).fetchall()
            if len(rows) != 1:
                raise ValueError('Missing or ambiguous output gene: ' + str(gene))
            result.append(rows[0][0])
        return sorted(set(result))

    def coverage(self, drug, cell, role='core'):
        d = self.drug(drug)['drug']
        if cell not in CELLS or role not in {'core', 'extra'}:
            raise ValueError('Invalid SciPlex cell/role.')
        row = self.conn.execute('SELECT payload FROM coverage WHERE drug=? AND role=?', (d, role)).fetchone()
        condition = self.conn.execute('SELECT payload FROM condition_coverage WHERE drug=? AND role=? AND cell=?', (d, role, cell)).fetchone()
        return {'background': json.loads(row[0]), 'cell': json.loads(condition[0])}

    def routes(self, drug, cell, *, role='core', genes=None, limit=3):
        d = self.drug(drug)['drug']
        if cell not in CELLS:
            raise ValueError('Invalid SciPlex cell.')
        gene_ids = self._query_genes(genes, role)
        limit = min(20, max(1, int(limit)))
        where = 'drug=? AND role=? AND cell IN (?,?)'
        args = [d, role, '', cell]
        if gene_ids:
            where += ' AND gene IN (' + ','.join(('?' for _ in gene_ids)) + ')'
            args.extend(gene_ids)
        priority = "CASE tier WHEN 'signed_transcript' THEN 0 WHEN 'directed' THEN 2 WHEN 'direct_target' THEN 3 WHEN 'shared_pathway_max200' THEN 4 WHEN 'shared_pathway' THEN 5 ELSE 6 END"
        selected = [r[0] for r in self.conn.execute(f'SELECT gene FROM routes WHERE {where} GROUP BY gene ORDER BY MIN({priority}),MIN(edge_count),gene LIMIT ?', (*args, limit))]
        if gene_ids is None:
            ambiguous = self.conn.execute("SELECT gene FROM routes WHERE drug=? AND role=? AND cell='' AND tier='signed_transcript' GROUP BY gene HAVING COUNT(DISTINCT net_sign)>1 ORDER BY MIN(edge_count),gene LIMIT 1", (d, role)).fetchone()
            compatible = self.conn.execute("SELECT gene FROM routes WHERE drug=? AND role=? AND cell=? AND tier='signed_transcript' ORDER BY edge_count,gene LIMIT 1", (d, role, cell)).fetchone()
            priority_genes = [r[0] for r in [ambiguous, compatible] if r is not None]
            selected = list(dict.fromkeys([*priority_genes, *selected]))[:limit]
        groups = []
        for gene in selected:
            rows = [json.loads(r[0]) for r in self.conn.execute('SELECT payload FROM routes WHERE drug=? AND gene=? AND cell IN (?,?) ORDER BY edge_count,route_id', (d, gene, '', cell))]
            transcript = [r for r in rows if r['tier'] == 'signed_transcript']
            global_signs = sorted({r['net_sign'] for r in transcript if not r['cell']})
            cell_signs = sorted({r['net_sign'] for r in transcript if r['cell'] == cell})
            chosen = []
            if transcript:
                for sign in global_signs:
                    candidates = [r for r in transcript if r['net_sign'] == sign]
                    chosen.append(min(candidates, key=lambda r: (r['cell'] != cell, len(r['network_edge_ids']), r['route_id'])))
            else:
                order = {'directed': 0, 'direct_target': 1, 'shared_pathway_max200': 2, 'shared_pathway': 3, 'signed_network': 4}
                chosen = [min(rows, key=lambda r: (order[r['tier']], len(r['network_edge_ids']), r['route_id']))]
            compact = []
            for r in chosen:
                compact.append({k: r[k] for k in ['route_id', 'tier', 'net_sign', 'nodes', 'evidence_ids', 'identity_tier', 'cell_compatibility', 'condition_effect_status']})
            groups.append({'gene': gene, 'gene_symbol': self.conn.execute('SELECT symbol FROM genes WHERE gene=?', (gene,)).fetchone()[0], 'global_polarities': global_signs, 'cell_compatible_polarities': cell_signs, 'direction_ambiguous': len(global_signs) > 1, 'condition_causal_effect_verified': False, 'routes': compact})
        return groups

    def condition(self, drug, cell, *, role='core', genes=None, max_chars=8000, model_id='', prompt_version='', fold_id=''):
        if max_chars < 2000:
            raise ValueError('At least 2000 characters are required for the condition contract.')
        d = self.drug(drug)
        gene_ids = self._query_genes(genes, role)
        if cell not in CELLS:
            raise ValueError('Invalid SciPlex cell.')
        cache_identity = {'schema': SCHEMA, 'runtime_code_sha256': sha256(Path(__file__)), 'background_reader_sha256': sha256(Path(__file__).with_name('sciplex_knowledge_store.py')), 'index_snapshot': self.snapshot, 'drug': d['drug'], 'cell': cell, 'role': role, 'genes': gene_ids, 'max_chars': max_chars, 'model_id': model_id, 'prompt_version': prompt_version, 'fold_id': fold_id}
        key = identifier('condition', cache_identity)
        path = self.cache_root / (key.replace(':', '_') + '.json')
        if path.exists():
            try:
                envelope = json.loads(path.read_text())
            except (ValueError, OSError):
                envelope = {}
            if not isinstance(envelope, dict):
                envelope = {}
            if envelope.get('identity') == cache_identity and isinstance(envelope.get('packet'), dict) and (identifier('body', envelope['packet']) == envelope.get('body_id')):
                self.stats['cache_hits'] += 1
                return copy.deepcopy(envelope['packet'])
            self.stats['cache_corruptions'] += 1
        self.stats['cache_misses'] += 1
        base = self.background.condition(d['drug'], cell, max_chars=max(2000, max_chars // 2))
        coverage = self.coverage(d['drug'], cell, role)
        packet = {'schema': SCHEMA, 'cache_key': key, 'drug': d['drug'], 'name': d['canonical_name'], 'cell': cell, 'role': role, 'identity_tier': d['chembl_identity_tier'], 'index_snapshot': self.snapshot, 'native_roles': base['native_roles'], 'native_pathway': base['native_pathway'], 'background_evidence': base['evidence'], 'target_background': base['target_background'], 'coverage': {k: coverage['background'][k] for k in ['gene_denominator', 'citable_connection', 'signed_transcript', 'opposing_signed_transcript_pairs']}, 'cell_coverage': {k: coverage['cell'][k] for k in ['signed_transcript_pairs', 'opposing_signed_transcript_pairs']}, 'route_groups': [], 'seed_action_conflicts': [c for c in self.conflicts if c['drug'] == d['drug']], 'limitations': ['Composed routes are hypotheses; expression compatibility is not target engagement.', 'Opposing paths remain unresolved; pathway membership is association.'], 'retrieval_truncated': bool(base['retrieval_truncated']), 'path_groups_omitted': 0, 'budget': {'max_chars': max_chars, 'serialization': 'compact_JSON_unicode_characters', 'serialized_chars': 0, 'actual_provider_tokens': None}}
        packet['requested_genes'] = gene_ids
        query = 'SELECT COUNT(DISTINCT gene) FROM routes WHERE drug=? AND role=? AND cell=?'
        arguments = [d['drug'], role, '']
        if gene_ids:
            query += ' AND gene IN (' + ','.join(('?' for _ in gene_ids)) + ')'
            arguments.extend(gene_ids)
        packet['matching_citable_genes'] = self.conn.execute(query, arguments).fetchone()[0]
        groups = self.routes(d['drug'], cell, role=role, genes=gene_ids, limit=3)
        if packet['matching_citable_genes'] > len(groups):
            packet['retrieval_truncated'] = True
        for group in groups:
            packet['route_groups'].append(group)
            while len(wire(packet)) > max_chars - 12 and packet['target_background']:
                packet['target_background'].pop()
                packet['retrieval_truncated'] = True
            while len(wire(packet)) > max_chars - 12 and packet['background_evidence']:
                packet['background_evidence'].pop()
                packet['retrieval_truncated'] = True
            if len(wire(packet)) > max_chars - 12:
                packet['route_groups'].pop()
                packet['path_groups_omitted'] += 1
                packet['retrieval_truncated'] = True
        while len(wire(packet)) > max_chars - 12 and packet['target_background']:
            packet['target_background'].pop()
            packet['retrieval_truncated'] = True
        while len(wire(packet)) > max_chars - 12 and packet['background_evidence']:
            packet['background_evidence'].pop()
            packet['retrieval_truncated'] = True
        for _ in range(3):
            packet['budget']['serialized_chars'] = len(wire(packet))
        if len(wire(packet)) > max_chars:
            raise ValueError('Budget too small for required identity/conflict metadata.')
        write_json(path, {'identity': cache_identity, 'body_id': identifier('body', packet), 'packet': packet})
        return packet

    def verify_response(self, packet, response):
        errors = []
        value = copy.deepcopy(response)
        if not isinstance(value, dict):
            return {'valid': False, 'errors': ['invalid_response_shape'], 'normalized_response': None, 'validation_scope': 'structured_evidence_and_hypothesis_contract', 'free_text_semantics': 'not_evaluated', 'condition_causal_validation': 'not_evaluated'}
        try:
            if self.drug(str(value.get('drug')))['drug'] != packet['drug']:
                errors.append('wrong_drug')
        except (ValueError, KeyError):
            errors.append('unknown_drug')
        if value.get('cell') != packet['cell']:
            errors.append('wrong_cell')
        if value.get('role') != packet['role']:
            errors.append('wrong_role')
        pathmap = {r['route_id']: r for g in packet['route_groups'] for r in g['routes']}
        allowed = {e for r in pathmap.values() for e in r['evidence_ids']}
        allowed.update((e['evidence_id'] for e in packet['background_evidence']))
        claims = value.get('supported_context_claims', [])
        hypotheses = value.get('mechanism_hypotheses', [])
        if not isinstance(claims, list):
            errors.append('invalid_claims')
            claims = []
        if not isinstance(hypotheses, list):
            errors.append('invalid_hypotheses')
            hypotheses = []
        for claim in claims:
            if not isinstance(claim, dict):
                errors.append('invalid_claim')
                continue
            eid = claim.get('evidence_id')
            if not isinstance(eid, str) or eid not in allowed:
                errors.append('unknown_or_unprovided_evidence')
                continue
            evidence = self.evidence(eid)
            for field in ['subject', 'predicate', 'object', 'action_sign']:
                if claim.get(field) != evidence.get(field):
                    errors.append('unsupported_relation_field:' + field)
            if isinstance(claim.get('action_sign'), bool):
                errors.append('invalid_sign_type')
            if claim.get('condition_effect_verified') is not False:
                errors.append('unsupported_condition_effect')
        for hypothesis in hypotheses:
            if not isinstance(hypothesis, dict):
                errors.append('invalid_hypothesis')
                continue
            rid = hypothesis.get('route_id')
            if not isinstance(rid, str) or rid not in pathmap:
                errors.append('unknown_or_unprovided_path')
                continue
            route = pathmap[rid]
            refs = hypothesis.get('evidence_ids')
            if not isinstance(refs, list) or not all((isinstance(e, str) for e in refs)) or set(refs) != set(route['evidence_ids']):
                errors.append('incomplete_or_wrong_path_evidence')
            if hypothesis.get('status') != 'hypothesis':
                errors.append('path_promoted_to_fact')
            if hypothesis.get('net_sign') != route['net_sign']:
                errors.append('incorrect_path_sign')
            if isinstance(hypothesis.get('net_sign'), bool):
                errors.append('invalid_sign_type')
            group = next((g for g in packet['route_groups'] if any((r['route_id'] == rid for r in g['routes']))))
            if hypothesis.get('direction_ambiguous') != group['direction_ambiguous']:
                errors.append('opposing_paths_not_disclosed')
            if hypothesis.get('condition_effect_verified') is not False:
                errors.append('unsupported_condition_effect')
        returned_ids = {h.get('route_id') for h in hypotheses if isinstance(h, dict) and isinstance(h.get('route_id'), str)}
        for group in packet['route_groups']:
            ids = {r['route_id'] for r in group['routes']}
            if group['direction_ambiguous'] and returned_ids & ids and (not ids.issubset(returned_ids)):
                errors.append('opposite_hypothesis_omitted')
        if packet['route_groups'] and (not hypotheses):
            errors.append('missing_requested_hypotheses')
        limitations = value.get('limitations')
        if not isinstance(limitations, list) or not limitations or (not all((isinstance(x, str) and x.strip() for x in limitations))):
            errors.append('missing_limitations')
        value['drug'] = packet['drug']
        value['cell'] = packet['cell']
        return {'valid': not errors, 'errors': sorted(set(errors)), 'normalized_response': value, 'validation_scope': 'structured_evidence_and_hypothesis_contract', 'free_text_semantics': 'not_evaluated', 'condition_causal_validation': 'not_evaluated'}

    def contract_example(self, packet):
        claims = {}
        hypotheses = []
        for group in packet['route_groups']:
            for route in group['routes']:
                for eid in route['evidence_ids']:
                    e = self.evidence(eid)
                    claims[eid] = {k: e[k] for k in ['evidence_id', 'subject', 'predicate', 'object', 'action_sign']}
                    claims[eid]['condition_effect_verified'] = False
                hypotheses.append({'route_id': route['route_id'], 'status': 'hypothesis', 'evidence_ids': route['evidence_ids'], 'net_sign': route['net_sign'], 'direction_ambiguous': group['direction_ambiguous'], 'condition_effect_verified': False})
        return {'drug': packet['drug'], 'cell': packet['cell'], 'role': packet['role'], 'supported_context_claims': list(claims.values()), 'mechanism_hypotheses': hypotheses, 'limitations': packet['limitations'], 'generation': 'deterministic_contract_fixture_not_llm_output'}
