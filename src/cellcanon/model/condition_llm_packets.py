from __future__ import annotations
import hashlib
import json
from cellcanon.resources.sciplex_route_knowledge import RouteKnowledge

def wire(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(wire(value).encode()).hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

class PacketBuilder:

    def __init__(self):
        self.store = RouteKnowledge()
        self.static = {}

    def drug(self, drug):
        if drug in self.static:
            return self.static[drug]
        card = self.store.drug(drug)
        rows = [json.loads(r[0]) for r in self.store.conn.execute('SELECT payload FROM evidence WHERE drug=? ORDER BY evidence_id', (drug,))]
        seeds = [r for r in rows if str(r.get('object', '')).startswith('GENE:') and r.get('predicate') in {'functional_target_action', 'measured_target_binding', 'curated_target_relation_unknown_action'}]
        by_target = {}
        for s in seeds:
            by_target.setdefault(s['object'], []).append(s)
        targets = []
        for target, records in by_target.items():
            records.sort(key=lambda r: (r.get('action_sign') is None, r.get('kind') != 'gtopdb_primary_functional_action', r['evidence_id']))
            symbol = target[5:]
            text = self.store.background.conn.execute("SELECT d.evidence_id,d.text FROM documents d JOIN document_symbols s ON d.evidence_id=s.evidence_id WHERE s.symbol=? AND d.kind IN ('reviewed_human_function','ncbi_human_gene_summary') ORDER BY d.kind,d.evidence_id LIMIT 1", (symbol,)).fetchone()
            targets.append({'target': target, 'actions': sorted({r['action_sign'] for r in records if r.get('action_sign') is not None}), 'records': records[:2], 'gene_text': {'evidence_id': text[0], 'text': text[1][:240]} if text else None})
        targets.sort(key=lambda t: (not bool(t['actions']), t['target']))
        value = {'card': card, 'targets': targets[:8], 'all_target_count': len(targets)}
        self.static[drug] = value
        return value

    def condition(self, drug, cell):
        static = self.drug(drug)
        base = self.store.condition(drug, cell, max_chars=8000)
        evidence = {}
        aliases = {}

        def add(eid, **extra):
            if eid in aliases:
                return aliases[eid]
            source = self.store.evidence(eid)
            alias = 'E' + str(len(aliases) + 1)
            aliases[eid] = alias
            evidence[alias] = {'source_id': eid, 'kind': source['kind'], 'subject': source.get('subject'), 'predicate': source.get('predicate'), 'object': source.get('object'), 'action_sign': source.get('action_sign'), **extra}
            return alias
        background = []
        for b in base['background_evidence'][:2]:
            background.append({'evidence': add(b['evidence_id']), 'text': b['text_excerpt'][:500]})
        targets = []
        for t in static['targets']:
            refs = [add(s['evidence_id']) for s in t['records']]
            row = self.store.conn.execute('SELECT payload FROM node_context WHERE node=? AND cell=?', (t['target'], cell)).fetchone()
            context = json.loads(row[0]) if row else {}
            measurements = context.get('measurements', [])
            control = {'status': context.get('status', 'unknown'), 'minimum_detection': context.get('minimum_detection'), 'mean_log1p_cp10k': measurements[0].get('control_mean_log1p_cp10k') if len(measurements) == 1 else None}
            gene_text = None
            if t['gene_text']:
                gene_text = {'evidence': add(t['gene_text']['evidence_id']), 'text': t['gene_text']['text']}
            targets.append({'target': t['target'], 'actions': t['actions'], 'evidence': refs, 'control': control, 'function': gene_text})
        routes = []
        for i, g in enumerate(base['route_groups']):
            paths = []
            for r in g['routes']:
                paths.append({'sign': r['net_sign'], 'nodes': r['nodes'], 'tier': r['tier'], 'cell_compatibility': r['cell_compatibility'], 'evidence': [add(e) for e in r['evidence_ids']]})
            routes.append({'group': 'R' + str(i + 1), 'gene': g['gene_symbol'], 'polarities': g['global_polarities'], 'cell_compatible_polarities': g['cell_compatible_polarities'], 'paths': paths})
        packet = {'drug': drug, 'drug_name': base['name'], 'cell': cell, 'native_roles': base['native_roles'], 'native_pathway': base['native_pathway'], 'identity_tier': base['identity_tier'], 'background': background, 'targets': targets, 'route_groups': routes, 'evidence': evidence, 'coverage': base['coverage'], 'cell_coverage': base['cell_coverage'], 'targets_available': static['all_target_count'], 'limitations': base['limitations'], 'variant': 'full'}
        return packet

    def close(self):
        self.store.close()
