from __future__ import annotations
import copy
import itertools
import json
from cellcanon.model.observation_state import digest
from cellcanon.model.explanation_benchmark import LABELS

def sources_for_packet(packet, store):
    sources = []
    for alias, meta in packet['evidence'].items():
        e = store.evidence(meta['source_id'])
        record = {k: e.get(k) for k in ('subject', 'predicate', 'object', 'action_sign', 'kind', 'source_url')}
        for field in ('subject', 'object'):
            raw = record.get(field)
            if isinstance(raw, str) and raw.startswith('ENSG'):
                rows = store.conn.execute('SELECT DISTINCT symbol FROM genes WHERE gene=?', (raw,)).fetchall()
                if len(rows) == 1 and rows[0][0]:
                    record['original_' + field] = raw
                    record[field] = 'GENE:' + rows[0][0]
        src = e.get('source_record', {})
        record['text'] = e.get('text') or src.get('assay_description') or src.get('mechanism_of_action') or ''
        record['text'] = str(record['text'])[:650]
        record['condition_effect_verified'] = False
        sources.append(dict(id=alias, source_id=meta['source_id'], record=record))
    for i, t in enumerate(packet['targets']):
        value = t['control'].get('minimum_detection')
        if value is not None:
            record = dict(subject=t['target'], predicate='vehicle_control_detection_fraction', object=float(value), cell=packet['cell'], time_hours=24, drug_treatment=False, condition_effect_verified=False, text='Vehicle control gene detection is not protein activity or drug target engagement.')
            sources.append(dict(id='B' + str(i + 1), source_id='control:' + digest(record), record=record))
    return sources

def build_case(packet, store, split):
    sources = sources_for_packet(packet, store)
    byid = {r['id']: r for r in sources}
    claims = []
    gold = []

    def add(kind, text, args, label, proofs, *, support=None, opposite=None, unknown=None):
        identity = 'Q' + digest([packet['drug'], packet['cell'], kind, args])[:10]
        previous = LABELS[(LABELS.index(label) + len(claims) % 2) % 3]
        claims.append(dict(id=identity, kind=kind, text=text, arguments=args, previous_label=previous))
        gold.append(dict(id=identity, label=label, evidence_sets=proofs, previous_label=previous, kind=kind, support_sets=support if support is not None else proofs if label == 'supported' else [], opposite_sets=opposite if opposite is not None else proofs if label == 'contradicted' else [], unresolved_sets=unknown or []))
    for target in packet['targets'][:2]:
        name = target['target']
        for sign, verb in [(-1, 'inhibits'), (1, 'activates')]:
            matching = [r for r in sources if r['record'].get('subject') == packet['drug'] and r['record'].get('object') == name and (r['record'].get('predicate') == 'functional_target_action')]
            supporting = [[r['id']] for r in matching if r['record'].get('action_sign') == sign]
            opposite = [[r['id']] for r in matching if r['record'].get('action_sign') == -sign]
            label = 'insufficient' if supporting and opposite else 'supported' if supporting else 'contradicted' if opposite else 'insufficient'
            proofs = supporting if label == 'supported' else opposite if label == 'contradicted' else []
            add('exclusive_target_action', f"The supplied functional target evidence establishes an exclusively {('inhibitory' if sign == -1 else 'activating')} action of {packet['drug_name']} on {name}.", {'drug': packet['drug'], 'target': name, 'sign': sign}, label, proofs, support=supporting, opposite=opposite)
        baseline = next((r for r in sources if r['record'].get('subject') == name and r['record'].get('predicate') == 'vehicle_control_detection_fraction'), None)
        for above in (True, False):
            value = baseline['record']['object'] if baseline else None
            label = 'insufficient' if value is None else 'supported' if (value >= 0.01) == above else 'contradicted'
            add('control_detection', f"In {packet['cell']} 24h vehicle controls, {name} was detected in {('at least' if above else 'fewer than')} 1% of cells.", {'target': name, 'cell': packet['cell'], 'threshold': 0.01, 'above': above}, label, [[baseline['id']]] if baseline else [])
        add('condition_engagement', f"The supplied evidence demonstrates that {packet['drug_name']} engaged {name} in {packet['cell']} cells at the tested pooled dose and 24h.", {'drug': packet['drug'], 'target': name, 'cell': packet['cell'], 'time_hours': 24, 'dose': 'pooled'}, 'insufficient', [])
    route_candidates = []
    valid = []
    related = []
    for gi, g in enumerate(packet['route_groups']):
        for pi, path in enumerate(g['paths']):
            rid = 'P' + digest([gi, pi, path])[:10]
            proof = [eid for eid in path['evidence'] if byid[eid]['record'].get('predicate') != 'vehicle_expression']
            alternatives = reference_alternatives(proof, byid)
            same = [p['evidence'] for p in alternatives if path['sign'] is None or p['sign'] == path['sign']]
            opp = [p['evidence'] for p in alternatives if path['sign'] is not None and p['sign'] in (-1, 1) and (p['sign'] != path['sign'])]
            unk = [p['evidence'] for p in alternatives if path['sign'] is not None and p['sign'] is None]
            value = dict(id=rid, gene=g['gene'], nodes=path['nodes'], sign=path['sign'], tier=path['tier'], evidence=proof, proof_sets=same, qualification='background_hypothesis')
            route_candidates.append(value)
            (valid if gi == 0 else related).append(rid)
            if gi == 0:
                add('path_statement', f"The supplied records support this exact background route to {g['gene']}, with the stated sign, as a mechanism hypothesis: " + json.dumps(value['nodes']) + '.', value, 'supported', same, support=same, opposite=opp, unknown=unk)
                pairs = [(packet['drug'], path['nodes'][0])] + list(zip(path['nodes'], path['nodes'][1:]))
                unambiguous = all((len({v['record'].get('action_sign') for v in sources if (v['record'].get('subject'), v['record'].get('object')) == pair and v['record'].get('action_sign') in (-1, 1)}) == 1 for pair in pairs))
                if path['sign'] in (-1, 1) and unambiguous:
                    false = copy.deepcopy(value)
                    false['id'] = 'P' + digest([rid, 'reverse_sign'])[:10]
                    false['sign'] = -path['sign']
                    route_candidates.append(false)
                    add('path_statement', f"The supplied records support this exact background route to {g['gene']}, with the stated sign, as a mechanism hypothesis: " + json.dumps(false['nodes']) + '.', false, 'contradicted', same, support=opp, opposite=same, unknown=unk)
                causal = copy.deepcopy(value)
                causal['qualification'] = 'condition_causal_fact'
                causal['id'] = 'P' + digest([rid, 'causal_promotion'])[:10]
                route_candidates.append(causal)
                add('causal_path', f"The supplied records experimentally establish this whole route as the cause of the response in {packet['cell']} at 24h pooled dose.", causal, 'insufficient', [])
    if not claims:
        for kind in ['target', 'transcription_route', 'cell_specific_cause']:
            add('unavailable_evidence', f"The supplied local evidence establishes a {kind} mechanism for {packet['drug_name']} in {packet['cell']}.", {'drug': packet['drug'], 'cell': packet['cell'], 'mechanism': kind}, 'insufficient', [])
    background = packet.get('background', [])
    if background:
        b = background[0]
        text = b['text'][:140]
        byid[b['evidence']]['record']['text'] = b['text']
        add('source_text', f"The source {b['evidence']} contains this quoted excerpt: {text}", {'source': b['evidence'], 'quote': text}, 'supported', [[b['evidence']]])
    order = sorted(range(len(claims)), key=lambda i: digest([packet['drug'], packet['cell'], claims[i]['id']]))
    route_candidates = sorted(route_candidates, key=lambda r: digest(['candidate-order', r['id']]))
    public = dict(id=packet['drug'] + '_' + packet['cell'], split=split, drug=packet['drug'], name=packet['drug_name'], cell=packet['cell'], time_hours=24, dose='pooled', sources=sources, claims=[claims[i] for i in order], route_candidates=route_candidates, requested_route_gene=packet['route_groups'][0]['gene'] if valid else None, semantics='Verify against supplied sources. Background relations and vehicle expression do not establish condition causal effects.')
    private = dict(id=public['id'], claims=[gold[i] for i in order], route=dict(id=public['id'], strict=sorted(valid)[0], valid=valid, related=related, candidates=[r['id'] for r in route_candidates]) if valid else None, reference_paths=[r for r in route_candidates if r['id'] in valid + related], source_ids=[r['source_id'] for r in sources], causal_gold=False)
    private = gold_after_removal(private, {s['id'] for s in sources})
    return (public, private)

def reference_alternatives(proof, byid):
    parts = []
    target_types = {'functional_target_action', 'measured_target_binding', 'curated_target_relation_unknown_action'}
    for eid in proof:
        record = byid[eid]['record']
        pred = record.get('predicate')
        choices = []
        for alias, row in byid.items():
            r = row['record']
            if (r.get('subject'), r.get('object')) != (record.get('subject'), record.get('object')):
                continue
            if r.get('predicate') == pred or (pred in target_types and r.get('predicate') in target_types):
                choices.append((alias, r.get('action_sign')))
        parts.append(choices)
    out = []
    for seq in itertools.product(*parts):
        sign = 1
        for _, s in seq:
            if s not in (-1, 1):
                sign = None
                break
            sign *= s
        out.append(dict(evidence=sorted(set((r[0] for r in seq))), sign=sign))
    return out

def public_view(case, keep=None):
    value = copy.deepcopy(case)
    if keep is not None:
        value['sources'] = [r for r in value['sources'] if r['id'] in keep]
    for source in value['sources']:
        source.pop('source_id', None)
    for c in value['route_candidates']:
        c.pop('evidence', None)
        c.pop('proof_sets', None)
    for c in value['claims']:
        c['arguments'].pop('evidence', None)
        c['arguments'].pop('proof_sets', None)
    return value

def gold_after_removal(private, keep):
    value = copy.deepcopy(private)
    for g in value['claims']:
        if 'support_sets' not in g:
            remaining = [s for s in g['evidence_sets'] if set(s) <= set(keep)]
            if g['evidence_sets'] and (not remaining):
                g['label'] = 'insufficient'
            g['evidence_sets'] = remaining
            continue
        support = [s for s in g['support_sets'] if set(s) <= set(keep)]
        opposite = [s for s in g['opposite_sets'] if set(s) <= set(keep)]
        unknown = [s for s in g['unresolved_sets'] if set(s) <= set(keep)]
        if g['kind'] == 'exclusive_target_action' and support and opposite:
            label = 'insufficient'
        elif support:
            label = 'supported'
        elif opposite and (not unknown):
            label = 'contradicted'
        else:
            label = 'insufficient'
        g['label'] = label
        g['evidence_sets'] = support if label == 'supported' else opposite if label == 'contradicted' else []
    if value['route']:
        paths = [r for r in value['reference_paths'] if any((set(p) <= set(keep) for p in r.get('proof_sets', [r['evidence']])))]
        value['route']['valid'] = [r['id'] for r in paths if r['id'] in value['route']['valid']]
        value['route']['related'] = [r['id'] for r in paths if r['id'] in value['route']['related']]
    return value
