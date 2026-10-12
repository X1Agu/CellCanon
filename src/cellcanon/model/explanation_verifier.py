from __future__ import annotations
import itertools
import numpy as np
from cellcanon.model.explanation_benchmark import LABELS

def path_proofs(candidate, sources, drug):
    nodes = candidate['nodes']
    rows = [(r['id'], r['record']) for r in sources]
    if not nodes:
        return []
    seed = [(eid, e) for eid, e in rows if e.get('subject') == drug and e.get('object') == nodes[0] and (e.get('predicate') in {'functional_target_action', 'measured_target_binding', 'curated_target_relation_unknown_action'})]
    parts = [seed]
    if candidate['tier'].startswith('shared_pathway'):
        if len(nodes) != 3:
            return []
        last = nodes[-1] if nodes[-1].startswith('GENE:') else 'GENE:' + nodes[-1]
        for node in (nodes[0], last):
            parts.append([(eid, e) for eid, e in rows if e.get('subject') in {node, node.removeprefix('GENE:')} and e.get('predicate') == 'member_of' and (e.get('object') == nodes[1])])
    else:
        for a, b in zip(nodes, nodes[1:]):
            parts.append([(eid, e) for eid, e in rows if e.get('subject') == a and e.get('object') == b and (e.get('predicate') not in {'member_of', 'described_in_source', 'vehicle_expression'})])
    if any((not v for v in parts)):
        return []
    out = []
    for sequence in itertools.product(*parts):
        signs = [r.get('action_sign') for _, r in sequence]
        sign = int(np.prod(signs)) if all((x in (-1, 1) for x in signs)) else None
        out.append(dict(evidence=[eid for eid, _ in sequence], sign=sign))
    return out

def decide_claim(claim, sources, drug):
    args = claim['arguments']
    kind = claim['kind']
    proofs = []
    label = 'insufficient'
    if kind == 'exclusive_target_action':
        rows = [r for r in sources if r['record'].get('subject') == drug and r['record'].get('object') == args['target'] and (r['record'].get('predicate') == 'functional_target_action')]
        signs = {r['record']['action_sign'] for r in rows if r['record'].get('action_sign') in (-1, 1)}
        if len(signs) == 1:
            label = 'supported' if args['sign'] in signs else 'contradicted'
            proofs = [[r['id']] for r in rows if r['record'].get('action_sign') in signs]
    elif kind == 'control_detection':
        rows = [r for r in sources if r['record'].get('subject') == args['target'] and r['record'].get('cell') == args['cell'] and (r['record'].get('predicate') == 'vehicle_control_detection_fraction')]
        if rows:
            answers = {(r['record']['object'] >= args['threshold']) == args['above'] for r in rows}
            if len(answers) == 1:
                label = 'supported' if True in answers else 'contradicted'
                proofs = [[r['id']] for r in rows]
    elif kind == 'source_text':
        rows = [r for r in sources if r['id'] == args['source']]
        if rows:
            label = 'supported' if args['quote'] in rows[0]['record'].get('text', '') else 'contradicted'
            proofs = [[rows[0]['id']]]
    elif kind == 'path_statement':
        paths = path_proofs(args, sources, drug)
        correct = [p for p in paths if args['sign'] is None or p['sign'] == args['sign']]
        if correct:
            label = 'supported'
            proofs = [p['evidence'] for p in correct]
        elif paths and all((p['sign'] is not None for p in paths)):
            label = 'contradicted'
            proofs = [p['evidence'] for p in paths]
    refs = min(proofs, key=lambda p: (len(p), p)) if proofs else []
    probs = [0.0, 0.0, 0.0]
    probs[LABELS.index(label)] = 1.0
    return dict(id=claim['id'], label=label, probabilities=probs, evidence=refs, reason='Typed source scope and relation composition.')
