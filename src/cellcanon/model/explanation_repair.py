from __future__ import annotations
import copy
import json
from pathlib import Path
from cellcanon.model.explanation_cases import public_view
from cellcanon.model.explanation_verifier import decide_claim, path_proofs
from cellcanon.model.observation_state import digest
TARGET_TYPES = {'functional_target_action', 'measured_target_binding', 'curated_target_relation_unknown_action'}

def write_record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError('The output path already contains a different record. Choose a new path.')
    path.write_text(text)

def canonical_case(case, keep=None):
    result = public_view(case, keep)
    for claim in result['claims']:
        claim.pop('previous_label', None)
        if claim['kind'] == 'path_statement':
            args = claim['arguments']
            claim['text'] = 'Assess this BACKGROUND route hypothesis, INCLUDING the drug-to-first-target edge. Required whole chain: ' + ' -> '.join([case['drug']] + args['nodes']) + '. Claimed net sign: ' + str(args['sign']) + '. A null claimed sign asks only whether the complete background route is sourced; unknown action signs are allowed for that unsigned question.'
        elif claim['kind'] in {'causal_path', 'condition_engagement'}:
            claim['evidence_scope_note'] = 'Judge the biological proposition under the supplied evidence. Insufficient means neither established nor experimentally refuted; it does NOT mean contradicted. A background record or vehicle RNA measurement cannot establish actual target engagement or a whole causal mechanism at this condition.'
    for route in result['route_candidates']:
        route['whole_chain'] = [case['drug']] + route['nodes']
    result['input_protocol'] = 'explicit-whole-chain-no-constructed-prior-v2'
    return result

def edge_slots(candidate, sources, drug):
    nodes = candidate['nodes']
    if not nodes:
        return []
    specs = [(drug, nodes[0], 'drug_target')]
    if candidate['tier'].startswith('shared_pathway'):
        if len(nodes) != 3:
            return []
        specs += [(nodes[0], nodes[1], 'member_of'), (nodes[-1], nodes[1], 'member_of')]
    else:
        specs += [(a, b, 'network') for a, b in zip(nodes, nodes[1:])]
    slots = []
    for a, b, role in specs:
        options = []
        for row in sources:
            r = row['record']
            left, right = (r.get('subject'), r.get('object'))
            if role == 'member_of':
                matches = str(left).removeprefix('GENE:') == a.removeprefix('GENE:') and right == b and (r.get('predicate') == 'member_of')
            else:
                matches = left == a and right == b
                if role == 'drug_target':
                    matches = matches and r.get('predicate') in TARGET_TYPES
                else:
                    matches = matches and r.get('predicate') not in {'member_of', 'described_in_source', 'vehicle_expression'}
            if matches:
                options.append(dict(evidence=row['id'], relation=r.get('predicate'), sign=r.get('action_sign')))
        slots.append(dict(subject=a, object=b, role=role, options=options))
    return slots

def graph_payload(case):
    value = copy.deepcopy(case)
    index = {}
    for candidate in case['route_candidates']:
        key = digest([candidate['nodes'], candidate['tier']])[:12]
        if key not in index:
            index[key] = edge_slots(candidate, case['sources'], case['drug'])
        next((r for r in value['route_candidates'] if r['id'] == candidate['id']))['edge_index'] = key
    value['edge_index'] = index
    for claim in value['claims']:
        args = claim['arguments']
        if claim['kind'] in {'path_statement', 'causal_path'}:
            claim['edge_index'] = digest([args['nodes'], args['tier']])[:12]
        elif claim['kind'] in {'exclusive_target_action', 'control_detection', 'condition_engagement'}:
            target = args['target']
            claim['relevant_source_ids'] = [s['id'] for s in case['sources'] if s['record'].get('subject') == target or s['record'].get('object') == target]
    return value

def compiled_payload(case):
    value = graph_payload(case)
    compositions = {}
    for candidate in case['route_candidates']:
        key = digest([candidate['nodes'], candidate['tier']])[:12]
        if key not in compositions:
            compositions[key] = path_proofs(candidate, case['sources'], case['drug'])
    value['public_path_computations'] = compositions
    for claim in value['claims']:
        args = claim['arguments']
        if claim['kind'] == 'control_detection':
            claim['public_threshold_computations'] = [dict(evidence=s['id'], fraction=s['record']['object'], relation_to_threshold='at_or_above' if s['record']['object'] >= args['threshold'] else 'below') for s in case['sources'] if s['record'].get('subject') == args['target'] and s['record'].get('cell') == args['cell'] and (s['record'].get('predicate') == 'vehicle_control_detection_fraction')]
        if claim['kind'] == 'exclusive_target_action':
            claim['public_functional_action_computations'] = [dict(evidence=s['id'], sign=s['record'].get('action_sign')) for s in case['sources'] if s['record'].get('subject') == case['drug'] and s['record'].get('object') == args['target'] and (s['record'].get('predicate') == 'functional_target_action')]
    return value

def audit_prediction(case, prediction):
    byid = {r['id']: r for r in prediction.get('claims', [])}
    issues = []
    for claim in case['claims']:
        proposed = byid.get(claim['id'], {})
        local = decide_claim(claim, case['sources'], case['drug'])
        valid = proposed.get('label') == local['label']
        refs = set(proposed.get('evidence', []))
        if claim['kind'] == 'path_statement' and local['label'] != 'insufficient':
            paths = path_proofs(claim['arguments'], case['sources'], case['drug'])
            groups = [set(p['evidence']) for p in paths if (claim['arguments']['sign'] is None or p['sign'] == claim['arguments']['sign']) == (local['label'] == 'supported')]
            complete = any((g <= refs for g in groups))
            admissible = set().union(*groups) if groups else set()
        else:
            selected = [s for s in case['sources'] if s['id'] in refs]
            complete = not refs if local['label'] == 'insufficient' else decide_claim(claim, selected, case['drug'])['label'] == local['label']
            admissible = {s['id'] for s in case['sources']}
        if not valid or not complete or refs - admissible:
            issues.append(dict(id=claim['id'], decision_source_consistent=valid, selected_proof_complete=complete, inadmissible_references=sorted(refs - admissible), instruction='Recheck scope, unknown versus opposite, and ALL edge slots including the drug seed.'))
    return issues
