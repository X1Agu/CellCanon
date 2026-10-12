from __future__ import annotations
import copy
import json
from pathlib import Path
from cellcanon.model.answerable_explanation import infer_explanation, evidence_payload
from cellcanon.model.surgical_explanation_review import review_surgical
from cellcanon.model.explanation_repair import audit_prediction, write_record
from cellcanon.model.observation_answer_benchmark import validate
from cellcanon.model.path_completion_receipts import completion_receipt
from cellcanon.model.route_answer_review import public_route_request, review_routes, validate_routes
from cellcanon.model.source_explain_prompts import COMPLETION_SYSTEM, ROUTE_TABLE_SYSTEM
from cellcanon.model.supported_act import digest
from cellcanon.model.supported_act_llm import checked_call

def normalize_source_positions(value, aliases):
    fixed = copy.deepcopy(value)
    changed = 0
    for row in fixed.get('answers', []):
        for i, ref in enumerate(row.get('refs', [])):
            if type(ref) is int and 0 <= ref < len(aliases):
                row['refs'][i] = aliases[ref]
                changed += 1
    return (fixed, changed)

def _recover_last_wire_reply(case, review, directory):
    current = copy.deepcopy(review['value'])
    lookup = {r['id']: r for r in current['claims']}
    trace = dict(attempted=False, converted_positions=0)
    if review['rounds'] and (not review['rounds'][-1]['success']):
        trace['attempted'] = True
        step = len(review['rounds']) - 1
        path = Path(directory) / ('round' + str(step) + '.json')
        raw = json.loads(path.read_text())
        ids = set(review['rounds'][-1]['questions'])
        subset = [q for q in case['claims'] if q['id'] in ids]
        aliases = [s['id'] for s in case['sources']]
        contract = dict(questions=[dict(q=i, id=q['id']) for i, q in enumerate(subset)], records=[dict(id=a) for a in aliases])
        try:
            value, count = normalize_source_positions(json.loads(raw['attempts'][-1]['raw']), aliases)
            bound = validate(contract, value)
            for row in bound['claims']:
                lookup[row['id']] = row
            trace.update(recovered=True, converted_positions=count)
        except (ValueError, KeyError, TypeError, IndexError):
            trace['recovered'] = False
    current['claims'] = [lookup[q['id']] for q in case['claims'] if q['id'] in lookup]
    return (current, trace)

def _review_completions(client, case, previous, path):
    issues = audit_prediction(case, previous)
    kinds = {q['id']: q['kind'] for q in case['claims']}
    required = {i['id'] for i in issues if kinds[i['id']] == 'path_statement'}
    current = copy.deepcopy(previous)
    if not required:
        return (current, dict(flagged=False, success=True))
    subset = {**case, 'claims': [q for q in case['claims'] if q['id'] in required]}
    context, aliases = evidence_payload(subset, cards=True)
    context.pop('route_candidates', None)
    for i, row in enumerate(context['sources']):
        row['source_id'] = aliases[i]
    for question in context['questions']:
        question['possible_completions'] = completion_receipt(question['tool_receipt']['complete_background_alternatives'], context['sources'])
    contract = dict(questions=[dict(q=i, id=q['id']) for i, q in enumerate(subset['claims'])], records=[dict(id=a) for a in aliases])
    result = checked_call(client, [dict(role='system', content=COMPLETION_SYSTEM), dict(role='user', content=json.dumps(context, ensure_ascii=False, separators=(',', ':')))], path, lambda v: validate(contract, v), logical_id='path-completion:' + case['id'], max_tokens=1500)
    if result['success']:
        replacement = {q['id']: q for q in result['value']['claims']}
        current['claims'] = [replacement.get(q['id'], q) for q in previous['claims']]
    return (current, dict(flagged=True, success=result['success'], questions=sorted(required)))

def _review_route_table(client, case, previous, path):
    payload, flags = public_route_request(case, previous)
    current = copy.deepcopy(previous)
    if not flags:
        return (current, dict(flagged=False, success=True))
    aliases = {s['source']: case['sources'][s['source']]['id'] for s in payload['sources']}
    table = dict(requested_endpoint=payload['requested_endpoint'], routes=[dict(route=r['route'], gene=r['gene'], whole_chain=r['whole_chain'], claimed_sign=r['sign'], qualification=r['qualification'], complete_alternatives=[dict(net_sign=a['net_sign'], source_ids=[aliases[i] for i in a['refs']]) for a in r['complete_background_alternatives']]) for r in payload['route_candidates']])
    result = checked_call(client, [dict(role='system', content=ROUTE_TABLE_SYSTEM), dict(role='user', content=json.dumps(table, separators=(',', ':')))], path, lambda v: validate_routes(case, v), logical_id='route-table:' + case['id'], max_tokens=400)
    if result['success']:
        current.update(result['value'])
    return (current, dict(flagged=True, success=result['success']))

def run_source_explanation(client, case, output):
    if any((key in case for key in ['gold', 'private', 'reference_answers', 'evaluator_truth'])):
        raise ValueError('Only a public source/question packet may enter inference.')
    out = Path(output)
    path = out / 'result.json'
    identity = digest(dict(interface='source-explain-pipeline-v1', case=case, model=client.model))
    if path.exists():
        old = json.loads(path.read_text())
        if old['identity'] != identity:
            raise ValueError('Saved results use different sources or a different model. Choose a new output directory.')
        return old
    initial = infer_explanation(client, case, out / 'initial', method='deliberate')
    reviewed = review_surgical(client, case, initial, out / 'surgical')
    current, wire = _recover_last_wire_reply(case, reviewed, out / 'surgical')
    current, completion = _review_completions(client, case, current, out / 'completion.json')
    current, route1 = review_routes(client, case, current, out / 'route_review.json')
    current, route2 = _review_route_table(client, case, current, out / 'route_table.json')
    current['selected_evidence'] = sorted({e for row in current['claims'] for e in row['evidence']})
    missing = sorted({q['id'] for q in case['claims']} - {q['id'] for q in current['claims']})
    result = dict(identity=identity, case_id=case['id'], model=client.model, value=current, complete_claim_contract=not missing, missing_question_ids=missing, stages=dict(initial=initial['success'], surgical_rounds=reviewed['rounds'], wire=wire, path_completions=completion, route_review=route1, route_table=route2), unresolved_public_source_checks=audit_prediction(case, current), unresolved_public_route_checks=public_route_request(case, current)[1], host_semantic_label_changes=0, host_route_choices=0, private_answer_access=False, biological_causality_verified=False, scope='Source entailment, evidence sufficiency and complete background route support within the supplied records.')
    write_record(path, result)
    return result
