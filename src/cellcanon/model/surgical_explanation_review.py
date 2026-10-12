from __future__ import annotations
import copy
import json
from pathlib import Path
from cellcanon.model.answerable_explanation import evidence_payload
from cellcanon.model.explanation_repair import audit_prediction, write_record
from cellcanon.model.observation_answer_benchmark import validate
from cellcanon.model.supported_act import digest
from cellcanon.model.supported_act_llm import checked_call
from cellcanon.model.verified_explanation_review import REVIEW_SYSTEM

def review_surgical(client, case, previous, directory, *, max_rounds=2):
    directory = Path(directory)
    final = directory / 'result.json'
    identity = digest(dict(case=case, previous=previous.get('request_hash'), model=client.model, max_rounds=max_rounds))
    if final.exists():
        old = json.loads(final.read_text())
        if old['identity'] != identity:
            raise ValueError('Saved corrections use different inputs. Choose a new output directory.')
        return old
    current = copy.deepcopy(previous.get('value', {})) if previous.get('success') else dict(claims=[])
    model_rows = {r['id']: r for r in current.get('claims', [])}
    rounds = []
    initial = None
    for step in range(max_rounds):
        issues = audit_prediction(case, dict(claims=list(model_rows.values())))
        if initial is None:
            initial = copy.deepcopy(issues)
        if not issues:
            break
        required = {r['id'] for r in issues}
        subset = copy.deepcopy(case)
        subset['claims'] = [c for c in case['claims'] if c['id'] in required]
        context, aliases = evidence_payload(subset, cards=True)
        context.pop('route_candidates', None)
        context.pop('requested_endpoint', None)
        for i, r in enumerate(context['sources']):
            r['source_id'] = aliases[i]
        context['public_source_checks'] = [dict(q=i, **{k: v for k, v in next((r for r in issues if r['id'] == c['id'])).items() if k != 'id'}) for i, c in enumerate(subset['claims'])]
        context['previous_answers'] = [dict(q=i, **{k: v for k, v in model_rows[c['id']].items() if k != 'id'}) for i, c in enumerate(subset['claims']) if c['id'] in model_rows]
        contract = dict(questions=[dict(q=i, id=c['id']) for i, c in enumerate(subset['claims'])], records=[dict(id=a) for a in aliases])
        messages = [dict(role='system', content=REVIEW_SYSTEM + '\nOnly the unresolved questions are shown. Re-evaluate their actual records; a prior judgment failed a public consistency check. Focus on the printed functional_action_records, named_cell_measurements and complete_background_alternatives for THIS question. Do not use general prose as a typed functional record. If functional_action_records is empty, exclusive target action cannot be established even if prose says inhibitor. Source confidence weights are independent of cited-source counts.'), dict(role='user', content=json.dumps(context, ensure_ascii=False, separators=(',', ':')))]
        r = checked_call(client, messages, directory / ('round' + str(step) + '.json'), lambda v: validate(contract, v), logical_id='surgical-explain:' + case['id'] + ':' + str(step), max_tokens=3200)
        rounds.append(dict(questions=[c['id'] for c in subset['claims']], success=r['success'], request_sha256=digest(messages)))
        if not r['success']:
            break
        for row in r['value']['claims']:
            model_rows[row['id']] = row
    remaining = audit_prediction(case, dict(claims=list(model_rows.values())))
    complete = len(model_rows) == len(case['claims'])
    result = dict(identity=identity, case_id=case['id'], success=complete, initial_public_issues=initial or [], remaining_public_issues=remaining, rounds=rounds, value=dict(claims=[model_rows[c['id']] for c in case['claims'] if c['id'] in model_rows], route_ranking=current.get('route_ranking', [])), claim_decisions_produced_by='LLM, with corrections for answers that failed source checks', host_label_changes=0, route_task_separately_unmodified=True)
    result['value']['selected_evidence'] = sorted({e for c in result['value']['claims'] for e in c['evidence']})
    write_record(final, result)
    return result
