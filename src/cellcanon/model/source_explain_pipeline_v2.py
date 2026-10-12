from __future__ import annotations
import copy
import json
from pathlib import Path
from cellcanon.model.source_explain_pipeline import run_source_explanation
from cellcanon.model.route_eligibility_review import review_eligibility
from cellcanon.model.route_answer_review import public_route_request
from cellcanon.model.supported_act import digest
from cellcanon.model.explanation_repair import write_record

def run_complete_source_explanation(client, case, output):
    if any((k in case for k in ['gold', 'private', 'reference_answers', 'evaluator_truth'])):
        raise ValueError('Only a public source/question packet may enter inference.')
    out = Path(output)
    identity = digest(dict(interface='source-explain-pipeline-v2', case=case, model=client.model))
    path = out / 'result.json'
    if path.exists():
        result = json.loads(path.read_text())
        if result['identity'] != identity:
            raise ValueError('Saved results use different sources or a different model. Choose a new output directory.')
        return result
    parent = run_source_explanation(client, case, out / 'parent')
    value, trace = review_eligibility(client, case, parent['value'], out / 'eligibility.json')
    assert value['claims'] == parent['value']['claims']
    result = copy.deepcopy(parent)
    result.update(identity=identity, parent_identity=parent['identity'], value=value)
    result['stages']['route_eligibility'] = trace
    result['unresolved_public_route_checks'] = public_route_request(case, value)[1]
    repaired = trace['flagged'] and trace['success']
    result.update(host_eligibility_changes=0, route_eligibility_producer='LLM', host_orders_routes=repaired, route_order_producer='deterministic_path_length_then_native_id' if repaired else 'LLM')
    result.pop('host_route_choices', None)
    write_record(path, result)
    return result
