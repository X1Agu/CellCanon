"""Route annotation must preserve independent claim mappings."""
import json

from cellcanon import workflow
from cellcanon.model import explanation_cases
from cellcanon.model.explanation_repair import canonical_case
from cellcanon.resources import sciplex_route_knowledge


class SyntheticStore:
    def __init__(self):
        self.closed = False

    def evidence(self, source_id):
        records = {
            'source-0': {'subject': 'synthetic-drug', 'object': 'synthetic-target',
                         'predicate': 'functional_target_action', 'action_sign': -1},
            'source-1': {'subject': 'synthetic-target', 'object': 'synthetic-gene',
                         'predicate': 'regulates', 'action_sign': 1},
        }
        return records[source_id]

    def close(self):
        self.closed = True


def test_prepare_explain_shared_and_serialized_inputs_have_identical_public_contract(tmp_path, monkeypatch):
    packet = {'drug': 'synthetic-drug', 'drug_name': 'Synthetic drug', 'cell': 'context-0',
              'evidence': {'E0': {'source_id': 'source-0'}, 'E1': {'source_id': 'source-1'}},
              'targets': [], 'route_groups': [{'gene': 'synthetic-gene', 'paths': [
                  {'nodes': ['synthetic-target', 'synthetic-gene'], 'evidence': ['E0', 'E1'],
                   'sign': -1, 'tier': 'signed_transcript'}]}]}
    case, private = explanation_cases.build_case(packet, SyntheticStore(), 'test')
    routes = {r['id']: r for r in case['route_candidates']}
    path_claims = [c for c in case['claims'] if c['kind'] in {'path_statement', 'causal_path'}]
    assert path_claims and all(c['arguments'] is routes[c['arguments']['id']] for c in path_claims)
    before = json.dumps(case, sort_keys=True)
    private_before = json.dumps(private, sort_keys=True)
    expected = canonical_case(json.loads(before))

    monkeypatch.setattr(workflow, 'configure_knowledge', lambda config: None)
    monkeypatch.setattr(workflow, 'protocol', lambda: {'test_drugs': ['synthetic-drug']})
    monkeypatch.setattr(workflow.data, 'CELLS', ('context-0',))
    outputs = []
    for name, supplied in [('shared', case), ('serialized', json.loads(before))]:
        run = tmp_path / name
        workflow.write(run / 'knowledge/packets.json', {'synthetic-drug_context-0': packet})
        store = SyntheticStore()
        monkeypatch.setattr(sciplex_route_knowledge, 'RouteKnowledge', lambda: store)
        monkeypatch.setattr(explanation_cases, 'build_case', lambda *args: (supplied, private))
        workflow.prepare_explain({'run_root': run})
        public = workflow.read(run / 'explain_inputs/public.json')
        outputs.append(public)
        assert public == [expected]
        assert workflow.read(run / 'explain_inputs/private.json') == [json.loads(private_before)]
        assert store.closed
        assert all('whole_chain' not in c['arguments'] for c in public[0]['claims'])
        assert all(r['whole_chain'] == ['synthetic-drug', 'synthetic-target', 'synthetic-gene']
                   for r in public[0]['route_candidates'])
        assert json.dumps(supplied, sort_keys=True) == before
    assert outputs[0] == outputs[1]
    assert json.dumps(case, sort_keys=True) == before
    assert json.dumps(private, sort_keys=True) == private_before
    assert all(c['arguments'] is routes[c['arguments']['id']] for c in path_claims)
