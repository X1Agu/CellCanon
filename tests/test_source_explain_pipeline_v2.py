import copy
from types import SimpleNamespace
import pytest
from cellcanon.model import source_explain_pipeline_v2 as module


def test_private_input_rejected_before_cache_or_client(tmp_path):
    with pytest.raises(ValueError, match='public'):
        module.run_complete_source_explanation(None, {'gold': []}, tmp_path)


def test_route_adapter_preserves_claims_and_attributes_host_order(tmp_path, monkeypatch):
    parent = dict(identity='parent', value=dict(claims=[dict(id='q', label='supported')]),
                  stages={}, host_route_choices=0, host_semantic_label_changes=0)
    monkeypatch.setattr(module, 'run_source_explanation', lambda *a: copy.deepcopy(parent))
    def review(client, case, value, path):
        return dict(**value, route_ranking=['r']), dict(flagged=True, success=True)
    monkeypatch.setattr(module, 'review_eligibility', review)
    monkeypatch.setattr(module, 'public_route_request', lambda *a: ({}, []))
    client = SimpleNamespace(model='fixture')
    result = module.run_complete_source_explanation(client, {'id': 'c'}, tmp_path)
    assert result['value']['claims'] == parent['value']['claims']
    assert result['host_orders_routes'] and result['host_eligibility_changes'] == 0
    assert result['route_order_producer'] == 'deterministic_path_length_then_native_id'
    assert 'host_route_choices' not in result
    assert module.run_complete_source_explanation(client, {'id': 'c'}, tmp_path) == result
    with pytest.raises(ValueError, match='new output directory'):
        module.run_complete_source_explanation(client, {'id': 'changed'}, tmp_path)
