import copy
import pytest
from cellcanon.model.source_explain_pipeline import normalize_source_positions,run_source_explanation


def test_wire_binding_keeps_labels_reasons_and_source_choices():
    value={'answers':[{'q':0,'label':'supported','refs':[0,'native-b',True,-1,4],'reason':'Original language remains unchanged.'}]}
    saved=copy.deepcopy(value);fixed,count=normalize_source_positions(value,['native-a','native-b'])
    assert count==1 and fixed['answers'][0]['refs']==['native-a','native-b',True,-1,4]
    assert value==saved and fixed['answers'][0]['label']==saved['answers'][0]['label']
    assert fixed['answers'][0]['reason']==saved['answers'][0]['reason']


def test_explicit_private_answer_packet_is_rejected_before_inference(tmp_path):
    class Client:
        model='no-call-test'
        def call(self,*args,**kwargs):raise AssertionError('No model may be called for a private answer packet.')
    with pytest.raises(ValueError,match='public'):run_source_explanation(Client(),{'gold':['answer']},tmp_path)
    assert not list(tmp_path.iterdir())
