"""Offline checks for externally configured model transport and private metadata."""
import copy
import io
import json
import uuid
import urllib.error
import urllib.request
import urllib.response
from email.message import Message
import pytest
from cellcanon import workflow


def test_compiler_cache_is_bound_to_model_prompt_and_generation_settings(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from cellcanon.model import full_condition_evidence as compiler
    from cellcanon.model.service_client import digest
    packet={'evidence': []}
    monkeypatch.setattr(compiler,'public_packet',lambda *_:packet)
    monkeypatch.setattr(compiler,'parsed',lambda receipt:({'selected':[]},[]))
    monkeypatch.setattr(compiler,'normalized',lambda packet,value:value)
    monkeypatch.setattr(compiler,'verify',lambda *_:[])
    calls=[]
    def call(*args,**kwargs):
        calls.append(kwargs)
        return {'attempt_id':'synthetic','response':{'choices':[{'finish_reason':'stop'}]}},False
    client=SimpleNamespace(model='configuration-a',call=call)
    first=compiler.compile_one(client,{'drug':'D1'},'A549',tmp_path)
    assert compiler.compile_one(client,{'drug':'D1'},'A549',tmp_path)==first
    assert len(calls)==1
    client.model='configuration-b'
    with pytest.raises(ValueError,match='settings'):
        compiler.compile_one(client,{'drug':'D1'},'A549',tmp_path)
    client.model='configuration-a'
    with pytest.raises(ValueError,match='settings'):
        compiler.compile_one(client,{'drug':'D1'},'A549',tmp_path,max_tokens=901)
    monkeypatch.setattr(compiler,'SYSTEM',compiler.SYSTEM+' New instruction.')
    with pytest.raises(ValueError,match='settings'):
        compiler.compile_one(client,{'drug':'D1'},'A549',tmp_path)
    assert len(calls)==1


def test_compiler_legacy_cache_requires_fresh_output_directory(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from cellcanon.model import full_condition_evidence as compiler
    from cellcanon.model.service_client import digest
    packet={'evidence':[]}
    monkeypatch.setattr(compiler,'public_packet',lambda *_:packet)
    (tmp_path/'D1_A549.json').write_text(json.dumps({'packet_sha256':digest(packet)}))
    with pytest.raises(ValueError,match='fresh output directory'):
        compiler.compile_one(SimpleNamespace(model='configuration-a'),{'drug':'D1'},'A549',tmp_path)


def make_config(tmp_path,monkeypatch,**changes):
    secret=uuid.uuid4().hex
    monkeypatch.setenv('TEST_MODEL_CREDENTIAL',secret)
    config={'run_root':tmp_path,'language_model':'private-test-deployment',
        'model_service_endpoint':'https://service.invalid/completion',
        'model_service_credential_env':'TEST_MODEL_CREDENTIAL','api_budget':1.,
        'api_rates_per_million':{'input':1.,'output':2.},
        'model_request_options':{'private_option':'private-runtime-setting'}}
    config.update(changes)
    return config,secret


def saved_text(root):
    return '\n'.join(p.read_text() for p in root.rglob('*.json'))


def assert_private_fields_absent(text,config,secret):
    for value in [secret,config['model_service_endpoint'],config['language_model'],
                  'private-runtime-setting','private-response-id','private-response-metadata']:
        assert value not in text


def test_configured_request_keeps_private_metadata_out_of_receipts_and_reuses_cache(tmp_path,monkeypatch):
    config,secret=make_config(tmp_path,monkeypatch)
    client=workflow.client(config,'test')
    calls=[]
    def transport(request,timeout):
        calls.append(request)
        assert request.full_url==config['model_service_endpoint']
        assert request.get_header('Authorization')=='Bearer '+secret
        payload=json.loads(request.data)
        assert payload['model']==config['language_model']
        assert payload['private_option']=='private-runtime-setting'
        return io.BytesIO(json.dumps({'id':'private-response-id','model':config['language_model'],
            'service_metadata':'private-response-metadata',
            'choices':[{'index':0,'message':{'role':'assistant','content':'{"answer": 1}',
                'internal_metadata':'private-response-metadata'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':100,'completion_tokens':10,'private_field':'private-response-metadata'}}).encode())
    monkeypatch.setattr(client._opener,'open',transport)
    messages=[{'role':'user','content':'Return JSON.'}];original=copy.deepcopy(messages)
    receipt,cached=client.call(messages,logical_id='test',max_tokens=40)
    assert not cached and json.loads(receipt['response']['choices'][0]['message']['content'])=={'answer':1}
    assert receipt['response']['model']==client.model
    assert receipt['usage']=={'prompt_tokens':100,'completion_tokens':10}
    assert messages==original
    repeated,cached=client.call(messages,logical_id='test',max_tokens=40)
    assert cached and repeated==receipt and len(calls)==1
    assert client.summary()['physical_calls']==1
    assert_private_fields_absent(saved_text(tmp_path),config,secret)


def test_configuration_identity_changes_when_service_or_request_options_change(tmp_path,monkeypatch):
    config,_=make_config(tmp_path,monkeypatch)
    first=workflow.client(config,'test')
    options=workflow.client({**config,'model_request_options':{'private_option':'changed'}},'test')
    endpoint=workflow.client({**config,'model_service_endpoint':'https://other.invalid/completion'},'test')
    assert len({first.model,options.model,endpoint.model})==3


def test_http_failure_does_not_record_service_error_body(tmp_path,monkeypatch):
    config,secret=make_config(tmp_path,monkeypatch)
    client=workflow.client(config,'test')
    body=' '.join([secret,config['model_service_endpoint'],config['language_model'],'private-response-metadata'])
    def transport(request,timeout):
        raise urllib.error.HTTPError(request.full_url,429,'private-response-metadata',{},io.BytesIO(body.encode()))
    monkeypatch.setattr(client._opener,'open',transport)
    with pytest.raises(RuntimeError) as exc:
        client.call([{'role':'user','content':'Return JSON.'}],logical_id='test',max_tokens=40)
    assert_private_fields_absent(str(exc.value)+'\n'+saved_text(tmp_path),config,secret)
    assert client.summary()['failed_calls']==1


def test_budget_is_checked_before_transport(tmp_path,monkeypatch):
    config,_=make_config(tmp_path,monkeypatch,api_budget=1e-10)
    client=workflow.client(config,'test')
    def transport(*args,**kwargs):raise AssertionError('Transport must not run over budget.')
    monkeypatch.setattr(client._opener,'open',transport)
    with pytest.raises(RuntimeError,match='budget'):
        client.call([{'role':'user','content':'Return JSON.'}],logical_id='test',max_tokens=40)


def test_placeholder_configuration_requires_explicit_external_values(tmp_path,monkeypatch):
    config,_=make_config(tmp_path,monkeypatch,model_service_endpoint='<MODEL_SERVICE_ENDPOINT>')
    with pytest.raises(ValueError):workflow.client(config,'test')


def test_request_options_cannot_override_the_requested_model_or_output_budget(tmp_path,monkeypatch):
    config,_=make_config(tmp_path,monkeypatch,model_request_options={'model':'changed','max_tokens':100000})
    with pytest.raises(ValueError):workflow.client(config,'test')


@pytest.mark.parametrize('status', [301, 302, 303, 307, 308])
@pytest.mark.parametrize('destination', [
    'https://other.invalid/collect',
    'http://other.invalid/collect',
    'https://service.invalid/moved',
])
def test_redirect_never_sends_a_second_request_or_records_private_details(
        tmp_path, monkeypatch, status, destination):
    config, secret = make_config(tmp_path, monkeypatch)
    client = workflow.client(config, 'test')
    requests = []
    responses = []

    class FakeHTTPS(urllib.request.HTTPSHandler):
        def https_open(self, request):
            requests.append(request)
            headers = Message()
            headers['Location'] = destination
            response = urllib.response.addinfourl(
                io.BytesIO(('private-response-metadata ' + secret).encode()),
                headers, request.full_url, status)
            response.msg = 'Found'
            responses.append(response)
            return response

    class RejectHTTP(urllib.request.HTTPHandler):
        def http_open(self, request):
            requests.append(request)
            raise AssertionError('A credential-bearing request must never reach HTTP.')

    # Exercise the real opener/error-handler chain without sockets or DNS.
    handlers = [handler for handler in client._opener.handlers
                if not isinstance(handler, (urllib.request.HTTPSHandler,
                                            urllib.request.HTTPHandler))]
    client._opener = urllib.request.build_opener(*handlers, FakeHTTPS(), RejectHTTP())
    with pytest.raises(RuntimeError, match='Completion failed: HTTPError ' + str(status)) as error:
        client.call([{'role': 'user', 'content': 'Return JSON.'}], logical_id='redirect', max_tokens=40)
    assert len(requests) == 1
    assert requests[0].full_url == config['model_service_endpoint']
    assert requests[0].get_header('Authorization') == 'Bearer ' + secret
    assert responses[0].closed
    text = str(error.value) + '\n' + saved_text(tmp_path)
    assert destination not in text
    assert_private_fields_absent(text, config, secret)
    assert client.summary()['failed_calls'] == 1
    assert client.summary()['unknown_usage_calls'] == 1
    assert client.summary()['conservative_charged'] > 0


def test_insecure_initial_endpoint_is_rejected(tmp_path, monkeypatch):
    config, _ = make_config(tmp_path, monkeypatch,
                           model_service_endpoint='http://service.invalid/completion')
    with pytest.raises(ValueError, match='HTTPS'):
        workflow.client(config, 'test')
