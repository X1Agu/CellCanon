from __future__ import annotations
from pathlib import Path
from cellcanon.model.condition_llm_packets import write
from cellcanon.model.condition_pilot_knowledge import SYSTEM, packet, normalized, verify
from cellcanon.model.condition_pilot_runtime import parsed
from cellcanon.model.service_client import digest, wire
from cellcanon.model.unified_atlas_training import read

def public_packet(card, cell):
    value = packet(card, cell)
    forbidden = {'logfoldchanges', 'de_label_strict', 'direction_label', 'test_labels', 'query_response', 'true_response', 'hidden_truth'}

    def walk(item):
        if isinstance(item, dict):
            if forbidden & item.keys():
                raise ValueError('Response labels cannot enter compiler inputs.')
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)
    walk(value)
    return value

def compile_one(client, card, cell, output, max_tokens=900):
    p = public_packet(card, cell)
    cache_identity = digest({'interface': 'evidence-compilation-v1', 'packet': p,
                             'model': client.model, 'system': SYSTEM, 'max_tokens': max_tokens})
    path = Path(output) / (card['drug'] + '_' + cell + '.json')
    if path.exists():
        record = read(path)
        if record['packet_sha256'] != digest(p):
            raise ValueError('Cached compilation belongs to another packet.')
        if record.get('cache_identity') != cache_identity:
            raise ValueError('Cached compilation uses different or unrecorded model/prompt settings. Use a fresh output directory.')
        return record
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': wire(p)}]
    current = messages
    versions = []
    result = None
    errors = ['not_attempted']
    for attempt in range(2):
        try:
            receipt, cached = client.call(current, logical_id='full-compile:' + card['drug'] + ':' + cell + ':' + str(attempt), max_tokens=max_tokens)
            raw, errors = parsed(receipt)
            result = normalized(p, raw)
            if not errors:
                errors = verify(p, result)
            versions.append({'attempt_id': receipt['attempt_id'], 'cached': cached, 'errors': errors, 'result': result, 'raw_result': raw, 'model_configuration_id': client.model, 'finish_reason': receipt['response']['choices'][0].get('finish_reason')})
            if not errors:
                break
            current = messages + [{'role': 'assistant', 'content': receipt['response']['choices'][0]['message']['content']}, {'role': 'user', 'content': 'Fix the JSON using only the original packet. Errors: ' + wire(errors)}]
        except RuntimeError as exc:
            if 'budget' in str(exc).lower():
                raise
            errors = [str(exc)[:160]]
            versions.append({'attempt_id': None, 'errors': errors, 'transport_failure': True})
            current = messages + [{'role': 'user', 'content': 'Return the required complete JSON using only the supplied packet.'}]
    record = {'drug': card['drug'], 'cell': cell, 'packet_sha256': digest(p), 'cache_identity': cache_identity, 'valid': not errors, 'result': result, 'errors': errors, 'versions': versions, 'model_configuration_id': client.model}
    write(path, record)
    return record
