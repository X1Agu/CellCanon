from __future__ import annotations
import json
from pathlib import Path
from cellcanon.model.supported_act import digest

def checked_call(client, messages, path, validate, *, logical_id, max_tokens=1400):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = digest(messages)
    if path.exists():
        old = json.loads(path.read_text())
        if old['request_sha256'] != identity:
            raise ValueError('The saved response belongs to a different request. Choose a new output path.')
        return old
    result = dict(request_sha256=identity, logical_id=logical_id, success=False, attempts=[])
    current = messages
    for attempt in range(2):
        try:
            receipt, cached = client.call(current, logical_id=logical_id + ':' + str(attempt), max_tokens=max_tokens)
            raw = receipt['response']['choices'][0]['message']['content']
            result['attempts'].append(dict(attempt_id=receipt['attempt_id'], cached=cached, raw=raw))
            value = json.loads(raw)
            result['value'] = validate(value)
            result['success'] = True
            break
        except (ValueError, KeyError, TypeError, RuntimeError, IndexError) as exc:
            result.setdefault('errors', []).append(str(exc)[:400])
            if attempt == 0:
                current = messages + [dict(role='user', content='Return the required JSON with valid supplied IDs and bounded values. Previous validation issue: ' + str(exc)[:250])]
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result
