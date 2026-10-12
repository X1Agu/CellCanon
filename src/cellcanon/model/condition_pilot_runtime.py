from __future__ import annotations
import json

def parsed(receipt):
    try:
        c = receipt['response']['choices'][0]
        if c['finish_reason'] != 'stop':
            return (None, ['incomplete_generation'])
        result = json.loads(c['message']['content'])
        return (result, []) if isinstance(result, dict) else (None, ['not_object'])
    except (KeyError, TypeError, IndexError, ValueError):
        return (None, ['invalid_json'])
