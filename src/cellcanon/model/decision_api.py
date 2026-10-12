"""JSON decision interface for an externally configured inference service."""
from __future__ import annotations
import copy
from cellcanon.model.service_client import ServiceClient


class DecisionClient(ServiceClient):
    def call(self, messages, *, logical_id, max_tokens=550):
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError('Positive output-token limit required.')
        messages = copy.deepcopy(messages)
        if not any('json' in m.get('content', '').lower() for m in messages):
            messages.insert(0, dict(role='system',
                content='Return valid JSON matching the supplied example.'))
        return super().call(messages, logical_id=logical_id, max_tokens=max_tokens)
