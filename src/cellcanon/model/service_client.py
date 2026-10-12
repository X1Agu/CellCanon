"""HTTPS inference transport with credential-safe receipts."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Send credentials only to the explicitly configured completion URL."""

    def http_error_302(self, request, response, code, message, headers):
        raise urllib.error.HTTPError(request.full_url, code,
                                     'Inference redirects are disabled.',
                                     headers, response)

    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302


def wire(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(wire(value).encode()).hexdigest()


def _number(value, *, positive):
    if isinstance(value, bool):
        raise ValueError('Rates and limits must be finite numbers.')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('Rates and limits must be finite numbers.') from None
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError('Rates must be nonnegative and limits must be positive.')
    return value


class ServiceClient:
    """Use caller-supplied rates per million tokens in a common budget unit.

    Connection details stay outside receipts. The neutral model identifier
    binds caches to the configured connection, deployment and request options.
    """

    _RESERVED = frozenset({'model', 'messages', 'temperature', 'max_tokens',
                           'response_format', 'stream'})

    def __init__(self, root, *, endpoint, deployment, credential_env,
                 input_rate, output_rate, cap, run_cap, run_id,
                 request_options=None):
        if (not isinstance(endpoint, str) or not endpoint or
                any(c in endpoint for c in '<>?#') or
                any(c.isspace() or ord(c) < 32 for c in endpoint)):
            raise ValueError('Configure a complete HTTPS inference endpoint.')
        try:
            address = urllib.parse.urlsplit(endpoint)
            valid = (address.scheme == 'https' and bool(address.hostname) and
                     address.username is None and address.password is None and
                     not address.query and not address.fragment)
            address.port
        except ValueError:
            valid = False
        if not valid:
            raise ValueError('The endpoint must use HTTPS without user information, query or fragment.')
        if (not isinstance(deployment, str) or not deployment.strip() or
                any(c in deployment for c in '<>') or
                any(ord(c) < 32 for c in deployment)):
            raise ValueError('Configure a non-placeholder deployment identifier.')
        if (not isinstance(credential_env, str) or
                re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', credential_env) is None):
            raise ValueError('Configure the name of the credential environment variable.')
        credential = os.environ.get(credential_env, '')
        if not credential or any(c in credential for c in '<>\r\n'):
            raise ValueError('The configured credential environment variable is missing or invalid.')
        if request_options is None:
            request_options = {}
        if (not isinstance(request_options, dict) or
                any(not isinstance(k, str) for k in request_options) or
                self._RESERVED.intersection(request_options)):
            raise ValueError('Request options must be an object without reserved fields.')
        try:
            options = json.loads(wire(request_options))
        except (TypeError, ValueError, OverflowError):
            raise ValueError('Request options must contain finite JSON values.') from None
        if not isinstance(run_id, str) or not run_id:
            raise ValueError('A nonempty run identifier is required.')
        self.rates = {'input': _number(input_rate, positive=False),
                      'output': _number(output_rate, positive=False)}
        self.cap = _number(cap, positive=True)
        self.run_cap = _number(run_cap, positive=True)
        self.run_id = run_id
        self.model = 'external-' + digest({'endpoint': endpoint,
            'deployment': deployment, 'options': options})[:32]
        self._endpoint = endpoint
        self._deployment = deployment
        self._credential = credential
        self._request_options = options
        self._opener = urllib.request.build_opener(_NoRedirect())
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _rows(self):
        path = self.root / 'ledger.json'
        return json.loads(path.read_text()) if path.exists() else []

    def _change(self, change):
        with (self.root / 'ledger.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            rows = self._rows()
            result = change(rows)
            path = self.root / 'ledger.json.tmp'
            path.write_text(wire(rows) + '\n')
            path.replace(self.root / 'ledger.json')
            return result

    @staticmethod
    def _charge(row):
        charge = row.get('cost_upper_bound')
        return row['reservation'] if charge is None else charge

    def _response(self, raw):
        """Keep only the fields consumed by the scientific inference pipeline."""
        if not isinstance(raw, dict) or not isinstance(raw.get('choices'), list):
            raise ValueError('Invalid completion response.')
        choices = []
        for choice in raw['choices']:
            if not isinstance(choice, dict) or not isinstance(choice.get('message'), dict):
                raise ValueError('Invalid completion response.')
            message = choice['message']
            content = message.get('content')
            if content is not None and not isinstance(content, str):
                raise ValueError('Invalid completion response.')
            entry = {'message': {'role': message.get('role', 'assistant'),
                                 'content': content},
                     'finish_reason': choice.get('finish_reason')}
            if type(choice.get('index')) is int:
                entry['index'] = choice['index']
            choices.append(entry)
        usage = raw.get('usage')
        usage = usage if isinstance(usage, dict) else {}
        usage = {k: usage[k] for k in ('prompt_tokens', 'completion_tokens')
                 if type(usage.get(k)) is int and usage[k] >= 0}
        return {'model': self.model, 'choices': choices, 'usage': usage}

    def call(self, messages, *, logical_id, max_tokens=700):
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError('A positive integer output-token limit is required.')
        public_request = dict(model=self.model, messages=copy.deepcopy(messages),
            temperature=0.0, max_tokens=max_tokens,
            response_format={'type': 'json_object'}, stream=False)
        key = digest(public_request)
        cache = self.root / (key + '.json')
        if cache.exists():
            return json.loads(cache.read_text()), True
        reservation = ((len(wire(messages).encode()) + 1024) * self.rates['input'] +
                       max_tokens * self.rates['output']) / 1000000.0
        if not math.isfinite(reservation):
            raise ValueError('Request reservation must be finite.')
        attempt = uuid.uuid4().hex

        def begin(rows):
            if sum(self._charge(r) for r in rows) + reservation > self.cap:
                raise RuntimeError('Task inference budget exhausted.')
            if (sum(self._charge(r) for r in rows if r['run_id'] == self.run_id) +
                    reservation > self.run_cap):
                raise RuntimeError('Run inference budget exhausted.')
            rows.append(dict(attempt_id=attempt, logical_id=logical_id,
                run_id=self.run_id, model=self.model, payload_sha256=key,
                reservation=reservation, status='pending', started_unix=time.time()))

        self._change(begin)
        started = time.monotonic()
        receipt = dict(attempt_id=attempt, payload_sha256=key,
                       request=public_request, model=self.model, logical_id=logical_id)
        response = None
        error = None
        try:
            payload = copy.deepcopy(public_request)
            payload['model'] = self._deployment
            payload.update(self._request_options)
            request = urllib.request.Request(self._endpoint,
                data=wire(payload).encode(), headers={
                    'Authorization': 'Bearer ' + self._credential,
                    'Content-Type': 'application/json'})
            with self._opener.open(request, timeout=60) as handle:
                response = self._response(json.load(handle))
        except urllib.error.HTTPError as exc:
            error = 'HTTPError'
            receipt['http_status'] = int(exc.code)
            exc.close()
        except Exception as exc:
            error = type(exc).__name__
        if response is not None:
            receipt['response'] = response
        usage = response['usage'] if response is not None else {}
        known = all(k in usage for k in ('prompt_tokens', 'completion_tokens'))
        charge = ((usage['prompt_tokens'] * self.rates['input'] +
                   usage['completion_tokens'] * self.rates['output']) / 1000000.0
                  if known else None)
        if charge is not None and not math.isfinite(charge):
            known = False
            charge = None
        receipt.update(usage=usage, usage_known=known,
            elapsed_seconds=time.monotonic() - started,
            cost_upper_bound=charge, success=response is not None, error_type=error)
        (self.root / ('attempt_' + attempt + '.json')).write_text(wire(receipt) + '\n')

        def finish(rows):
            row = next(r for r in rows if r['attempt_id'] == attempt)
            row.update(status='success' if response is not None else 'failure',
                usage=usage, usage_known=known, elapsed_seconds=receipt['elapsed_seconds'],
                http_status=receipt.get('http_status'))
            if charge is not None:
                row['cost_upper_bound'] = charge

        self._change(finish)
        if error:
            raise RuntimeError('Completion failed: ' + error +
                               ' ' + str(receipt.get('http_status', ''))) from None
        cache.write_text(wire(receipt) + '\n')
        return receipt, False

    def summary(self):
        rows = self._rows()
        known = [r for r in rows if r.get('usage_known')]
        return dict(physical_calls=len(rows),
            failed_calls=sum(r['status'] == 'failure' for r in rows),
            pending_calls=sum(r['status'] == 'pending' for r in rows),
            unknown_usage_calls=sum(not r.get('usage_known', False) for r in rows),
            known_input_tokens=sum(r['usage']['prompt_tokens'] for r in known),
            known_output_tokens=sum(r['usage']['completion_tokens'] for r in known),
            conservative_charged=sum(self._charge(r) for r in rows), cap=self.cap)
