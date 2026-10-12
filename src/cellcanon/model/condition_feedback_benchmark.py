from __future__ import annotations
import hashlib
import json
import numpy as np

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float64).tobytes()).hexdigest()

def condition_partition(drugs, cells, seed=20260916, identities=None):
    observation, hidden = ([], [])
    for di, drug in enumerate(drugs):
        identity = identities[drug] if identities is not None else drug
        ci = int(hashlib.sha256(f'{seed}|{identity}'.encode()).hexdigest(), 16) % len(cells)
        hidden.append(di * len(cells) + ci)
        observation.extend((di * len(cells) + c for c in range(len(cells)) if c != ci))
    return {'seed': seed, 'observation': observation, 'hidden': hidden, 'identities': [identities[d] if identities is not None else d for d in drugs], 'unit': 'canonical drug identity x cell line; 24h pooled dose; complete condition assay'}
