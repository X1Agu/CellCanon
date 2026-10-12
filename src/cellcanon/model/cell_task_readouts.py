from __future__ import annotations
import numpy as np

def decode(scores, core_count, extra_count, *, ncore=2000):
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError('Expected rows by genes.')
    result = np.zeros_like(scores, dtype=np.uint8)
    for sl, k in [(slice(0, ncore), int(core_count)), (slice(ncore, None), int(extra_count))]:
        p = scores[:, sl]
        out = result[:, sl]
        if k < 0 or k > p.shape[1]:
            raise ValueError('Predicted set count is outside the gene panel.')
        rank = np.argsort(-p, axis=-1, kind='stable')
        np.put_along_axis(out, rank[:, :k], 1, axis=-1)
    return result
