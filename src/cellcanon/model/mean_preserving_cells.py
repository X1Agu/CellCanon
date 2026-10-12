from __future__ import annotations
import numpy as np
MAX_EXPRESSION = float(np.log1p(10000.0))

def _validate(x, genes, ids, expected):
    x = np.asarray(x, dtype=np.float64)
    ids = np.asarray(ids)
    if x.ndim != 2 or x.shape[1] != len(genes) or (not np.array_equal(genes, expected)):
        raise ValueError('Gene identity/order or expression shape mismatch.')
    if len(set(genes)) != len(genes) or ids.shape != (len(x),):
        raise ValueError('Unique genes and one stable sample ID per row are required.')
    if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
        raise ValueError('Sample IDs must be nonnegative integers.')
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('Controls must be finite and nonnegative.')
    return (x, ids.astype(np.uint64))

def _jitter(ids, ngenes):
    with np.errstate(over='ignore'):
        z = ids[:, None] + np.arange(ngenes, dtype=np.uint64)[None, :] * np.uint64(11400714819323198485) + np.uint64(20260913)
        z = (z ^ z >> np.uint64(30)) * np.uint64(13787848793156543929)
        z = (z ^ z >> np.uint64(27)) * np.uint64(10723151780598845931)
        z = z ^ z >> np.uint64(31)
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / 2 ** 53)

def _capped_gain(positive, mask, target, upper):
    weights = np.where(mask, positive, 0.0)
    total = target * len(weights)
    capped = np.zeros_like(mask)
    for _ in range(len(weights) + 1):
        remaining = np.maximum(total - upper * capped.sum(0), 0.0)
        free_sum = np.where(capped, 0.0, weights).sum(0)
        gain = np.divide(remaining, free_sum, out=np.zeros_like(total), where=free_sum > 0)
        new = mask & (weights * gain > upper)
        if np.array_equal(new, capped):
            result = np.minimum(weights * gain, upper)
            if not np.allclose(result.mean(0), target, atol=1e-09, rtol=1e-09):
                raise AssertionError('Capped normalization failed to preserve target means.')
            return gain
        capped = new
    raise AssertionError('Capped normalization did not converge.')
