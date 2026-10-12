from __future__ import annotations
import numpy as np

def project_library_mass(expression):
    x = np.asarray(expression)
    if x.ndim != 2 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Expected a finite nonnegative cells-by-genes matrix.')
    counts = np.expm1(x.astype(np.float64))
    mass = counts.sum(1)
    bad = mass > 10000.0
    out = x.astype(np.float32, copy=True)
    out[bad] = np.log1p(counts[bad] * (9999.99 / mass[bad])[:, None]).astype(np.float32)
    if (np.expm1(out.astype(np.float64)).sum(1) > 10000.01).any():
        raise AssertionError('Projection failed the CP10K mass constraint.')
    return out
