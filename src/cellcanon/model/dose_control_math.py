from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.linalg import solve
from scipy.special import ndtr
DOSES = np.array([10.0, 100.0, 1000.0, 10000.0])

def bh(p):
    order = np.argsort(p, axis=-1, kind='stable')
    ranked = np.take_along_axis(p, order, axis=-1)
    q = ranked * p.shape[-1] / np.arange(1, p.shape[-1] + 1)
    q = np.minimum.accumulate(q[..., ::-1], axis=-1)[..., ::-1]
    return np.take_along_axis(np.minimum(q, 1), np.argsort(order, axis=-1), axis=-1)

def group_statistics(x, codes, n_groups, control):
    x = sparse.csc_matrix(x, dtype=np.float64)
    x.eliminate_zeros()
    if np.any(x.data < 0) or not np.isfinite(x.data).all():
        raise ValueError('Expected finite nonnegative log1p CP10K expression.')
    counts = np.bincount(codes[codes >= 0], minlength=n_groups)
    nc = int(control.sum())
    if nc < 2 or np.any(counts < 2):
        raise ValueError('Each expression-testing group requires at least two treated cells and two control cells.')
    mean = np.zeros((n_groups, x.shape[1]))
    cm = np.zeros(x.shape[1])
    p = np.zeros_like(mean)
    sd = np.sqrt(counts * nc * (counts + nc + 1) / 12.0)
    for g in range(x.shape[1]):
        a, b = x.indptr[g:g + 2]
        rows = x.indices[a:b]
        v = x.data[a:b]
        ctrl = np.sort(v[control[rows]])
        nz = nc - len(ctrl)
        cm[g] = ctrl.sum() / nc
        keep = codes[rows] >= 0
        values = v[keep]
        groups = codes[rows[keep]]
        nonzeros = np.bincount(groups, minlength=n_groups)
        mean[:, g] = np.bincount(groups, weights=values, minlength=n_groups) / counts
        lo = np.searchsorted(ctrl, values, side='left')
        hi = np.searchsorted(ctrl, values, side='right')
        u = np.bincount(groups, weights=nz + 0.5 * (lo + hi), minlength=n_groups).astype(float)
        u += (counts - nonzeros) * 0.5 * nz
        p[:, g] = 2 * ndtr(-np.abs((u - counts * nc / 2.0) / sd))
    lfc = mean_to_lfc(mean, cm)
    q = bh(p)
    de = (q < 0.05) & (np.abs(lfc) > 0.25)
    return dict(mean=mean.astype('f4'), control_mean=cm.astype('f4'), lfc=lfc.astype('f4'), p=p.astype('f4'), q=q.astype('f4'), de=de, up=np.where(de, lfc > 0, np.nan).astype('f4'), counts=counts)

def mean_to_lfc(mean, control):
    return np.log2((np.expm1(mean) + 1e-09) / (np.expm1(control) + 1e-09))

def lfc_to_mean(lfc, control):
    return np.log1p(np.maximum((np.expm1(control) + 1e-09) * np.exp2(np.clip(lfc, -60, 60)) - 1e-09, 0))

def dose_features(x, doses=DOSES):
    doses = np.asarray(doses, dtype=float)
    if np.any(doses <= 0):
        raise ValueError('This response head is defined for positive treated doses.')
    u = (np.log10(doses) - 2.5) / 1.5
    basis = np.stack([np.ones_like(u), u, u * u], axis=-1)
    a = np.concatenate([np.ones(x.shape[:-1] + (1,)), x], axis=-1)
    return np.einsum('dcf,lb->dlcfb', a, basis).reshape(len(x) * len(doses), x.shape[1], -1)

def flat(a):
    return a.swapaxes(1, 2).reshape(a.shape[0] * a.shape[2], a.shape[1], *a.shape[3:])

def unflat(a, ndose=4):
    return a.reshape(-1, ndose, *a.shape[1:]).swapaxes(1, 2)

def ridge_fit(x, y, penalty, rank=None):
    centers = x.mean(0)
    intercept = y.mean(0)
    models = []
    for c in range(x.shape[1]):
        a = x[:, c] - centers[c]
        b = y[:, c] - intercept[c]
        if rank is not None:
            u, s, vt = np.linalg.svd(b, full_matrices=False)
            k = min(rank, len(s))
            b = u[:, :k] * s[:k] @ vt[:k]
        models.append(solve(a.T @ a + penalty * np.eye(a.shape[1]), a.T @ b, assume_a='pos'))
    return dict(center=centers, intercept=intercept, beta=np.stack(models))

def ridge_predict(x, model):
    return model['intercept'] + np.einsum('ncf,cfg->ncg', x - model['center'], model['beta'])

def row_corr(a, b):
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    num = np.sum(a * b, axis=-1)
    den = np.sqrt(np.sum(a * a, -1) * np.sum(b * b, -1))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)

def sym_function(a, function, floor=0.0001):
    a = (a + a.swapaxes(-1, -2)) / 2
    w, v = np.linalg.eigh(a)
    w = function(np.maximum(w, floor))
    return v * w[..., None, :] @ v.swapaxes(-1, -2)

def covariance(x, shrink=0.1):
    c = np.cov(x, rowvar=False)
    return (1 - shrink) * c + shrink * np.eye(c.shape[0]) * np.trace(c) / c.shape[0] + 0.0001 * np.eye(c.shape[0])

def covariance_decode(v, rank):
    a = v.reshape(*v.shape[:-1], rank, rank)
    a = 0.5 * (a + a.swapaxes(-1, -2))
    w, q = np.linalg.eigh(a)
    return q * np.exp(np.clip(w, -9, 8))[..., None, :] @ q.swapaxes(-1, -2)

def gaussian_transport(control_cov, target_cov):
    s = sym_function(control_cov, np.sqrt)
    inv = sym_function(control_cov, lambda w: 1 / np.sqrt(w))
    return inv @ sym_function(s @ target_cov @ s, np.sqrt) @ inv
