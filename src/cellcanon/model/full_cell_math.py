from __future__ import annotations
import numpy as np
from scipy.special import log_ndtr, ndtr
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from cellcanon.model.dose_control_math import bh, row_corr

def logfc(mean, control):
    mean, control = (np.asarray(mean, dtype=np.float64), np.asarray(control, dtype=np.float64))
    return np.log2((np.expm1(mean) + 1e-09) / (np.expm1(control) + 1e-09))

def finish_statistics(sums, u, counts, control_mean, control_n):
    counts = np.asarray(counts)
    if np.any(counts < 2) or control_n < 2:
        raise ValueError('At least two cells per population are required.')
    mean = sums / counts[..., None]
    z = (u - counts[..., None] * control_n / 2) / np.sqrt(counts[..., None] * control_n * (counts[..., None] + control_n + 1) / 12)
    p = 2 * ndtr(-np.abs(z))
    q = bh(p)
    lfc = logfc(mean, control_mean)
    de = (q < 0.05) & (np.abs(lfc) > 0.25)
    de_score = -(np.log(2) + log_ndtr(-np.abs(z))) / np.log(10)
    return dict(mean=mean, lfc=lfc, z=z, p=p, q=q, de=de, up=lfc > 0, de_score=de_score, counts=counts)

def numpy_sufficient(x, control):
    u = np.empty(x.shape[1])
    for g in range(x.shape[1]):
        c = np.sort(control[:, g])
        u[g] = 0.5 * (np.searchsorted(c, x[:, g], 'left') + np.searchsorted(c, x[:, g], 'right')).sum(dtype=np.float64)
    return (np.sum(x, axis=0, dtype=np.float64), u)

def continuous_rows(y, pred):
    y, pred = (np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64))
    sign = lambda a: np.where(np.abs(a) <= 1e-07, 0, np.sign(a))
    return dict(pearson=row_corr(y, pred), spearman=row_corr(rankdata(y, axis=-1), rankdata(pred, axis=-1)), rmse=np.sqrt(np.mean((y - pred) ** 2, axis=-1)), mae=np.abs(y - pred).mean(-1), sign_accuracy_strict=(np.sign(y) == np.sign(pred)).mean(-1), sign_accuracy_zero_tol_1e7=(sign(y) == sign(pred)).mean(-1))

def binary_scores(y, score, hard, weights=None):
    y = np.asarray(y, dtype=int).ravel()
    score = np.asarray(score).ravel()
    hard = np.asarray(hard, dtype=int).ravel()
    weights = None if weights is None else np.broadcast_to(weights, np.shape(hard)).ravel()
    if len(y) == 0:
        return dict(n=0, positives=0)
    args = dict(sample_weight=weights)
    return dict(n=len(y), positives=int(y.sum()), predicted_positives=int(hard.sum()), ap=float(average_precision_score(y, score, **args)) if y.any() else 0.0, auroc=float(roc_auc_score(y, score, **args)) if len(np.unique(y)) == 2 else None, positive_f1=float(f1_score(y, hard, zero_division=0, **args)), macro_f1=float(f1_score(y, hard, labels=[0, 1], average='macro', zero_division=0, **args)), precision=float(precision_score(y, hard, zero_division=0, **args)), recall=float(recall_score(y, hard, zero_division=0, **args)), accuracy=float(accuracy_score(y, hard, **args)))
