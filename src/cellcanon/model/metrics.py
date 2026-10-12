from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score

def row_correlation(a, b):
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide((a * b).sum(-1), denominator, out=np.full(a.shape[:-1], np.nan), where=denominator > 1e-14)

def safe_ap(y, p):
    return float(average_precision_score(y, p)) if len(y) and np.unique(y).size == 2 else float('nan')
