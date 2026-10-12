from __future__ import annotations
import numpy as np

def count_features(observations, target):
    features = []
    for c in range(3):
        if c in observations:
            o = observations[c]
            de = np.asarray(o['de'])
            up = np.nan_to_num(o['up'])
            y = np.asarray(o['continuous'])
            core = len(y)
            features.extend([1.0, np.log1p(de[:core].sum()), np.log1p(de[core:].sum()), float((de * up).sum() / max(de.sum(), 1)), float(np.mean(np.tanh(y / 5))), float(np.mean(np.tanh(abs(y) / 5))), float(np.mean(y == 0))])
        else:
            features.extend([0.0] * 7)
    features.extend(np.eye(3)[target])
    return np.array(features, float)

def decode_count_feedback(probability, original_hard, predicted_log_counts, blend, baseline_scale=1.5):
    from .cell_task_readouts import decode
    g = len(probability)
    core = min(2000, g)
    base = np.array([original_hard[:core].sum(), original_hard[core:].sum()], float) * baseline_scale
    proposed = np.maximum(np.expm1(np.clip(predicted_log_counts, 0, np.log1p(g))), 0)
    count = np.rint((1 - blend) * base + blend * proposed).astype(int)
    count = np.clip(count, 0, [core, g - core])
    return (decode(np.asarray(probability)[None], int(count[0]), int(count[1]), ncore=core)[0], count)
