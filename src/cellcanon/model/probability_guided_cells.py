from __future__ import annotations
import copy
import numpy as np
from cellcanon.model.observation_joint_predictor import retarget_map

def response_experts(response, de, up, shrinkage=8.0):
    response = np.asarray(response, float)
    de = np.asarray(de, bool)
    up = np.asarray(up, float)
    if response.shape != de.shape or response.shape != up.shape or response.ndim != 3:
        raise ValueError('Support drug/cell/gene axes must align.')
    if not np.isfinite(response).all() or shrinkage <= 0:
        raise ValueError('Finite support responses and positive shrinkage required.')
    means = []
    counts = []
    for direction in [0, 1]:
        mask = de & (up == direction)
        n = mask.sum(0)
        sums = np.where(mask, np.clip(response, -8, 8), 0.0).sum(0)
        global_mean = sums.sum(1) / np.maximum(n.sum(1), 1)
        means.append((sums + shrinkage * global_mean[:, None]) / (n + shrinkage))
        counts.append(n)
    return dict(mean=np.stack(means), counts=np.stack(counts), shrinkage=np.array(shrinkage))

def decision_weight(probability, hard_de):
    p = np.asarray(probability, float)
    hard = np.asarray(hard_de, bool)
    if p.shape != hard.shape or p.ndim != 1 or (not np.isfinite(p).all()) or np.any((p < 0) | (p > 1)):
        raise ValueError('Aligned finite DE probability and predicted set required.')
    w = np.empty_like(p)
    for sl in [slice(0, min(2000, len(p))), slice(2000, len(p))]:
        a, h = (p[sl], hard[sl])
        if not len(a):
            continue
        if h.all():
            threshold = max(float(a.min()) / 2, 1e-06)
        elif not h.any():
            threshold = max(0.5, float(a.max()))
        else:
            if a[h].min() + 1e-07 < a[~h].max():
                raise ValueError('Predicted set must be a top-probability set.')
            threshold = float((a[h].min() + a[~h].max()) / 2)
        threshold = np.clip(threshold, 1e-06, 1 - 1e-06)
        numerator = a * (1 - threshold)
        w[sl] = numerator / np.maximum(numerator + threshold * (1 - a), 1e-15)
    return w

def guided_response(original, weight, up_probability, experts, *, strength=1.0, cap=1.0):
    original = np.asarray(original, float)
    w = np.asarray(weight, float)
    u = np.asarray(up_probability, float)
    e = np.asarray(experts, float)
    if original.shape != w.shape or original.shape != u.shape or e.shape != (2, len(original)):
        raise ValueError('Response guidance axes do not align.')
    if not all((np.isfinite(x).all() for x in [original, w, u, e])) or np.any((w < 0) | (w > 1)) or np.any((u < 0) | (u > 1)) or (not 0 <= strength <= 1) or (cap < 0):
        raise ValueError('Invalid response guidance values.')
    target = (1 - u) * e[0] + u * e[1]
    delta = np.clip(strength * w * np.abs(2 * u - 1) * (target - original), -cap, cap)
    return original + delta

def guided_map(params, controls, ids, target_mean, weight, up_probability, detection_experts, *, detection_strength=1.0, shape_strength=0.0):
    if not 0 <= detection_strength <= 1 or shape_strength != 0:
        raise ValueError('Selected detection guidance requires zero shape guidance.')
    p = copy.deepcopy(params)
    w, u, q = map(lambda a: np.asarray(a, float), (weight, up_probability, detection_experts))
    if w.shape != p['target_mean'].shape or u.shape != w.shape or q.shape != (2, len(w)):
        raise ValueError('Detection guidance axes do not align.')
    if any((not np.isfinite(a).all() or np.any((a < 0) | (a > 1)) for a in [w, u, q])):
        raise ValueError('Detection experts and mixture weights must be probabilities.')
    expert = (1 - u) * q[0] + u * q[1]
    mix = detection_strength * w
    p['requested_detection'] = (1 - mix) * p['requested_detection'] + mix * expert
    mix = shape_strength * w
    p['amplitude_retention'] = (1 - mix) * p['amplitude_retention'] + mix * 0.1
    p = retarget_map(p, controls, ids, target_mean)
    p.update(guidance_weight=w, guidance_up_probability=u, guidance_detection_strength=np.asarray(detection_strength), guidance_shape_strength=np.asarray(shape_strength))
    return p
