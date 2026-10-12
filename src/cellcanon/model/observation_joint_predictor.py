from __future__ import annotations
import copy
import numpy as np
from cellcanon.model.condition_led_cells import _fields
from cellcanon.model.mean_preserving_cells import _capped_gain, _validate, MAX_EXPRESSION

def retarget_map(params, reference_controls, reference_ids, target_mean):
    p = copy.deepcopy(params)
    target = np.asarray(target_mean, float)
    reference_controls, reference_ids = _validate(reference_controls, p['genes'], reference_ids, p['genes'])
    if target.shape != p['target_mean'].shape or not np.isfinite(target).all():
        raise ValueError('Invalid target mean.')
    target = np.clip(target, 0, MAX_EXPRESSION)
    p['target_mean'] = target
    score, positive = _fields(reference_controls, reference_ids, p)
    n = len(reference_controls)
    g = score.shape[1]
    quota = np.maximum(np.rint(p['requested_detection'] * n).astype(int), np.ceil(target * n / MAX_EXPRESSION).astype(int))
    quota = np.where(target > 0, quota, 0).clip(0, n)
    order = np.sort(score, axis=0)
    lo = order[np.clip(n - quota - 1, 0, n - 1), np.arange(g)]
    hi = order[np.clip(n - quota, 0, n - 1), np.arange(g)]
    p.update(quota=quota, cutoff=lo + (hi - lo) / 2, all_positive=quota == n, all_zero=quota == 0)
    mask = (score >= p['cutoff']) | p['all_positive']
    mask[:, p['all_zero']] = False
    if not np.array_equal(mask.sum(0), quota):
        raise ValueError('Frozen reference ranks cannot reproduce the new quota.')
    p['gain'] = _capped_gain(positive, mask, target, MAX_EXPRESSION)
    return p
