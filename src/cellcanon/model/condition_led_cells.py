from __future__ import annotations
import numpy as np
from cellcanon.model import mean_preserving_cells as base
from cellcanon.model.cell_library_projection import project_library_mass

def class_detection_experts(detection, control_detection, de, up, *, shrinkage=8.0):
    q = np.asarray(detection, float)
    q0 = np.asarray(control_detection, float)
    de = np.asarray(de, bool)
    up = np.asarray(up)
    if q.shape != de.shape or up.shape != de.shape or q0.shape != (q.shape[1], q.shape[3]):
        raise ValueError('Detection and class labels must align.')
    delta = q - q0[None, :, None, :]
    experts = []
    counts = []
    for is_up in [False, True]:
        mask = de & (up == is_up)
        n = mask.sum((0, 2))
        sums = np.where(mask, delta, 0.0).sum((0, 2))
        global_delta = sums.sum(1) / np.maximum(n.sum(1), 1)
        estimate = (sums + shrinkage * global_delta[:, None]) / (n + shrinkage)
        experts.append(np.clip(q0 + estimate, 0.0, 1.0))
        counts.append(n)
    return dict(probability=np.stack(experts), support_counts=np.stack(counts), shrinkage=np.asarray(shrinkage))

def _fields(x, ids, p):
    ncore = p['components'].shape[1]
    latent = np.einsum('ng,kg->nk', x[:, :ncore] - p['state_center'], p['components'], optimize=False)
    moved = np.einsum('nk,kl->nl', latent, p['transform'].T - np.eye(len(p['components'])), optimize=False)
    change = np.einsum('nk,kg->ng', moved, p['components'], optimize=False)
    raw = x.copy()
    raw[:, :ncore] += change * p['state_strength']
    score = raw + base._jitter(ids, x.shape[1]) * 1e-06
    scale = p['positive_scale']
    local = np.logaddexp(0.0, raw / scale) * scale
    positive = scale + p['amplitude_retention'] * (local - scale)
    return (score, positive)

def prepare_condition_map(reference_controls, genes, sample_ids, *, control_mean, control_detection, components, state_center, transform, target_mean, learned_detection, class_detection, hard_de, up_probability, expert_strength=0.0, amplitude_retention=0.25, state_strength=0.1):
    genes = np.asarray(genes, dtype=str)
    x, ids = base._validate(reference_controls, genes, sample_ids, genes)
    if len(x) < 2 or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique sample IDs and at least two controls required.')
    if not all((0 <= v <= 1 for v in [expert_strength, amplitude_retention, state_strength])):
        raise ValueError('Mixture weights must be in [0, 1].')
    if amplitude_retention == 0:
        raise ValueError('A nonzero control-amplitude contribution is required.')
    g = len(genes)
    cm, q0, target, q, up = [np.asarray(v, float) for v in [control_mean, control_detection, target_mean, learned_detection, up_probability]]
    hard = np.asarray(hard_de, bool)
    expert = np.asarray(class_detection, float)
    if any((v.shape != (g,) for v in [cm, q0, target, q, up, hard])) or expert.shape != (2, g):
        raise ValueError('Condition arrays must align with genes.')
    if any((not np.isfinite(v).all() for v in [cm, q0, target, q, up, expert])):
        raise ValueError('Condition arrays must be finite.')
    if np.any(target < 0) or np.any(target > base.MAX_EXPRESSION):
        raise ValueError('Mean outside feasible per-gene range.')
    if expert_strength != 0:
        raise ValueError('The selected main model applies probability guidance after this map.')
    q = np.clip(q, 0.0, 1.0)
    quota = np.maximum(np.rint(q * len(x)).astype(int), np.ceil(target * len(x) / base.MAX_EXPRESSION).astype(int))
    quota = np.where(target > 0, quota, 0).clip(0, len(x))
    p = dict(genes=genes, control_mean=cm, target_mean=target, components=np.asarray(components, float), state_center=np.asarray(state_center, float), transform=np.asarray(transform, float), state_strength=np.asarray(state_strength), positive_scale=np.maximum(np.divide(cm, q0, out=np.zeros(g), where=q0 > 0), 0.001), amplitude_retention=np.where(hard, amplitude_retention, 1.0), requested_detection=q, quota=quota, reference_n=np.asarray(len(x)), hard_de=hard, up_probability=up, expert_strength=np.asarray(expert_strength), upper=np.asarray(base.MAX_EXPRESSION))
    score, positive = _fields(x, ids, p)
    ordered = np.sort(score, axis=0)
    lo = ordered[np.clip(len(x) - quota - 1, 0, len(x) - 1), np.arange(g)]
    hi = ordered[np.clip(len(x) - quota, 0, len(x) - 1), np.arange(g)]
    p.update(cutoff=lo + (hi - lo) / 2, all_positive=quota == len(x), all_zero=quota == 0)
    mask = (score >= p['cutoff']) | p['all_positive']
    mask[:, p['all_zero']] = False
    if not np.array_equal(mask.sum(0), quota):
        raise AssertionError('Rank-based detection counts do not match the reference quotas.')
    p['gain'] = base._capped_gain(positive, mask, target, base.MAX_EXPRESSION)
    return p

def predict_condition_cells(control_input, genes, sample_ids, params, *, project=True):
    x, ids = base._validate(control_input, np.asarray(genes, dtype=str), sample_ids, params['genes'])
    score, positive = _fields(x, ids, params)
    mask = (score >= params['cutoff']) | params['all_positive']
    mask[:, params['all_zero']] = False
    y = np.where(mask, np.minimum(positive * params['gain'], params['upper']), 0.0).astype('f4')
    return project_library_mass(y) if project else y
