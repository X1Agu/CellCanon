from __future__ import annotations
import numpy as np
from scipy.special import expit, logit
from cellcanon.model.cell_task_readouts import decode

def state_probabilities(de, up):
    return np.stack([1 - de, de * (1 - up), de * up], axis=-1)

def label_states(de, up):
    de = np.asarray(de)
    up = np.asarray(up)
    if de.shape != up.shape or not np.isin(de, [0, 1]).all() or (not np.isfinite(up[de == 1]).all()):
        raise ValueError('Valid source DE and conditional direction required')
    return np.where(de == 1, 1 + np.nan_to_num(up), 0).astype(int)

class ClassResponseTransfer:

    def __init__(self, support_de, support_up):
        self.de = np.asarray(support_de, float).copy()
        self.up = np.asarray(support_up, float).copy()
        self.states = label_states(self.de, self.up)
        self.shape = self.de.shape[1:]
        self.cache = {}

    def likelihood(self, target, source, task, smoothing):
        key = (target, source, task, smoothing)
        if key in self.cache:
            return self.cache[key]
        if task not in ('de', 'up') or smoothing <= 0:
            raise ValueError('Known task and positive smoothing required')
        valid = np.ones_like(self.de[:, target], bool) if task == 'de' else self.de[:, target] == 1
        y = self.de[:, target] if task == 'de' else np.nan_to_num(self.up[:, target])
        counts = np.array([[(valid & (y == t) & (self.states[:, source] == s)).sum(0) for s in range(3)] for t in range(2)], float)
        pooled = counts.sum(2) + 0.5
        pooled /= pooled.sum(1, keepdims=True)
        probs = (counts + smoothing * pooled[..., None]) / (counts.sum(1, keepdims=True) + smoothing)
        llr = np.log(np.maximum(probs[1], 1e-09)) - np.log(np.maximum(probs[0], 1e-09))
        self.cache[key] = np.clip(llr.T, -4, 4)
        return self.cache[key]

    def evidence(self, de_prior, up_prior, observations, task, config):
        d = np.asarray(de_prior, float)
        u = np.asarray(up_prior, float)
        if d.shape != self.shape or u.shape != self.shape or (not np.isfinite(d).all()) or (not np.isfinite(u).all()):
            raise ValueError('Finite aligned parent heads required')
        out = np.zeros_like(d)
        for c, source_probs in observations.items():
            p = np.asarray(source_probs, float)
            if type(c) is not int or not 0 <= c < self.shape[0] or p.shape != (self.shape[1], 3) or (not np.isfinite(p).all()) or np.any(p < 0) or (not np.allclose(p.sum(1), 1)):
                raise ValueError('Aligned source state probabilities required')
            expected = state_probabilities(d[c], u[c])
            for target in range(self.shape[0]):
                if target in observations:
                    continue
                if config.get('family', 'likelihood') != 'likelihood':
                    raise ValueError('Only the selected likelihood transfer is included.')
                out[target] += np.sum((p - expected) * self.likelihood(target, c, task, config['smoothing']), axis=1)
        return out

    def probabilities(self, parent, evidence, config):
        strength = float(config['strength'])
        if not 0 <= strength <= 2:
            raise ValueError('Bounded evidence strength required')
        return expit(logit(np.clip(parent, 1e-06, 1 - 1e-06)) + strength * evidence)

def decode_feedback(probability, original_probability, original_hard, margin=1.0, count_scale=1.0):
    p = np.clip(original_probability, 1e-06, 1 - 1e-06)
    q = np.clip(probability, 1e-06, 1 - 1e-06)
    scores = logit(q) + margin * (2 * np.asarray(original_hard, float) - 1)
    counts = [min(sl.stop - sl.start, max(0, int(round(original_hard[sl].sum() * count_scale)))) for sl in [slice(0, min(2000, len(p))), slice(min(2000, len(p)), len(p))]]
    return decode(scores[None], *counts, ncore=min(2000, len(p)))[0]
