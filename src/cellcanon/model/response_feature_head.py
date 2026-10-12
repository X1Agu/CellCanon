from __future__ import annotations
import numpy as np

class ResponseFeatureBuilder:

    def __init__(self, de, up):
        self.de = np.asarray(de, float)
        self.up = np.where(self.de > 0, np.nan_to_num(up), 0.0)
        self.n = len(self.de)

    def source_rates(self, exclude=None):
        n = self.n if exclude is None else self.n - 1
        ds = self.de.sum(0) - (0 if exclude is None else self.de[exclude])
        us = self.up.sum(0) - (0 if exclude is None else self.up[exclude])
        return ((ds + 0.5) / (n + 1), (us + 0.5) / (ds + 1))

    def features(self, parent, heads, observations, target, exclude=None):
        cells = sorted(observations)
        g = heads['de'].shape[-1]
        core = parent.shape[-1]
        rate, urate = self.source_rates(exclude)
        values = []

        def pad(x):
            return np.r_[np.asarray(x, float), np.zeros(g - core)]
        srcde = np.stack([observations[c]['de'] for c in cells])
        srcup = np.stack([observations[c]['up'] for c in cells])
        raw = np.stack([pad(observations[c]['continuous']) for c in cells])
        pr = np.stack([pad(parent[c]) for c in cells])
        res = raw - pr
        values += [srcde.mean(0), (srcde * srcup).mean(0), (srcde * (1 - srcup)).mean(0), np.tanh(raw / 5).mean(0), np.tanh(abs(raw) / 5).mean(0), (raw == 0).mean(0), np.tanh(pr / 5).mean(0), np.tanh(res / 5).mean(0), np.tanh(abs(res) / 5).mean(0), np.tanh(pad(parent[target]) / 5)]
        values += [rate[target], urate[target], rate[cells].mean(0), urate[cells].mean(0), np.r_[np.ones(core), np.zeros(g - core)]]
        values += [np.full(g, int(target == c)) for c in range(3)] + [np.full(g, int(c in cells)) for c in range(3)]
        values += [np.full(g, srcde.mean()), np.full(g, np.mean(np.tanh(abs(raw[:, :core]) / 5))), np.full(g, len(cells))]
        return np.stack(values, axis=1).astype(np.float32)

class FeatureResponseUpdate:

    def __init__(self, support_de, support_up, models, configs, decoder):
        self.builder = ResponseFeatureBuilder(support_de, support_up)
        self.models = models
        self.configs = configs
        self.decoder = decoder

    def apply(self, parent, prior_heads, current_heads, observations, target):
        from scipy.special import expit, logit
        from .class_response_transfer import decode_feedback
        result = {k: current_heads[k][target].copy() for k in ['de', 'up', 'hard_de']}
        if not observations:
            return result
        expected = {c: {'continuous': parent[c], 'de': prior_heads['de'][c], 'up': prior_heads['up'][c]} for c in observations}
        actual = {c: {k: np.asarray(v).copy() for k, v in r.items()} for c, r in observations.items()}
        x = self.builder.features(parent, prior_heads, actual, target)
        counter = self.builder.features(parent, prior_heads, expected, target)
        for task in ['de', 'up']:
            strength = self.configs[task]['strength']
            if strength == 0:
                continue
            model = self.models[task]
            a = logit(np.clip(model.predict_proba(x)[:, 1], 1e-05, 1 - 1e-05))
            b = logit(np.clip(model.predict_proba(counter)[:, 1], 1e-05, 1 - 1e-05))
            delta = np.clip(a - b, -6, 6)
            if not np.any(abs(delta) > 1e-12):
                continue
            result[task] = expit(logit(np.clip(current_heads[task][target], 1e-06, 1 - 1e-06)) + strength * delta)
            if task == 'de':
                result['hard_de'] = decode_feedback(result['de'], prior_heads['de'][target], prior_heads['hard_de'][target], self.decoder['margin'], self.decoder['count_scale'])
        return result
