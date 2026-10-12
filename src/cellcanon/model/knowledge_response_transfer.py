from __future__ import annotations
import numpy as np

class KnowledgeResponseTransfer:

    def __init__(self, truth, oof, support_kernel, query_kernel):
        self.y = np.asarray(truth, float)
        self.r = self.y - np.asarray(oof, float)
        self.kt = np.asarray(support_kernel, float)
        self.kq = np.asarray(query_kernel, float)
        self.cache = {}
        if self.kt.shape != (len(self.y), len(self.y)) or self.kq.shape[1] != len(self.y):
            raise ValueError('Aligned public knowledge kernels required')

    def fit_map(self, cells, target, config):
        key = (cells, target, config['kernel'], config['drug_mix'], config['ridge'], config['output'])
        if key in self.cache:
            return self.cache[key]
        x = self.y[:, cells].reshape(len(self.y), -1)
        xm = x.mean(0)
        xs = np.maximum(x.std(0), 0.05)
        x = (x - xm) / xs
        dot = x @ x.T / x.shape[1]
        norm = np.maximum(np.diag(dot), 1e-09)
        dist = np.maximum(norm[:, None] + norm[None] - 2 * dot, 0)
        length = max(np.median(dist[np.triu_indices(len(x), 1)]), 1e-06)
        if config['kernel'] != 'linear' or config['output'] != 'response' or config.get('preserve_zero', False):
            raise ValueError('Only the selected response kernel is included.')
        kr = dot
        alpha = config['drug_mix']
        k = kr * (1 - alpha + alpha * self.kt)
        km = k.mean(0)
        center = k.mean()
        kc = k - km[:, None] - km[None] + center
        y = self.y[:, target]
        ym = y.mean(0)
        dual = np.linalg.solve(kc + config['ridge'] * np.eye(len(x)), y - ym)
        self.cache[key] = (x, xm, xs, norm, length, km, center, ym, dual)
        return self.cache[key]

    def predict(self, parent, observations, config, query_index):
        p = np.asarray(parent, float)
        if not observations:
            return p.copy()
        if not 0 <= query_index < len(self.kq):
            raise ValueError('Known public query descriptor required')
        cells = tuple(sorted(observations))
        raw = np.stack([observations[c] for c in cells]).ravel()
        out = p.copy()
        if not np.isfinite(raw).all():
            raise ValueError('Finite admitted source profiles required')
        for target in range(p.shape[0]):
            if target in cells:
                continue
            x, xm, xs, norm, length, km, center, ym, dual = self.fit_map(cells, target, config)
            q = (raw - xm) / xs
            dot = x @ q / x.shape[1]
            kr = dot
            alpha = config['drug_mix']
            k = kr * (1 - alpha + alpha * self.kq[query_index])
            kc = k - km - k.mean() + center
            pred = ym + kc @ dual
            delta = pred - p[target]
            out[target] += config['strength'] * delta
        return out
