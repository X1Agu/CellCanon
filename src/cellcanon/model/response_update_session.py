from __future__ import annotations
import copy
import numpy as np
from cellcanon.model.condition_feedback_benchmark import array_digest, digest
from cellcanon.model.observation_updaters import ResidualObservation
from cellcanon.model.class_response_transfer import label_states, decode_feedback

class ResponseUpdateSession:

    def __init__(self, parent, heads, partition, continuous_model, continuous_config, class_model, class_config, budget, legacy_models=None, legacy_strengths=None):
        self.parent = np.asarray(parent, float).copy()
        self.heads = {k: np.asarray(heads[k]).copy() for k in ['de', 'up', 'hard_de']}
        self.partition = copy.deepcopy(partition)
        self.cm = continuous_model
        self.cc = copy.deepcopy(continuous_config)
        self.hm = class_model
        self.hc = copy.deepcopy(class_config)
        self.budget = budget
        self.observations = {}
        self.receipts = []
        self.ncell = self.parent.shape[1]
        self.legacy_models = legacy_models
        self.legacy_strengths = legacy_strengths
        if set(partition['hidden']) & set(partition['observation']):
            raise ValueError('Protected pool overlaps')
        if self.parent.shape[:2] != self.heads['de'].shape[:2]:
            raise ValueError('Aligned public predictions required')
        ids = partition['identities']
        protected = {(ids[j // self.ncell], j % self.ncell) for j in partition['hidden']}
        if protected & {(ids[j // self.ncell], j % self.ncell) for j in partition['observation']}:
            raise ValueError('Hidden alias in observation pool')

    @property
    def state_id(self):
        return digest({'receipts': self.receipts, 'continuous': self.cc, 'classes': self.hc})

    def legal(self):
        ids = self.partition['identities']
        spent = {(ids[j // self.ncell], j % self.ncell) for j in self.observations}
        keys = {}
        for j in self.partition['observation']:
            key = (ids[j // self.ncell], j % self.ncell)
            if key not in spent:
                keys.setdefault(key, j)
        return list(keys.values())

    def observe(self, candidate, continuous, de, up):
        if type(candidate) is not int or candidate not in self.legal() or len(self.observations) >= self.budget:
            raise ValueError('Hidden/repeated/invalid/overbudget assay')
        c = np.asarray(continuous, float)
        d = np.asarray(de, float)
        u = np.asarray(up, float)
        if c.shape != (self.parent.shape[-1],) or d.shape != (self.heads['de'].shape[-1],) or u.shape != d.shape or (not np.isfinite(c).all()):
            raise ValueError('Complete aligned response required')
        states = label_states(d, u)
        previous = self.state_id
        r = {'candidate': candidate, 'continuous_sha256': array_digest(c), 'de_sha256': array_digest(d), 'state_sha256': array_digest(states), 'cost': 1, 'previous_state_id': previous}
        self.observations[candidate] = {'continuous': c.copy(), 'states': states.copy()}
        self.receipts.append(r)
        return dict(r, state_id=self.state_id)

    def predict(self):
        out = {'continuous': self.parent.copy(), **{k: v.copy() for k, v in self.heads.items()}}
        if not self.observations:
            return out
        ids = self.partition['identities']
        group = {}
        for j in self.observations:
            group.setdefault(ids[j // self.ncell], []).append(j)
        for identity, js in group.items():
            for di in [i for i, x in enumerate(ids) if x == identity]:
                co = {}
                ho = {}
                for j in js:
                    sd, sc = divmod(j, self.ncell)
                    value = self.observations[j]
                    co[sc] = value['continuous']
                    ho[sc] = np.eye(3)[value['states']]
                out['continuous'][di] = self.cm.predict(self.parent[di], co, self.cc)
                for task in ['de', 'up']:
                    cfg = self.hc[task]
                    evidence = self.hm.evidence(self.heads['de'][di], self.heads['up'][di], ho, task, cfg)
                    prob = self.hm.probabilities(self.heads[task][di], evidence, cfg)
                    blend = float(cfg.get('legacy_mix', 0.0))
                    if blend:
                        if self.legacy_models is None:
                            raise ValueError('The selected blend requires fitted classification residual models.')
                        field = 'de_probability' if task == 'de' else 'up_probability'
                        records = []
                        for sc, sp in ho.items():
                            source_de = sp[:, 1] + sp[:, 2]
                            values = source_de if task == 'de' else np.divide(sp[:, 2], source_de, out=np.zeros_like(source_de), where=source_de > 0)
                            ix = np.arange(len(source_de)) if task == 'de' else np.flatnonzero(source_de > 0.5)
                            if len(ix):
                                records.append(ResidualObservation(sc, ix, values[ix] - self.heads[task][di, sc, ix], np.zeros(len(ix))))
                        delta = self.legacy_models[field].infer(records)[0] if records else np.zeros_like(prob)
                        old = np.clip(self.heads[task][di] + self.legacy_strengths[field] * delta, 0, 1)
                        prob = (1 - blend) * prob + blend * old
                    for tc in range(self.ncell):
                        if tc in co:
                            continue
                        if np.any(abs(evidence[tc]) > 1e-12):
                            out[task][di, tc] = prob[tc]
                        if task == 'de' and np.any(abs(evidence[tc]) > 1e-12):
                            out['hard_de'][di, tc] = decode_feedback(out['de'][di, tc], self.heads['de'][di, tc], self.heads['hard_de'][di, tc], cfg['margin'], cfg['count_scale'])
        return out
