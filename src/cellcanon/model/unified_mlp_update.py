from __future__ import annotations
from pathlib import Path
import copy
import json
import joblib
import numpy as np
from cellcanon.model.data import sha256
from cellcanon.model.condition_feedback_benchmark import array_digest, digest
from cellcanon.model.response_update_session import ResponseUpdateSession
from cellcanon.model.class_response_transfer import ClassResponseTransfer
from cellcanon.model.knowledge_response_transfer import KnowledgeResponseTransfer
from cellcanon.model.response_feature_head import FeatureResponseUpdate
from cellcanon.model.response_count_head import count_features, decode_count_feedback
from cellcanon.model.observation_updaters import ResidualUpdater

class _IdentityContinuous:

    def predict(self, parent, observations, config):
        return np.asarray(parent).copy()

class UnifiedMLPUpdate:

    def __init__(self, parent, heads, partition, budget, support, oof, kernels, selection, residual_models, trees, count_model, binding=None):
        if np.shape(parent)[1:] != (3, 2000) or np.shape(heads['de'])[1:] != (3, 2931):
            raise ValueError('Expected three cell contexts with 2000 continuous-response genes and 2931 classification genes per drug.')
        self.selection = copy.deepcopy(selection)
        self.session = ResponseUpdateSession(parent, heads, partition, _IdentityContinuous(), {}, ClassResponseTransfer(support['de'], support['up']), selection['classes'], budget, residual_models, selection['residual_strengths'])
        self.knowledge = KnowledgeResponseTransfer(support['continuous'], oof['continuous'], kernels[0], kernels[1])
        self.features = FeatureResponseUpdate(support['de'], support['up'], trees, selection['features'], selection['classes']['de'])
        self.count_model = count_model
        self.records = {}
        self.binding = {'models': binding or {}, 'selection': self.selection, 'parent': array_digest(np.asarray(parent)), 'heads': {k: array_digest(np.asarray(heads[k])) for k in ['de', 'up', 'hard_de']}, 'kernels': [array_digest(np.asarray(x)) for x in kernels], 'partition': copy.deepcopy(partition), 'budget': budget}

    @classmethod
    def from_artifacts(cls, root, parent, heads, partition, budget, kernels):
        root = Path(root)
        selected = json.loads((root / 'selection.json').read_text())
        for path, h in selected['files_sha256'].items():
            if sha256(root / path) != h:
                raise ValueError('Update artifact checksum mismatch: ' + path)
        for path, h in selected['source_sha256'].items():
            if sha256(Path(path)) != h:
                raise ValueError('Update source checksum mismatch: ' + path)
        support = dict(np.load(root / 'fit/support.npz'))
        oof = dict(np.load(root / 'oof/predictions.npz'))
        residual = {k: ResidualUpdater.load(root / 'fit' / (k + '.npz')) for k in selected['residual_strengths']}
        trees = {k: joblib.load(root / 'fit' / selected['features'][k]['model']) for k in ['de', 'up']}
        count = joblib.load(root / 'fit' / selected['counts']['model'])
        return cls(parent, heads, partition, budget, support, oof, kernels, selected, residual, trees, count, {'selection_sha256': sha256(root / 'selection.json')})

    @property
    def state_id(self):
        return digest({'sources': self.session.state_id, 'binding': self.binding})

    def legal(self):
        return self.session.legal()

    def observe(self, candidate, continuous, de, up):
        d, u = (np.asarray(de), np.asarray(up))
        if d.shape != u.shape or not np.isin(u[d == 1], [0, 1]).all():
            raise ValueError('Observed DE genes require binary direction.')
        receipt = self.session.observe(candidate, continuous, de, up)
        self.records[candidate] = {'continuous': np.asarray(continuous, float).copy(), 'de': np.asarray(de, float).copy(), 'up': np.nan_to_num(np.asarray(up, float))}
        return {**receipt, 'unified_state_id': self.state_id}

    def predict(self):
        s = self.session
        sel = self.selection
        out = s.predict()
        if not self.records:
            return out
        parent, heads, ids = (s.parent, s.heads, s.partition['identities'])
        records = {}
        for j, r in self.records.items():
            di, c = divmod(j, 3)
            records[j] = r
        own = parent.copy()
        for di, identity in enumerate(ids):
            obs = {j % 3: r for j, r in records.items() if ids[j // 3] == identity}
            if not obs:
                continue
            if sel['continuous']['strength']:
                own[di] = self.knowledge.predict(parent[di], {c: r['continuous'] for c, r in obs.items()}, sel['continuous'], di)
            prior = {k: heads[k][di] for k in ['de', 'up', 'hard_de']}
            current = {k: out[k][di] for k in prior}
            for target in range(3):
                if target in obs:
                    continue
                updated = self.features.apply(parent[di], prior, current, obs, target)
                if sel['counts']['enabled']:
                    counts_obs = obs
                    counts = self.count_model.predict(count_features(counts_obs, target)[None])[0]
                    updated['hard_de'] = decode_count_feedback(updated['de'], prior['hard_de'][target], counts, sel['counts']['blend'], baseline_scale=sel['counts']['baseline_scale'])[0]
                for k, a in updated.items():
                    out[k][di, target] = a
        out['continuous'] = own
        return out
