from __future__ import annotations
from dataclasses import asdict, dataclass
import copy
from functools import wraps
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from cellcanon.model.observation_updaters import ResidualObservation

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def array_digest(value):
    a = np.ascontiguousarray(value, dtype='<f8')
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()

def transactional(method):

    @wraps(method)
    def run(self, *args, **kwargs):
        if self._identity() != self.state_id:
            raise ValueError('Numerical state was mutated outside its transition API.')
        names = ('values', 'variance', 'observations', 'retracted', 'events', 'write_rules', 'claims', 'cached_views', 'state_id')
        previous = {name: copy.deepcopy(getattr(self, name)) for name in names}
        try:
            return method(self, *args, **kwargs)
        except Exception:
            for name, value in previous.items():
                setattr(self, name, value)
            raise
    return run

@dataclass(frozen=True)
class Observation:
    observation_id: str
    drug: str
    condition: str
    field: str
    genes: tuple[str, ...]
    values: tuple[float, ...]
    variances: tuple[float, ...]
    units: str
    source_id: str
    source_sha256: str
    cost: float = 1.0

class NumericalWorld:

    def __init__(self, *, drug, conditions, genes, prior, variances, units, model_id, budget, output=None):
        self.drug = str(drug)
        self.conditions = tuple(map(str, conditions))
        self.genes = {k: tuple(map(str, v)) for k, v in genes.items()}
        self.units = dict(units)
        self.model_id = str(model_id)
        self.budget = float(budget)
        if not self.drug or not self.model_id or (not np.isfinite(self.budget)) or (self.budget < 0) or (len(set(self.conditions)) != len(self.conditions)) or (not self.conditions):
            raise ValueError('Unique conditions, model identity and finite budget required.')
        if set(prior) != set(variances) or set(prior) != set(genes) or set(prior) != set(units):
            raise ValueError('Field metadata do not align.')
        self.prior = {k: np.asarray(v, dtype=float).copy() for k, v in prior.items()}
        self.prior_variance = {k: np.asarray(v, dtype=float).copy() for k, v in variances.items()}
        for k, values in self.prior.items():
            shape = (len(self.conditions), len(self.genes[k]))
            if values.shape != shape or self.prior_variance[k].shape != shape or (not np.isfinite(values).all()) or (not np.isfinite(self.prior_variance[k]).all()) or np.any(self.prior_variance[k] < 0) or (len(set(self.genes[k])) != shape[1]):
                raise ValueError('Finite, aligned unique gene fields required.')
            if self.units[k] == 'probability' and np.any((values < 0) | (values > 1)):
                raise ValueError('A probability field must lie in [0,1].')
            values.setflags(write=False)
            self.prior_variance[k].setflags(write=False)
        self.values = {k: v.copy() for k, v in self.prior.items()}
        self.variance = {k: v.copy() for k, v in self.prior_variance.items()}
        self.observations = {}
        self.retracted = set()
        self.events = []
        self.updaters = {}
        self.write_rules = {}
        self.claims = {}
        self.cached_views = {}
        self.output = Path(output) if output is not None else None
        if self.output is not None and (self.output / 'latest.json').exists():
            raise FileExistsError('A world state already exists in this directory. Choose a new directory to create another state.')
        self.state_id = self._identity()
        self._persist()

    @property
    def spent(self):
        return sum((o.cost for o in self.observations.values()))

    def _identity(self):
        return digest(dict(drug=self.drug, conditions=self.conditions, genes=self.genes, units=self.units, model_id=self.model_id, budget=self.budget, prior={k: array_digest(v) for k, v in self.prior.items()}, prior_variance={k: array_digest(v) for k, v in self.prior_variance.items()}, fields={k: array_digest(v) for k, v in self.values.items()}, variance={k: array_digest(v) for k, v in self.variance.items()}, observations={k: asdict(v) for k, v in self.observations.items()}, retracted=sorted(self.retracted), events=self.events, claims=self.claims, write_rules=self._serialized_rules()))

    def _serialized_rules(self):
        return [dict(field=f, target=t, **rule) for (f, t), rule in sorted(self.write_rules.items())]

    def _persist(self):
        if self.output is None:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        destination = self.output / self.state_id
        directory = Path(tempfile.mkdtemp(prefix='.transition-', dir=self.output))
        values = {'value__' + k: v for k, v in self.values.items()}
        values.update({'variance__' + k: v for k, v in self.variance.items()})
        values.update({'prior__' + k: v for k, v in self.prior.items()})
        values.update({'prior_variance__' + k: v for k, v in self.prior_variance.items()})
        meta = dict(state_id=self.state_id, drug=self.drug, model_id=self.model_id, conditions=self.conditions, genes=self.genes, units=self.units, budget=self.budget, spent=self.spent, events=self.events, claims=self.claims, observations={k: asdict(v) for k, v in self.observations.items()}, retracted=sorted(self.retracted), write_rules=self._serialized_rules(), format_version=2)
        try:
            np.savez_compressed(directory / 'state.npz', **values)
            meta['array_sha256'] = hashlib.sha256((directory / 'state.npz').read_bytes()).hexdigest()
            (directory / 'state.json').write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
            if destination.exists():
                existing = type(self).load(self.output, state_id=self.state_id)
                if existing.state_id != self.state_id:
                    raise ValueError('Existing state identity mismatch.')
            else:
                directory.rename(destination)
            pointer = self.output / 'latest.json.tmp'
            pointer.write_text(json.dumps({'state_id': self.state_id}) + '\n')
            pointer.replace(self.output / 'latest.json')
        finally:
            if directory.exists():
                shutil.rmtree(directory)

    @classmethod
    def load(cls, output, *, state_id=None):
        output = Path(output)
        if state_id is None:
            state_id = json.loads((output / 'latest.json').read_text())['state_id']
        if len(state_id) != 64 or any((c not in '0123456789abcdef' for c in state_id)):
            raise ValueError('Invalid state identifier.')
        directory = output / state_id
        meta = json.loads((directory / 'state.json').read_text())
        if meta.get('format_version') != 2:
            raise ValueError('Unsupported snapshot format.')
        if hashlib.sha256((directory / 'state.npz').read_bytes()).hexdigest() != meta['array_sha256']:
            raise ValueError('Snapshot array integrity failure.')
        with np.load(directory / 'state.npz', allow_pickle=False) as arrays:
            world = cls(drug=meta['drug'], conditions=meta['conditions'], genes=meta['genes'], prior={k: arrays['prior__' + k] for k in meta['genes']}, variances={k: arrays['prior_variance__' + k] for k in meta['genes']}, units=meta['units'], model_id=meta['model_id'], budget=meta['budget'])
            world.values = {k: arrays['value__' + k].copy() for k in meta['genes']}
            world.variance = {k: arrays['variance__' + k].copy() for k in meta['genes']}
        for key, item in meta['observations'].items():
            item = {**item, **{k: tuple(item[k]) for k in ['genes', 'values', 'variances']}}
            world.observations[key] = Observation(**item)
        world.retracted = set(meta['retracted'])
        world.events = meta['events']
        world.claims = meta['claims']
        world.write_rules = {(r['field'], r['target']): {k: v for k, v in r.items() if k not in ['field', 'target']} for r in meta['write_rules']}
        world.state_id = world._identity()
        if world.state_id != state_id or meta['state_id'] != state_id or abs(world.spent - meta['spent']) > 1e-10:
            raise ValueError('Snapshot state identity failure.')
        world.output = output
        return world

    @classmethod
    def replay(cls, output, *, state_id=None):
        result = cls.load(output, state_id=state_id)
        current = result
        visited = set()
        while current.events:
            if current.state_id in visited:
                raise ValueError('Cyclic state history.')
            visited.add(current.state_id)
            parent = cls.load(output, state_id=current.events[-1]['parent_state_id'])
            if current.events[:-1] != parent.events or current.model_id != parent.model_id:
                raise ValueError('Broken state event chain.')
            current = parent
        return result

    def _advance(self, event):
        old = self.state_id
        self.events.append({'parent_state_id': old, **event})
        self.cached_views.clear()
        self.state_id = self._identity()
        self._persist()
        return self.state_id

    @transactional
    def observe(self, observation):
        o = observation
        if o.observation_id in self.observations:
            if o != self.observations[o.observation_id]:
                raise ValueError('An observation ID cannot acquire different content.')
            return self.state_id
        if o.drug != self.drug or o.condition not in self.conditions or o.field not in self.prior or (o.units != self.units[o.field]):
            raise ValueError('Observation drug, condition, field or units mismatch.')
        values = np.asarray(o.values)
        variance = np.asarray(o.variances)
        if not o.observation_id or not o.source_id or len(o.source_sha256) != 64 or any((c not in '0123456789abcdef' for c in o.source_sha256.lower())) or (len(o.genes) == 0) or (len(set(o.genes)) != len(o.genes)) or (not set(o.genes) <= set(self.genes[o.field])) or (values.shape != (len(o.genes),)) or (variance.shape != values.shape) or (not np.isfinite(values).all()) or (not np.isfinite(variance).all()) or np.any(variance < 0) or (not np.isfinite(o.cost)) or (o.cost < 0) or (self.spent + o.cost > self.budget + 1e-10):
            raise ValueError('Invalid observation, provenance, measured mask or budget.')
        self.observations[o.observation_id] = o
        if self.units[o.field] == 'probability' and np.any((values < 0) | (values > 1)):
            raise ValueError('Observed probabilities must lie in [0,1].')
        return self._advance({'type': 'observation', 'observation_id': o.observation_id, 'field': o.field, 'condition': o.condition, 'measured_genes': len(o.genes), 'cost': o.cost})

    def residual_observations(self, field, *, ids=None):
        result = []
        for key, o in self.observations.items():
            if key in self.retracted or o.field != field or (ids is not None and key not in ids):
                continue
            c = self.conditions.index(o.condition)
            ix = np.array([self.genes[field].index(g) for g in o.genes])
            result.append(ResidualObservation(c, ix, np.asarray(o.values) - self.prior[field][c, ix], np.asarray(o.variances)))
        return result

    def register_updater(self, field, updater, *, updater_id):
        if field not in self.prior or updater.shape != self.prior[field].shape:
            raise ValueError('Updater shape differs from the numerical field.')
        if not updater_id:
            raise ValueError('A versioned updater identity is required.')
        if any((f == field and r['updater_id'] != updater_id for (f, _), r in self.write_rules.items())):
            raise ValueError('Cannot replace an updater bound to existing writes.')
        self.updaters[field] = (str(updater_id), updater)

    def propose_update(self, *, field, targets, mode, strength):
        if field not in self.updaters:
            raise ValueError('No updater is configured for this field.')
        targets = tuple(targets)
        if not targets or len(set(targets)) != len(targets) or (not set(targets) <= set(self.conditions)):
            raise ValueError('Unique legal target conditions required.')
        updater_id, updater = self.updaters[field]
        obs = self.residual_observations(field)
        if not obs:
            raise ValueError('An observation is required before a state update.')
        delta, var, diagnostic = updater.infer(obs, mode=mode, strength=strength)
        ix = np.array([self.conditions.index(t) for t in targets])
        if not np.isfinite(delta).all() or not np.isfinite(var).all() or np.any(var < 0):
            raise ValueError('Updater proposed an invalid numerical state.')
        values = self.prior[field][ix] + delta[ix]
        if self.units[field] == 'probability':
            values = np.clip(values, 0.0, 1.0)
        proposed = dict(base_state_id=self.state_id, field=field, targets=targets, mode=mode, strength=float(strength), updater_id=updater_id, values=values, variance=var[ix], diagnostic=diagnostic, observation_ids=tuple((k for k, o in self.observations.items() if k not in self.retracted and o.field == field)))
        proposed['proposal_hash'] = self._proposal_hash(proposed)
        return proposed

    @staticmethod
    def _proposal_hash(p):
        return digest({k: array_digest(v) if isinstance(v, np.ndarray) else v for k, v in p.items() if k != 'proposal_hash'})

    def _validate_evidence(self, evidence_ids):
        revoked = {e['evidence_id'] for e in self.events if e['type'] == 'evidence_retraction'}
        for key in evidence_ids:
            if not isinstance(key, str) or not key or key in revoked:
                raise ValueError('Evidence is empty or withdrawn; use a new verified source snapshot.')
            if key.startswith('claim:') and (key[6:] not in self.claims or self.claims[key[6:]]['status'] == 'invalidated'):
                raise ValueError('A dependent claim is missing or invalidated.')

    def _invalidate_dependencies(self, *, evidence_ids=(), claim_ids=(), prediction_fields=(), exclude=()):
        refs = set(evidence_ids)
        fields = set(prediction_fields)
        invalidated = set(claim_ids)
        rolled = []
        refs.update(('claim:' + key for key in invalidated))
        for key in invalidated:
            self.claims[key]['status'] = 'invalidated'
        changed = True
        while changed:
            changed = False
            for key, claim in self.claims.items():
                if key in exclude or key in invalidated:
                    continue
                if refs.intersection(claim['evidence_ids']) or fields.intersection(claim.get('prediction_fields', [])):
                    claim['status'] = 'invalidated'
                    invalidated.add(key)
                    refs.add('claim:' + key)
                    changed = True
            for (field, target), rule in list(self.write_rules.items()):
                if refs.intersection(rule.get('evidence_ids', [])):
                    ix = self.conditions.index(target)
                    self.values[field][ix] = self.prior[field][ix]
                    self.variance[field][ix] = self.prior_variance[field][ix]
                    del self.write_rules[field, target]
                    rolled.append([field, target])
                    fields.add(field)
                    changed = True
        return (sorted(invalidated), rolled)

    @transactional
    def commit(self, proposal, *, reason, evidence_ids=()):
        p = proposal
        if p['base_state_id'] != self.state_id or self._proposal_hash(p) != p['proposal_hash']:
            raise ValueError('Stale or modified update proposal.')
        if not reason:
            raise ValueError('A post-observation write reason is required.')
        self._validate_evidence(evidence_ids)
        field = p['field']
        ix = [self.conditions.index(t) for t in p['targets']]
        before = self.values[field][ix].copy()
        self.values[field][ix] = p['values']
        self.variance[field][ix] = p['variance']
        for target in p['targets']:
            self.write_rules[field, target] = {k: p[k] for k in ['mode', 'strength', 'updater_id']}
            self.write_rules[field, target].update(observation_ids=list(p['observation_ids']), evidence_ids=list(evidence_ids))
        invalidated, rolled_back = self._invalidate_dependencies(prediction_fields=[field])
        event = {k: v for k, v in p.items() if k not in ['values', 'variance', 'base_state_id']}
        event.update(type='numerical_update', reason=str(reason), evidence_ids=list(evidence_ids), invalidated_claims=invalidated, rolled_back_targets=rolled_back, replan_required=bool(rolled_back), changed_values=int(np.count_nonzero(before != self.values[field][ix])), max_absolute_change=float(np.max(np.abs(before - self.values[field][ix]))))
        return self._advance(event)

    @transactional
    def retract(self, observation_id, *, reason):
        if observation_id not in self.observations:
            raise ValueError('Unknown observation.')
        if not reason:
            raise ValueError('A retraction reason is required.')
        if observation_id in self.retracted:
            return self.state_id
        field = self.observations[observation_id].field
        self.retracted.add(observation_id)
        self.values[field] = self.prior[field].copy()
        self.variance[field] = self.prior_variance[field].copy()
        for (f, target), rule in self.write_rules.items():
            if f != field:
                continue
            obs = self.residual_observations(field, ids=rule.get('observation_ids'))
            if not obs:
                continue
            updater_id, updater = self.updaters[f]
            if updater_id != rule['updater_id']:
                raise ValueError('Updater version changed before retraction.')
            delta, var, _ = updater.infer(obs, mode=rule['mode'], strength=rule['strength'])
            ix = self.conditions.index(target)
            values = self.prior[f][ix] + delta[ix]
            self.values[f][ix] = np.clip(values, 0.0, 1.0) if self.units[f] == 'probability' else values
            self.variance[f][ix] = var[ix]
        seeds = [key for key, claim in self.claims.items() if observation_id in claim['observation_ids']]
        invalidated, rolled_back = self._invalidate_dependencies(claim_ids=seeds, prediction_fields=[field])
        return self._advance(dict(type='observation_retraction', observation_id=observation_id, reason=str(reason), invalidated_claims=invalidated, field=field, rolled_back_targets=rolled_back, replan_required=bool(rolled_back)))

    @transactional
    def retract_evidence(self, evidence_id, *, reason):
        if not evidence_id or not reason:
            raise ValueError('Evidence identity and reason required.')
        invalidated, rolled_back = self._invalidate_dependencies(evidence_ids=[evidence_id])
        if not invalidated and (not rolled_back):
            raise ValueError('Evidence has no current dependent state.')
        return self._advance(dict(type='evidence_retraction', evidence_id=evidence_id, reason=reason, invalidated_claims=invalidated, rolled_back_targets=rolled_back, replan_required=bool(rolled_back)))

    @transactional
    def add_claim(self, claim_id, *, statement, observation_ids=(), evidence_ids=(), status='hypothesis', prediction_fields=()):
        if status not in {'hypothesis', 'observation_summary', 'unknown'}:
            raise ValueError('Claim status must be hypothesis, observation_summary or unknown.')
        if any((k not in self.observations or k in self.retracted for k in observation_ids)):
            raise ValueError('Claim depends on an unavailable observation.')
        if not claim_id or not statement or claim_id in self.claims or (not set(prediction_fields) <= set(self.prior)):
            raise ValueError('A unique claim and legal prediction dependencies are required.')
        self._validate_evidence(evidence_ids)
        self.claims[str(claim_id)] = dict(statement=str(statement), observation_ids=list(observation_ids), evidence_ids=list(evidence_ids), status=status, prediction_fields=list(prediction_fields), prediction_state_id=self.state_id)
        return self._advance(dict(type='claim', claim_id=str(claim_id)))

    def predict(self, field):
        return dict(state_id=self.state_id, field=field, conditions=self.conditions, genes=self.genes[field], values=self.values[field].copy(), variance=self.variance[field].copy(), units=self.units[field], model_id=self.model_id)

    @transactional
    def revise_claim(self, claim_id, *, statement, evidence_ids, reason, status='observation_summary'):
        if claim_id not in self.claims or not statement or (not reason) or (not evidence_ids):
            raise ValueError('An existing claim and evidenced correction are required.')
        if status not in {'hypothesis', 'observation_summary', 'unknown'}:
            raise ValueError('Unsupported causal status.')
        self._validate_evidence(evidence_ids)
        previous = copy.deepcopy(self.claims[claim_id])
        invalidated, rolled_back = self._invalidate_dependencies(evidence_ids=['claim:' + claim_id], exclude=[claim_id])
        self.claims[claim_id] = dict(statement=str(statement), evidence_ids=list(evidence_ids), observation_ids=[], status=status, prediction_fields=[], prediction_state_id=self.state_id)
        return self._advance(dict(type='claim_revision', claim_id=claim_id, previous_claim=previous, reason=reason, invalidated_claims=invalidated, rolled_back_targets=rolled_back, replan_required=bool(rolled_back)))

def observations_from_table(path, *, source_id, cost=1.0):
    import pandas as pd
    """Read explicit measured genes only; missing rows never become zero."""
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    required = {'observation_id', 'drug', 'condition', 'field', 'gene', 'value', 'variance', 'units'}
    if not required <= set(frame):
        raise ValueError('Observation table lacks required columns.')
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    result = []
    for (oid, block) in frame.groupby('observation_id', sort=False):
        keys = ['drug', 'condition', 'field', 'units']
        if any((block[k].nunique() != 1 for k in keys)):
            raise ValueError('One observation cannot span scopes or units.')
        first = block.iloc[0]
        result.append(Observation(str(oid), str(first.drug), str(first.condition), str(first.field), tuple(block.gene.astype(str)), tuple(block.value.astype(float)), tuple(block.variance.astype(float)), str(first.units), str(source_id), source_sha, float(cost)))
    return result
