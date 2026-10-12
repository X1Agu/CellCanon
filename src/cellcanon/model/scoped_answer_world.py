from __future__ import annotations
import copy
import json
import numpy as np
from cellcanon.model.observation_answer_benchmark import build_case
from cellcanon.model.supported_act import array_sha, digest

class ScopedAnswerWorld:

    def __init__(self, session):
        self.session = session

    def prepare(self, candidate):
        s = self.session
        p = s.problem
        if type(candidate) is not int or candidate not in p.legal:
            raise ValueError('A source-qualified condition is required.')
        di, ci = divmod(candidate, 3)
        world = s._world(di)
        gene_index = int(np.flatnonzero(p.gw[:, s.goal['program_index']])[0])
        fields = [s.goal['program_index'], s.goal['protected_index']]
        predicted = p.prior[di, ci] @ p.gw[:, fields]
        active = candidate in s.observations and candidate not in s.withdrawn
        state = 'observed' if active else 'withdrawn' if candidate in s.withdrawn else 'before'
        values = s.observations[candidate]['values'] if active else np.zeros(2000)
        actual = values @ p.gw[:, fields]
        identity = p.public['drugs'][di] + '_' + p.public['cells'][ci] + '_g' + str(s.goal['id'])
        case, _ = build_case(identity=identity, drug=p.public['drugs'][di], cell=p.public['cells'][ci], goal=s.goal, gene=p.public['genes'][gene_index], actual=[*actual.tolist(), float(values[gene_index])], prediction=[*predicted.tolist(), float(p.prior[di, ci, gene_index])], source_sha256=array_sha(values) if active else '0' * 64, state=state)
        bindings = {'P0': dict(evidence_id='frozen-parent-model:' + s.input_fingerprint, observation_id=None)}
        if active:
            oid = 'assay:' + str(candidate)
            observation = world.observations[oid]
            if oid in world.retracted or array_sha(np.asarray(observation.values)) != array_sha(values):
                raise ValueError('Session and world RNA source disagree.')
            bindings['R0'] = dict(evidence_id=observation.source_id, observation_id=oid)
        token = dict(candidate=candidate, session_state_id=s.state_id, world_state_id=world.state_id, public_case_hash=digest(case), bindings=bindings)
        token['request_id'] = digest(token)
        return (case, token)

    def apply(self, case, token, answer, *, producer):
        if not producer:
            raise ValueError('Explicit model/producer attribution is required.')
        s = self.session
        j = token['candidate']
        world = s._world(j // 3)
        expected, actual_token = self.prepare(j)
        if token != actual_token or digest(case) != token['public_case_hash']:
            raise ValueError('Stale or modified explanation request.')
        rows = answer.get('claims', [])
        ids = [r.get('id') for r in rows]
        if len(ids) != len(set(ids)) or set(ids) != {c['id'] for c in case['questions']}:
            raise ValueError('Return every current question ID exactly once.')
        bindings = token['bindings']
        for row in rows:
            if row.get('label') not in {'supported', 'contradicted', 'insufficient'} or not isinstance(row.get('reason'), str):
                raise ValueError('A labeled model judgment and natural-language reason are required.')
            if not isinstance(row.get('evidence'), list) or any((e not in bindings for e in row['evidence'])):
                raise ValueError('A citation is not currently available.')
        revised = copy.deepcopy(world)
        revised.output = None
        recorded = []
        all_observations = sorted({b['observation_id'] for b in bindings.values() if b['observation_id']})
        all_evidence = sorted({b['evidence_id'] for b in bindings.values()})
        for row in rows:
            cid = 'scoped-answer:' + digest([token['request_id'], row])[:24]
            statement = dict(type='llm_source_scoped_answer', decision=copy.deepcopy(row), producer=producer, condition=case['cell'], request_id=token['request_id'], public_case_hash=token['public_case_hash'], causal_effect_verified=False, source_entailment_not_self_certified=True)
            revised.add_claim(cid, statement=json.dumps(statement, sort_keys=True), observation_ids=all_observations, evidence_ids=all_evidence, status='unknown' if row['label'] == 'insufficient' else 'observation_summary' if all_observations else 'hypothesis', prediction_fields=['continuous_core'])
            recorded.append(cid)
        s.worlds[j // 3] = revised
        s.events.append(dict(type='scoped_language_answers', candidate=j, request_id=token['request_id'], claims=recorded, producer=producer, previous_world_state=world.state_id, world_state=revised.state_id))
        return dict(claim_ids=recorded, world_state_id=revised.state_id, session_state_id=s.state_id, numerical_values_changed=False, biological_truth_certified=False)
