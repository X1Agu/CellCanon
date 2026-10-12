from __future__ import annotations
import numpy as np
from cellcanon.model.observation_answer_benchmark import build_case
from cellcanon.model.scoped_answer_world import ScopedAnswerWorld
from cellcanon.model.supported_act import array_sha, digest
from cellcanon.model.supported_act_extensions import task_loss

def target_distance(target, goal):
    if goal.get('mode', 'exact') not in {'exact', 'threshold'} or not np.isfinite(goal['value']):
        raise ValueError('A finite target and supported exact/threshold task mode are required.')
    return float(task_loss(np.array([target, 0.0]), goal))

def build_task_case(*, identity, drug, cell, goal, gene, actual, prediction, current_prediction, source_sha256, prediction_sha256, state):
    mode = goal.get('mode', 'exact')
    target_distance(0.0, goal)
    current = np.asarray(current_prediction, float)
    if current.shape != (3,) or not np.isfinite(current).all():
        raise ValueError('Three finite named current prediction values required.')
    task_key = digest(dict(version='task-source-view-v1', identity=identity, mode=mode, target=goal['value'], gene=gene))[:20]
    case, gold = build_case(identity=identity + ':' + task_key, drug=drug, cell=cell, goal=goal, gene=gene, actual=actual, prediction=prediction, source_sha256=source_sha256, state=state)
    case['interface'] = 'task-source-view-v1'
    case['goal']['mode'] = mode
    case['goal']['target_metric'] = 'absolute_distance' if mode == 'exact' else 'directional_shortfall'
    case['goal']['protection_bound'] = 0.5
    case['goal']['hit_loss_at_most'] = 0.1
    case['records'].append(dict(id='P1', kind='current_act_model_prediction', drug=drug, cell=cell, target_program_mean=float(current[0]), protected_program_mean=float(current[1]), gene=gene, gene_log2_fold_change=float(current[2]), full_prediction_sha256=prediction_sha256, observed=False))
    case['questions'][5]['text'] = 'The measured target-program response has smaller ' + case['goal']['target_metric'] + ' to the stated target than a zero response has.'
    if state == 'observed':
        actual = np.asarray(actual, float)
        actual_truth = {2: float(task_loss(actual[:2], goal)) <= 0.1, 5: target_distance(actual[0], goal) < target_distance(0.0, goal)}
        for index, truth in actual_truth.items():
            gold[index]['label'] = 'supported' if truth else 'contradicted'
    model_facts = [('model_target_positive', 'The CURRENT Act model predicts a target-program mean greater than zero.', current[0] > 0), ('model_goal_hit', 'The CURRENT Act model predicts a hit under the stated goal mode and protection constraint.', float(task_loss(current[:2], goal)) <= 0.1)]
    for kind, text, truth in model_facts:
        q = len(case['questions'])
        qid = digest([case['id'], state, q, prediction_sha256])[:20]
        case['questions'].append(dict(q=q, id=qid, kind=kind, text=text))
        gold.append(dict(id=qid, kind=kind, label='supported' if truth else 'contradicted', evidence_sets=[['P1']], case=case['id'], drug=drug, cell=cell, state=state))
    return (case, gold)

class TaskScopedAnswerWorld(ScopedAnswerWorld):

    def __init__(self, session, *, gene=None):
        super().__init__(session)
        target_distance(0.0, session.goal)
        genes = session.problem.public['genes']
        if gene is not None and (not isinstance(gene, str) or genes.count(gene) != 1):
            raise ValueError('Choose one uniquely named gene in the ordered2000-gene panel.')
        self.gene = gene

    def prepare(self, candidate):
        _, token = super().prepare(candidate)
        s = self.session
        p = s.problem
        di, ci = divmod(candidate, 3)
        gi = p.public['genes'].index(self.gene) if self.gene else int(np.flatnonzero(p.gw[:, s.goal['program_index']])[0])
        fields = [s.goal['program_index'], s.goal['protected_index']]
        current = np.asarray(s.predicted_rna(di), float)[ci]
        active = candidate in s.observations and candidate not in s.withdrawn
        actual = s.observations[candidate]['values'] if active else np.zeros(2000)
        state = 'observed' if active else 'withdrawn' if candidate in s.withdrawn else 'before'
        case, _ = build_task_case(identity=p.public['drugs'][di] + '_' + p.public['cells'][ci] + '_g' + str(s.goal['id']), drug=p.public['drugs'][di], cell=p.public['cells'][ci], goal=s.goal, gene=p.public['genes'][gi], actual=[*(actual @ p.gw[:, fields]).tolist(), float(actual[gi])], prediction=[*(p.prior[di, ci] @ p.gw[:, fields]).tolist(), float(p.prior[di, ci, gi])], current_prediction=[*(current @ p.gw[:, fields]).tolist(), float(current[gi])], source_sha256=array_sha(actual) if active else '0' * 64, prediction_sha256=array_sha(current), state=state)
        token['bindings']['P1'] = dict(evidence_id='current-act-prediction:' + array_sha(current), observation_id=None)
        token['public_case_hash'] = digest(case)
        token.pop('request_id')
        token['request_id'] = digest(token)
        return (case, token)
