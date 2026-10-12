from __future__ import annotations
import numpy as np
from cellcanon.model.compact_act_context import label
from cellcanon.model.supported_act import digest
from cellcanon.model.explanation_benchmark import LABELS

def questions(goal, gene, direction):
    return [('target_positive', 'The measured target-program mean for this assay is greater than zero.'), ('protected_limit', 'The magnitude of the measured protected-program mean is at most0.5.'), ('goal_hit', 'This measured assay is a hit for the stated target-and-protection goal, with loss at most0.1.'), ('target_direction', 'The measured target-program direction is ' + direction + '.'), ('gene_positive', 'The measured log2 fold change of gene ' + gene + ' is greater than zero.'), ('target_improvement', 'The measured target-program mean is closer to the target goal than a zero response would be.'), ('causal_mechanism', 'These records establish a complete causal molecular mechanism for this drug response in this cell.'), ('protein_engagement', 'These RNA records establish direct engagement of the drug with its target protein in this cell.')]

def build_case(*, identity, drug, cell, goal, gene, actual, prediction, source_sha256, state):
    if state not in {'before', 'observed', 'withdrawn'}:
        raise ValueError('Unknown observation source state.')
    a = np.asarray(actual, float)
    predicted = np.asarray(prediction, float)
    if a.shape != (3,) or predicted.shape != (3,) or (not np.isfinite(a).all()) or (not np.isfinite(predicted).all()):
        raise ValueError('Named target/protected/gene values required.')
    direction = ['decrease', 'near_zero', 'increase'][int(digest(identity)[:8], 16) % 3]
    loss = abs(a[0] - goal['value']) + 5 * max(abs(a[1]) - 0.5, 0)
    truth = [a[0] > 0, abs(a[1]) <= 0.5, loss <= 0.1, label(a[0]) == direction, a[2] > 0, abs(a[0] - goal['value']) < abs(goal['value'])]
    present = state == 'observed'
    records = [dict(id='P0', kind='prior_prediction', drug=drug, cell=cell, target_program_mean=float(predicted[0]), protected_program_mean=float(predicted[1]), gene=gene, gene_log2_fold_change=float(predicted[2]), observed=False)]
    if present:
        records.append(dict(id='R0', kind='admitted_actual_RNA_observation', drug=drug, cell=cell, target_program_mean=float(a[0]), protected_program_mean=float(a[1]), gene=gene, gene_log2_fold_change=float(a[2]), full_profile_genes=2000, original_full_RNA_sha256=source_sha256, observed=True))
    claims = []
    gold = []
    for q, (kind, text) in enumerate(questions(goal, gene, direction)):
        qid = digest([identity, state, q])[:20]
        claims.append(dict(q=q, id=qid, kind=kind, text=text))
        answer = ('supported' if truth[q] else 'contradicted') if present and q < 6 else 'insufficient'
        gold.append(dict(id=qid, kind=kind, label=answer, evidence_sets=[['R0']] if answer != 'insufficient' else [], case=identity, drug=drug, cell=cell, state=state))
    public = dict(id=identity, drug=drug, cell=cell, goal=dict(target=goal['target'], protected=goal['protected'], target_goal=goal['value']), units='log2_fold_change', records=records, questions=claims)
    if state == 'withdrawn':
        public['withdrawn_record_ids'] = ['R0']
    return (public, gold)

def validate(case, result):
    rows = result.get('answers', [])
    qs = [r.get('q') for r in rows]
    if any((type(q) is not int for q in qs)) or sorted(qs) != list(range(len(case['questions']))):
        raise ValueError('Each integer question index must appear exactly once.')
    allowed = {r['id'] for r in case['records']}
    bound = []
    for row in rows:
        if row.get('label') not in LABELS:
            raise ValueError('Unknown class label.')
        p = np.asarray(row.get('scores'), float)
        if p.shape != (3,) or not np.isfinite(p).all() or np.any(p < 0) or (p.sum() <= 0):
            raise ValueError('Three finite nonnegative confidence weights with positive total required.')
        refs = row.get('refs')
        if not isinstance(refs, list) or any((r not in allowed for r in refs)):
            raise ValueError('Use only currently available record IDs.')
        if not isinstance(row.get('reason'), str) or not 8 <= len(row['reason']) <= 650:
            raise ValueError('A concise natural-language reason is required.')
        bound.append(dict(id=case['questions'][row['q']]['id'], label=row['label'], probabilities=(p / p.sum()).tolist(), raw_confidence_weights=p.tolist(), evidence=list(dict.fromkeys(refs)), reason=row['reason']))
    return dict(claims=bound)
