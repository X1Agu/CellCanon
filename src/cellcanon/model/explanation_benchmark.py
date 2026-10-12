from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
LABELS = ('supported', 'contradicted', 'insufficient')

def ratio(a, b):
    return float(a / b) if b else None

def probability(value):
    p = np.asarray(value, float)
    if p.shape != (3,) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError('Three finite nonnegative class probabilities required.')
    if not np.isclose(p.sum(), 1.0, atol=0.02):
        raise ValueError('Class probabilities must sum to one.')
    return p / p.sum()

def claim_metrics(gold, predictions):
    if not gold or len({r['id'] for r in gold}) != len(gold):
        raise ValueError('A nonempty, unique gold question universe is required.')
    if any((r['label'] not in LABELS for r in gold)):
        raise ValueError('Unknown gold label.')
    ids = [r.get('id') for r in predictions]
    if len(set(ids)) != len(ids) or set(ids) - {r['id'] for r in gold}:
        raise ValueError('Duplicate or unexpected predicted question IDs.')
    lookup = {r['id']: r for r in predictions}
    truth, labels, exact, answered, changes, corrected, old_wrong = ([], [], [], [], [], [], [])
    tp = emitted = relevant = complete = 0
    probability_rows = []
    rows = []
    for g in gold:
        p = lookup.get(g['id'], {})
        label = p.get('label', 'invalid')
        if label not in LABELS:
            label = 'invalid'
        refs = p.get('evidence', [])
        if not isinstance(refs, list) or any((not isinstance(e, str) for e in refs)):
            refs = []
            label = 'invalid'
        refs = set(refs)
        groups = [set(x) for x in g.get('evidence_sets', []) if x]
        valid = set().union(*groups) if groups else set()
        decision_ok = label == g['label']
        sufficient = any((group <= refs for group in groups)) if groups else not refs
        end_to_end = decision_ok and sufficient and (refs <= valid)
        best = max(groups, key=lambda s: (len(s & refs) / len(s), -len(s))) if groups else set()
        good_refs = len(refs & valid) if decision_ok else 0
        tp += good_refs
        emitted += len(refs)
        relevant += len(best)
        complete += int(decision_ok and sufficient)
        previous = g.get('previous_label', 'insufficient')
        changed = label in LABELS and label != previous
        truth.append(g['label'])
        labels.append(label)
        exact.append(end_to_end)
        answered.append(label in LABELS[:2])
        changes.append(changed)
        corrected.append(changed and end_to_end)
        old_wrong.append(previous != g['label'])
        try:
            prob = probability(p['probabilities'])
            yi = LABELS.index(g['label'])
            probability_rows.append((yi, prob))
        except (ValueError, KeyError, TypeError):
            pass
        rows.append(dict(id=g['id'], label_correct=decision_ok, exact=end_to_end, evidence_complete=sufficient, answered=answered[-1], changed=changed))
    recalls = []
    for g in gold:
        if not g.get('evidence_sets'):
            continue
        p = lookup.get(g['id'], {})
        refs = p.get('evidence', [])
        refs = set(refs) if isinstance(refs, list) and all((isinstance(x, str) for x in refs)) else set()
        recalls.append(max((len(set(s) & refs) / len(set(s)) for s in g['evidence_sets'] if s)) if p.get('label') == g['label'] else 0.0)
    ep = ratio(tp, emitted)
    er = float(np.mean(recalls)) if recalls else None
    corrections = sum(corrected)
    answer_correct = sum((e and a for e, a in zip(exact, answered)))
    answerable = [g['label'] != 'insufficient' for g in gold]
    result = dict(questions=len(gold), returned=len(predictions), invalid_or_missing=sum((x == 'invalid' for x in labels)), accuracy=float(accuracy_score(truth, labels)), macro_f1=float(f1_score(truth, labels, labels=list(LABELS), average='macro', zero_division=0)), macro_precision=float(precision_score(truth, labels, labels=list(LABELS), average='macro', zero_division=0)), macro_recall=float(recall_score(truth, labels, labels=list(LABELS), average='macro', zero_division=0)), exact_accuracy=float(np.mean(exact)), label_and_complete_evidence_accuracy=complete / len(gold), evidence_precision=ep, proof_recall_macro=er, evidence_f1=None if ep is None or er is None else ratio(2 * ep * er, ep + er) if ep + er else 0.0, answer_coverage=float(np.mean(answered)), answer_exact_precision=ratio(answer_correct, sum(answered)), answer_exact_risk=None if not sum(answered) else 1 - answer_correct / sum(answered), answerable_questions=sum(answerable), answerable_resolution=ratio(sum((e and a for e, a in zip(exact, answerable))), sum(answerable)), prior_wrong=sum(old_wrong), corrections=sum(changes), strict_correction_precision=ratio(corrections, sum(changes)), strict_correction_recall=ratio(corrections, sum(old_wrong)), strict_correction_f1=ratio(2 * corrections, sum(changes) + sum(old_wrong)), overcorrection_rate=ratio(sum((not w and l != g['label'] for w, l, g in zip(old_wrong, labels, gold))), sum((not w for w in old_wrong))), probability_questions=len(probability_rows), causal_accuracy=None)
    if probability_rows:
        y = np.array([x[0] for x in probability_rows])
        ps = np.stack([x[1] for x in probability_rows])
        result.update(brier=float(np.mean(np.sum((ps - np.eye(3)[y]) ** 2, axis=1))), log_loss=float(-np.mean(np.log(np.clip(ps[np.arange(len(y)), y], 1e-06, 1.0)))))
    else:
        result.update(brier=None, log_loss=None)
    return (result, rows)

def route_metrics(cases, rankings, k=3):
    if not cases:
        return dict(cases=0, strict_at_k=None, any_valid_at_k=None, related_at_k=None)
    out = []
    for c in cases:
        ranking = rankings.get(c['id'], [])
        if len(set(ranking)) != len(ranking) or set(ranking) - set(c['candidates']):
            ranking = []
        prefix = ranking[:k]
        valid = set(c['valid'])
        related = valid | set(c.get('related', []))
        out.append(dict(id=c['id'], strict=int(c['strict'] in prefix), any_valid=int(bool(valid & set(prefix))), related=int(bool(related & set(prefix))), precision=ratio(len(valid & set(prefix)), len(prefix)), recall=ratio(len(valid & set(prefix)), len(valid)), reciprocal_rank=next((1.0 / (i + 1) for i, v in enumerate(ranking) if v in valid), 0.0), invalid_or_missing=not bool(ranking)))
    return dict(cases=len(out), k=k, strict_at_k=float(np.mean([r['strict'] for r in out])), any_valid_at_k=float(np.mean([r['any_valid'] for r in out])), related_at_k=float(np.mean([r['related'] for r in out])), precision_at_k_including_missing=float(np.mean([r['precision'] or 0.0 for r in out])), recall_at_k=float(np.mean([r['recall'] or 0.0 for r in out])), mrr=float(np.mean([r['reciprocal_rank'] for r in out])), missing=sum((r['invalid_or_missing'] for r in out)), scope='Route recovery against the supplied source reference.', rows=out)
