from __future__ import annotations
import numpy as np
import pandas as pd
from cellcanon.model.data import CELLS
from cellcanon.math_utils import count_calibration as count_calibration
from cellcanon.math_utils.classification_metrics import _binary_from_confusion, _confusion_binary, _set_values, _multiclass_values

def frame_from_scores(drugs, genes, scores, *, labels=None, split='query'):
    n, c, g = scores.shape
    assert c == len(CELLS) and g == len(genes)
    frame = pd.DataFrame({'drug': np.repeat(drugs, c * g), 'cell_line': np.tile(np.repeat(CELLS, g), n), 'gene': np.tile(genes, n * c), 'gene_index': np.tile(np.arange(g), n * c), 'de_score': scores.reshape(-1)})
    frame['condition_id'] = split + '::' + frame['drug'] + '::' + frame['cell_line']
    if labels is not None:
        frame['is_de'] = labels.reshape(-1)
    return frame

def fit_matched_decoder(validation, probabilities):
    specs = {}
    for scope, sl in [('core', slice(0, 2000)), ('extra', slice(2000, 2931))]:
        frame = frame_from_scores(validation.drugs, validation.genes[sl], probabilities[..., sl], labels=validation.de[..., sl], split='validation')
        specs[scope] = count_calibration._fit_budget_decoder(frame, score='de_score', alpha_grid=[0.1, 1.0, 10.0, 100.0])
    return specs

def apply_matched_decoder(drugs, genes, probabilities, specs):
    result = np.empty(probabilities.shape, dtype=np.int8)
    for scope, sl in [('core', slice(0, 2000)), ('extra', slice(2000, 2931))]:
        frame = frame_from_scores(drugs, genes[sl], probabilities[..., sl])
        _, hard = count_calibration._apply_budget_decoder(frame, specs[scope], score='de_score')
        result[..., sl] = hard.reshape(len(drugs), len(CELLS), -1)
    return result

def hard_metrics(panel, predicted):
    result = {}
    hard_de = predicted['hard_de'].astype(bool)
    hard_up = predicted.get('hard_up', predicted['up'] >= 0.5).astype(bool)
    for scope, sl in [('core', slice(0, 2000)), ('extra', slice(2000, len(panel.genes)))]:
        if not panel.de[..., sl].size:
            continue
        mask = panel.sampled[..., sl]
        de, up = (panel.de[..., sl].astype(bool), panel.up[..., sl])
        hp, hu = (hard_de[..., sl], hard_up[..., sl])
        binary = _binary_from_confusion(_confusion_binary(de[mask], hp[mask]))
        for key in ('accuracy', 'balanced_accuracy', 'precision', 'recall', 'macro_f1', 'mcc'):
            result[f'{scope}_sampled_de_{key}'] = binary[key]
        true_de = mask & de
        direction = _binary_from_confusion(_confusion_binary(up[true_de] == 1, hu[true_de]))
        for key in ('accuracy', 'macro_f1', 'mcc'):
            result[f'{scope}_conditional_dir_{key}'] = direction[key]
        truth_state = np.where(de, np.where(up == 1, 1, 2), 0)
        pred_state = np.where(hp, np.where(hu, 1, 2), 0)
        confusion = np.bincount(3 * truth_state[mask] + pred_state[mask], minlength=9).reshape(3, 3)
        result[f'{scope}_three_state_macro_f1'] = _multiclass_values(confusion)['macro_f1']
        sets = []
        for di in range(len(panel.drugs)):
            for ci in range(len(panel.cells)):
                selected = mask[di, ci]
                sets.append(_set_values(de[di, ci][selected], hp[di, ci][selected]))
        for key in ('precision', 'recall', 'f1', 'jaccard', 'k_abs_error'):
            result[f'{scope}_condition_set_{key}'] = float(np.mean([row[key] for row in sets]))
    return result
