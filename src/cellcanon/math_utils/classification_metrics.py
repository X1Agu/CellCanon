from __future__ import annotations
import math
import numpy as np

def _confusion_binary(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    y = np.asarray(truth, dtype=bool)
    p = np.asarray(prediction, dtype=bool)
    return np.array([[int((~y & ~p).sum()), int((~y & p).sum())], [int((y & ~p).sum()), int((y & p).sum())]], dtype=np.int64)

def _binary_from_confusion(confusion: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = (int(item) for item in confusion.ravel())
    rows = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    positive_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    negative_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {'rows': rows, 'accuracy': (tp + tn) / rows if rows else float('nan'), 'balanced_accuracy': (recall + specificity) / 2, 'precision': precision, 'recall': recall, 'specificity': specificity, 'positive_f1': positive_f1, 'negative_precision': negative_precision, 'negative_recall': specificity, 'negative_f1': negative_f1, 'macro_f1': (positive_f1 + negative_f1) / 2, 'mcc': (tp * tn - fp * fn) / denominator if denominator else 0.0, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

def _set_values(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(truth, dtype=bool)
    p = np.asarray(prediction, dtype=bool)
    tp = int((y & p).sum())
    true_k = int(y.sum())
    pred_k = int(p.sum())
    union = int((y | p).sum())
    precision = tp / pred_k if pred_k else float(true_k == 0)
    recall = tp / true_k if true_k else float(pred_k == 0)
    f1 = 2 * tp / (true_k + pred_k) if true_k + pred_k else 1.0
    return {'true_k': true_k, 'predicted_k': pred_k, 'true_positive': tp, 'precision': precision, 'recall': recall, 'f1': f1, 'jaccard': tp / union if union else 1.0, 'k_abs_error': abs(pred_k - true_k)}

def _multiclass_values(confusion: np.ndarray) -> dict[str, float | int]:
    total = int(confusion.sum())
    recalls = []
    f1_values = []
    for index in range(3):
        tp = int(confusion[index, index])
        fn = int(confusion[index, :].sum() - tp)
        fp = int(confusion[:, index].sum() - tp)
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        recalls.append(recall)
        f1_values.append(f1)
    return {'rows': total, 'accuracy': float(np.trace(confusion) / total) if total else float('nan'), 'balanced_accuracy': float(np.mean(recalls)), 'macro_f1': float(np.mean(f1_values))}
