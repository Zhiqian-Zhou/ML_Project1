"""Classification metrics. AIcrowd ranks by F1 score, so F1 is our main metric.

All functions take labels in {-1, 1}. Remember that always predicting -1
gives ~91% accuracy but F1 = 0: never judge a model by accuracy alone.
"""

import numpy as np


def accuracy(y_true, y_pred):
    """Fraction of correct predictions.

    Args:
        y_true: numpy array of shape (N,), values in {-1, 1}.
        y_pred: numpy array of shape (N,), values in {-1, 1}.

    Returns:
        float in [0, 1].
    """
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred):
    """Precision, recall and F1 of the positive class (+1).

    F1 = 2 * P * R / (P + R) = 2 TP / (2 TP + FP + FN); 0 when there is no
    true positive.

    Args:
        y_true: numpy array of shape (N,), values in {-1, 1}.
        y_pred: numpy array of shape (N,), values in {-1, 1}.

    Returns:
        precision, recall, f1 (floats).
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == -1))
    fn = np.sum((y_pred == -1) & (y_true == 1))
    if tp == 0:
        return 0.0, 0.0, 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return float(precision), float(recall), float(f1)


def predict_labels(scores, threshold):
    """Turn real-valued scores (e.g. probabilities) into labels in {-1, 1}.

    Args:
        scores: numpy array of shape (N,).
        threshold: float, predict +1 when score >= threshold.

    Returns:
        numpy array of shape (N,), values in {-1, 1}.
    """
    return np.where(scores >= threshold, 1, -1)


def best_threshold(y_true, scores, candidates=None):
    """Find the decision threshold that maximizes F1 on validation data.

    With ~9% positives the best threshold is usually well below 0.5.

    Args:
        y_true: numpy array of shape (N,), values in {-1, 1}.
        scores: numpy array of shape (N,).
        candidates: numpy array of thresholds to try (default: 101 quantiles
            of the scores).

    Returns:
        threshold: float, the best threshold.
        f1: float, the F1 obtained with it.
    """
    if candidates is None:
        candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 101)))
    f1s = [
        precision_recall_f1(y_true, predict_labels(scores, t))[2] for t in candidates
    ]
    best = int(np.argmax(f1s))
    return float(candidates[best]), float(f1s[best])
