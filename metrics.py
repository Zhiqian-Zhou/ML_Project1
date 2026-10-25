"""Classification metrics for EPFL CS-433 Project 1.

All functions evaluate binary classification with labels in {-1, 1}.
Precision, recall, and F1-score are computed strictly for the positive class (+1).
"""

import numpy as np


def accuracy(y_true, y_pred):
    """Compute fraction of correct classifications.

    Args:
        y_true (np.ndarray): Ground truth labels of shape (N,), values in {-1, 1}.
        y_pred (np.ndarray): Predicted labels of shape (N,), values in {-1, 1}.

    Returns:
        float: Accuracy score in [0.0, 1.0].
    """
    assert y_true.shape == y_pred.shape, "Shape mismatch between y_true and y_pred"
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred):
    """Compute precision, recall, and F1-score of the positive class (+1).

    F1 = 2 * P * R / (P + R) = 2 TP / (2 TP + FP + FN); returns 0.0 when there is no true positive.

    Args:
        y_true (np.ndarray): Ground truth labels of shape (N,), values in {-1, 1}.
        y_pred (np.ndarray): Predicted labels of shape (N,), values in {-1, 1}.

    Returns:
        tuple: (precision, recall, f1) as float values in [0.0, 1.0].
    """
    assert y_true.shape == y_pred.shape, "Shape mismatch between y_true and y_pred"
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == -1) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == -1)))

    if tp == 0.0:
        return 0.0, 0.0, 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    return float(precision), float(recall), float(f1)


def confusion_matrix(y_true, y_pred):
    """Compute confusion matrix counts and balanced accuracy.

    Args:
        y_true (np.ndarray): Ground truth labels of shape (N,), values in {-1, 1}.
        y_pred (np.ndarray): Predicted labels of shape (N,), values in {-1, 1}.

    Returns:
        dict: Counts of TP, FP, TN, FN and balanced accuracy.
    """
    assert y_true.shape == y_pred.shape, "Shape mismatch between y_true and y_pred"
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == -1) & (y_pred == 1)))
    tn = int(np.sum((y_true == -1) & (y_pred == -1)))
    fn = int(np.sum((y_true == 1) & (y_pred == -1)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = 0.5 * (sensitivity + specificity)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "balanced_accuracy": balanced_acc,
    }


def predict_labels(scores, threshold):
    """Turn real-valued scores into binary classification labels {-1, 1}.

    Rule: If score >= threshold -> +1, else -> -1.

    Args:
        scores (np.ndarray): Continuous prediction scores of shape (N,).
        threshold (float): Decision cutoff boundary.

    Returns:
        np.ndarray: Predicted labels in {-1, 1} of shape (N,).
    """
    return np.where(scores >= threshold, 1, -1)


def best_threshold(y_true, scores, candidates=None):
    """Find the decision threshold that maximizes positive class F1-score.

    Uses O(C log N) searchsorted evaluation over threshold candidates.

    Args:
        y_true (np.ndarray): Ground truth labels of shape (N,), values in {-1, 1}.
        scores (np.ndarray): Continuous prediction scores of shape (N,).
        candidates (np.ndarray, optional): Array of threshold values to evaluate.

    Returns:
        tuple: (best_threshold, best_f1)
    """
    if candidates is None:
        candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 101)))

    candidates = np.sort(candidates)
    pos_scores = np.sort(scores[y_true == 1])
    neg_scores = np.sort(scores[y_true == -1])
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0:
        return float(candidates[0]), 0.0

    # Number of positive predictions for each candidate threshold
    tp = n_pos - np.searchsorted(pos_scores, candidates, side="left")
    fp = n_neg - np.searchsorted(neg_scores, candidates, side="left")

    precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recall = tp / n_pos
    f1 = np.where(
        precision + recall > 0.0,
        2.0 * precision * recall / (precision + recall),
        0.0,
    )

    best_idx = int(np.argmax(f1))
    return float(candidates[best_idx]), float(f1[best_idx])
