"""Classification metrics. AIcrowd ranks by F1 score, so F1 is our main metric.

STATUS: template.

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
    raise NotImplementedError("Pending: accuracy")


def precision_recall_f1(y_true, y_pred):
    """Precision, recall and F1 of the positive class (+1).

    F1 = 2 * P * R / (P + R); return 0 when the denominator is 0.

    Args:
        y_true: numpy array of shape (N,), values in {-1, 1}.
        y_pred: numpy array of shape (N,), values in {-1, 1}.

    Returns:
        precision, recall, f1 (floats).
    """
    raise NotImplementedError("Pending: precision_recall_f1")


def predict_labels(scores, threshold):
    """Turn real-valued scores (e.g. probabilities) into labels in {-1, 1}.

    Args:
        scores: numpy array of shape (N,).
        threshold: float, predict +1 when score >= threshold.

    Returns:
        numpy array of shape (N,), values in {-1, 1}.
    """
    raise NotImplementedError("Pending: predict_labels")


def best_threshold(y_true, scores, candidates=None):
    """Find the decision threshold that maximizes F1 on validation data.

    With ~9% positives the best threshold is usually well below 0.5.

    Args:
        y_true: numpy array of shape (N,), values in {-1, 1}.
        scores: numpy array of shape (N,).
        candidates: numpy array of thresholds to try (default: a grid).

    Returns:
        threshold: float, the best threshold.
        f1: float, the F1 obtained with it.
    """
    raise NotImplementedError("Pending: best_threshold")
