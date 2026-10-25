"""Stratified K-fold cross-validation and hyperparameter search harness.

Based on labs/ex04 (build_k_indices, cross_validation, best_degree_selection),
adapted to classification and the F1 score. cross_validate and grid_search
provide the linear baselines of the report; cv_final.py reuses the folds for
the final model.

Leakage rules: the preprocessor (fit_preprocessor) is fitted on the training
folds only and applied unchanged to the held-out fold, and the decision
threshold is chosen on the training-fold scores.
"""

import itertools
import numpy as np

from implementations import reg_logistic_regression, ridge_regression, sigmoid
from metrics import (
    accuracy,
    best_threshold,
    confusion_matrix,
    precision_recall_f1,
    predict_labels,
)
from preprocessing import fit_preprocessor, merge_config, transform

METRICS = ("f1", "accuracy", "precision", "recall", "threshold", "f1_train")


def build_k_indices(y, k_fold=5, seed=42):
    """Split the row indices into k_fold stratified random folds.

    Positives and negatives are shuffled separately and each class is cut
    into k_fold nearly equal parts, so every fold has the same positive ratio
    (8.8% positives) and no row is dropped.

    Args:
        y: numpy array of shape (N,), labels in {-1, 1}.
        k_fold: int, number of folds (default: 5).
        seed: int, random seed (default: 42).

    Returns:
        List of k_fold sorted int arrays; every row is in exactly one fold.
    """
    rng = np.random.default_rng(seed)
    pos = rng.permutation(np.flatnonzero(np.asarray(y) == 1))
    neg = rng.permutation(np.flatnonzero(np.asarray(y) != 1))
    pos_parts = np.array_split(pos, k_fold)
    neg_parts = np.array_split(neg, k_fold)
    return [
        np.sort(np.concatenate([pos_parts[k], neg_parts[k]])).astype(np.int64)
        for k in range(k_fold)
    ]


def fold_split(k_indices, k):
    """Return (training rows, held-out rows) of fold k."""
    te_idx = np.asarray(k_indices[k])
    tr_idx = np.sort(
        np.concatenate([k_indices[i] for i in range(len(k_indices)) if i != k])
    )
    return tr_idx, te_idx


def preprocess_fold(x_tr, x_te=None, add_bias=True):
    """Preprocess fold data in a strictly leak-free manner.

    Imputation medians and standardization moments (mean, std) are calculated
    exclusively from the training fold x_tr and applied to x_te.

    Args:
        x_tr (np.ndarray): Training features of shape (N_tr, D).
        x_te (np.ndarray, optional): Validation/test features of shape (N_te, D).
        add_bias (bool): If True, prepends an intercept column of ones.

    Returns:
        tuple: (tx_tr, tx_te) if x_te is provided, else tx_tr.
    """
    x_tr_clean = x_tr.copy()
    x_te_clean = x_te.copy() if x_te is not None else None

    # 1. Median imputation fit strictly on training fold
    med = np.nanmedian(x_tr_clean, axis=0)
    med = np.nan_to_num(med, nan=0.0)

    nan_mask_tr = np.isnan(x_tr_clean)
    for col_idx in range(x_tr_clean.shape[1]):
        if np.any(nan_mask_tr[:, col_idx]):
            x_tr_clean[nan_mask_tr[:, col_idx], col_idx] = med[col_idx]

    if x_te_clean is not None:
        nan_mask_te = np.isnan(x_te_clean)
        for col_idx in range(x_te_clean.shape[1]):
            if np.any(nan_mask_te[:, col_idx]):
                x_te_clean[nan_mask_te[:, col_idx], col_idx] = med[col_idx]

    # 2. Standardization fit strictly on training fold
    mean_tr = np.mean(x_tr_clean, axis=0)
    std_tr = np.std(x_tr_clean, axis=0)
    std_tr[std_tr == 0.0] = 1.0

    x_tr_clean = (x_tr_clean - mean_tr) / std_tr
    if x_te_clean is not None:
        x_te_clean = (x_te_clean - mean_tr) / std_tr

    # 3. Intercept column
    if add_bias:
        x_tr_clean = np.c_[np.ones(len(x_tr_clean)), x_tr_clean]
        if x_te_clean is not None:
            x_te_clean = np.c_[np.ones(len(x_te_clean)), x_te_clean]

    if x_te is not None:
        return x_tr_clean, x_te_clean
    return x_tr_clean


def train_reg_logistic(y, tx, config):
    """Regularized logistic regression by GD (labels mapped to {0, 1})."""
    cfg = merge_config(config)
    y01 = (np.asarray(y) == 1).astype(np.float64)
    w, _ = reg_logistic_regression(
        y01, tx, cfg["lambda_"], np.zeros(tx.shape[1]), cfg["max_iters"], cfg["gamma"]
    )
    return w


def logistic_scores(w, tx):
    """Predicted probabilities sigmoid(tx @ w)."""
    return sigmoid(tx @ w)


def train_ridge(y, tx, config):
    """Ridge regression on the {-1, 1} labels."""
    w, _ = ridge_regression(np.asarray(y, dtype=np.float64), tx, config["lambda_"])
    return w


def linear_scores(w, tx):
    """Linear scores tx @ w (ridge / least squares)."""
    return tx @ w


def weighted_ridge_regression(y, tx, lambda_, w_pos=1.0):
    """Cost-sensitive sample-weighted Ridge Regression in pure NumPy.

    Solves the exact closed-form normal equations:
        w* = (X^T W X + 2N * lambda * I)^(-1) X^T W y
    where W = diag(w_1, ..., w_N), w_i = w_pos for positive class (+1),
    and 1.0 for negative class (-1).

    Args:
        y (np.ndarray): Labels array of shape (N,), values in {-1, 1}.
        tx (np.ndarray): Feature matrix of shape (N, D).
        lambda_ (float): Regularization parameter >= 0.
        w_pos (float): Relative loss weight on positive samples.

    Returns:
        tuple: (w, loss)
    """
    N, D = tx.shape
    sw = np.where(y == 1, w_pos, 1.0)
    W_tx = tx * sw[:, np.newaxis]
    A = tx.T @ W_tx + 2.0 * N * lambda_ * np.eye(D)
    b = tx.T @ (sw * y)
    w = np.linalg.solve(A, b)

    e = y - tx @ w
    loss = (1.0 / (2.0 * N)) * np.sum(sw * (e**2)) + lambda_ * np.sum(w**2)
    return w, loss


def cross_validate_weighted_ridge(
    y,
    x,
    k_indices,
    feature_names=None,
    lambda_=1e-4,
    w_pos=2.5,
    threshold_candidates=None,
):
    """Run stratified 5-fold cross-validation for Cost-Sensitive Weighted Ridge Regression.

    Args:
        y (np.ndarray): Labels array of shape (N,), values in {-1, 1}.
        x (np.ndarray): Feature matrix of shape (N, D).
        k_indices (list): Stratified fold splits.
        feature_names (list, optional): List of feature names.
        lambda_ (float): Regularization strength.
        w_pos (float): Positive class weight.
        threshold_candidates (np.ndarray, optional): Threshold candidates.

    Returns:
        dict: Detailed cross-validation metrics across all folds.
    """
    if threshold_candidates is None:
        threshold_candidates = np.linspace(-1.0, 1.0, 101)

    k_fold = len(k_indices)
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_balanced_accs = []
    fold_thresholds = []

    for k in range(k_fold):
        val_idx = k_indices[k]
        tr_idx = np.concatenate([k_indices[i] for i in range(k_fold) if i != k])

        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        tx_tr, tx_val = preprocess_fold(x_tr, x_val, add_bias=True)

        w, _ = weighted_ridge_regression(
            y_tr.astype(float), tx_tr, lambda_=lambda_, w_pos=w_pos
        )

        scores_tr = tx_tr @ w
        tau, _ = best_threshold(y_tr, scores_tr, threshold_candidates)
        fold_thresholds.append(tau)

        scores_val = tx_val @ w
        y_pred = predict_labels(scores_val, tau)

        acc = accuracy(y_val, y_pred)
        prec, rec, f1 = precision_recall_f1(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)

        fold_accuracies.append(acc)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)
        fold_balanced_accs.append(cm["balanced_accuracy"])

    return {
        "accuracy_mean": float(np.mean(fold_accuracies)),
        "accuracy_std": float(np.std(fold_accuracies)),
        "precision_mean": float(np.mean(fold_precisions)),
        "precision_std": float(np.std(fold_precisions)),
        "recall_mean": float(np.mean(fold_recalls)),
        "recall_std": float(np.std(fold_recalls)),
        "f1_mean": float(np.mean(fold_f1s)),
        "f1_std": float(np.std(fold_f1s)),
        "balanced_accuracy_mean": float(np.mean(fold_balanced_accs)),
        "balanced_accuracy_std": float(np.std(fold_balanced_accs)),
        "threshold_mean": float(np.mean(fold_thresholds)),
        "thresholds": fold_thresholds,
    }


def prepare_folds(y, xc, names, k_indices, config):
    """Fit the preprocessor on each training fold and transform both parts.

    Args:
        y: numpy array of shape (N,), labels in {-1, 1}.
        xc: numpy array of shape (N, D), clean_data output.
        names: list of D column names of xc.
        k_indices: output of build_k_indices.
        config: dict, preprocessing options (merged with DEFAULT_CONFIG).

    Returns:
        List of k tuples (tx_tr, y_tr, tx_te, y_te, state).
    """
    folds = []
    for k in range(len(k_indices)):
        tr_idx, te_idx = fold_split(k_indices, k)
        state, tx_tr = fit_preprocessor(
            xc[tr_idx], names, config, y=y[tr_idx], return_train=True
        )
        tx_te = transform(xc[te_idx], state)
        folds.append((tx_tr, y[tr_idx], tx_te, y[te_idx], state))
    return folds


def summarize(res):
    """Add "<m>_mean" and "<m>_std" to a dict of per-fold metric lists."""
    out = {}
    for m in METRICS:
        out[m] = np.array(res[m], dtype=np.float64)
        out[m + "_mean"] = float(np.mean(out[m]))
        out[m + "_std"] = float(np.std(out[m]))
    return out


def cross_validate(
    y,
    xc,
    names=None,
    k_indices=None,
    config=None,
    train_fn=train_reg_logistic,
    score_fn=logistic_scores,
    folds=None,
    verbose=False,
):
    """K-fold cross-validation of one preprocessing + model setting.

    For each fold: preprocess with statistics of the training folds only,
    train, choose the threshold that maximizes F1 on the training-fold
    scores, then score the held-out fold.

    Args:
        y: numpy array of shape (N,), labels in {-1, 1}.
        xc: numpy array of shape (N, D), clean_data output (unused if folds).
        names: list of D column names of xc (unused if folds).
        k_indices: output of build_k_indices.
        config: dict, preprocessing and model options.
        train_fn: callable (y_tr, tx_tr, config) -> w.
        score_fn: callable (w, tx) -> scores of shape (N,).
        folds: optional output of prepare_folds (skips the preprocessing).
        verbose: bool, print one line per fold.

    Returns:
        dict with per-fold arrays "f1", "accuracy", "precision", "recall",
        "threshold", "f1_train", their "_mean" and "_std", and "n_features".
    """
    if k_indices is None:
        k_indices = build_k_indices(y, k_fold=5, seed=42)
    cfg = merge_config(config)
    if folds is None:
        folds = prepare_folds(y, xc, names, k_indices, cfg)
    res = dict((m, []) for m in METRICS)
    for k, (tx_tr, y_tr, tx_te, y_te, _) in enumerate(folds):
        w = train_fn(y_tr, tx_tr, cfg)
        thr, f1_tr = best_threshold(y_tr, score_fn(w, tx_tr))
        pred = predict_labels(score_fn(w, tx_te), thr)
        prec, rec, f1 = precision_recall_f1(y_te, pred)
        for m, v in zip(METRICS, (f1, accuracy(y_te, pred), prec, rec, thr, f1_tr)):
            res[m].append(v)
        if verbose:
            print("fold %d: F1 %.4f (train %.4f) thr %.4f" % (k, f1, f1_tr, thr))
    out = summarize(res)
    out["n_features"] = int(folds[0][0].shape[1])
    return out


def grid_search(
    y,
    xc,
    names,
    k_indices,
    config,
    param_grid,
    train_fn=train_reg_logistic,
    score_fn=logistic_scores,
    verbose=False,
):
    """Cross-validate every combination of MODEL parameters in param_grid.

    The folds are preprocessed once with config; compare preprocessing
    options with separate cross_validate calls.

    Args:
        y, xc, names, k_indices, config: as in cross_validate.
        param_grid: dict name -> list of values, e.g. {"lambda_": [...]}.
        train_fn, score_fn: as in cross_validate.
        verbose: bool, print one line per combination.

    Returns:
        best_params: dict, the combination with the highest mean F1.
        results: list of (params, cv_result) in grid order.
    """
    combos = [{}]
    for key in param_grid:
        combos = [dict(c, **{key: v}) for c in combos for v in param_grid[key]]
    folds = prepare_folds(y, xc, names, k_indices, config)
    results = []
    for params in combos:
        cfg = merge_config(dict(config or {}, **params))
        res = cross_validate(
            y, xc, names, k_indices, cfg, train_fn, score_fn, folds=folds
        )
        results.append((params, res))
        if verbose:
            print("%s: F1 %.4f +/- %.4f" % (params, res["f1_mean"], res["f1_std"]))
    best = max(range(len(results)), key=lambda i: results[i][1]["f1_mean"])
    return results[best][0], results
