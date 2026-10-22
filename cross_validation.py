"""Stratified K-fold cross-validation and hyperparameter search harness.

Guarantees leak-free evaluation: all imputation, standardization, and threshold
calibration are fit strictly on training folds and tested on held-out validation folds.
"""

import itertools
import numpy as np

from metrics import (
    accuracy,
    precision_recall_f1,
    predict_labels,
    best_threshold,
    confusion_matrix,
)


def build_k_indices(y, k_fold=5, seed=42):
    """Split sample indices into k_fold stratified folds without dropping remainders.

    Preserves exact class balance (e.g. 8.83% positive) across every fold.

    Args:
        y (np.ndarray): Labels array of shape (N,), values in {-1, 1}.
        k_fold (int): Number of folds (default: 5).
        seed (int): Random seed for reproducible shuffling.

    Returns:
        list: List of k_fold 1D numpy arrays containing sample indices for each fold.
    """
    rng = np.random.default_rng(seed)
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == -1)[0]

    shuffled_pos = rng.permutation(pos_indices)
    shuffled_neg = rng.permutation(neg_indices)

    pos_splits = np.array_split(shuffled_pos, k_fold)
    neg_splits = np.array_split(shuffled_neg, k_fold)

    folds = []
    for k in range(k_fold):
        fold_k = rng.permutation(np.concatenate([pos_splits[k], neg_splits[k]]))
        folds.append(fold_k)

    return folds


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


def cross_validate(
    y,
    x,
    k_indices,
    train_fn,
    predict_fn,
    tune_threshold=False,
    threshold_candidates=None,
    preprocess=True,
):
    """Run stratified K-fold cross-validation for a given model pipeline.

    Args:
        y (np.ndarray): Labels of shape (N,), values in {-1, 1}.
        x (np.ndarray): Features of shape (N, D).
        k_indices (list): Output of build_k_indices.
        train_fn (callable): Function (y_tr, x_tr) -> model.
        predict_fn (callable): Function (model, x_val) -> labels or real-valued scores.
        tune_threshold (bool): If True, calibrates threshold on training folds.
        threshold_candidates (np.ndarray, optional): Threshold candidates to search.
        preprocess (bool): If True, applies leak-free median imputation and standardization.

    Returns:
        dict: Detailed metrics containing per-fold lists, means, and std deviations.
    """
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

        if preprocess:
            x_tr_proc, x_val_proc = preprocess_fold(x_tr, x_val, add_bias=True)
        else:
            x_tr_proc, x_val_proc = x_tr, x_val

        model = train_fn(y_tr, x_tr_proc)

        if tune_threshold:
            # Calibrate decision threshold exclusively on training fold predictions
            scores_tr = predict_fn(model, x_tr_proc)
            tau, _ = best_threshold(y_tr, scores_tr, threshold_candidates)
            fold_thresholds.append(tau)

            scores_val = predict_fn(model, x_val_proc)
            y_pred = predict_labels(scores_val, tau)
        else:
            y_pred = predict_fn(model, x_val_proc)
            # Ensure predictions are in {-1, 1}
            if not np.all(np.isin(y_pred, [-1, 1])):
                y_pred = np.where(y_pred >= 0.0, 1, -1)

        acc = accuracy(y_val, y_pred)
        prec, rec, f1 = precision_recall_f1(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)

        fold_accuracies.append(acc)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)
        fold_balanced_accs.append(cm["balanced_accuracy"])

    results = {
        "accuracy": fold_accuracies,
        "accuracy_mean": float(np.mean(fold_accuracies)),
        "accuracy_std": float(np.std(fold_accuracies)),
        "precision": fold_precisions,
        "precision_mean": float(np.mean(fold_precisions)),
        "precision_std": float(np.std(fold_precisions)),
        "recall": fold_recalls,
        "recall_mean": float(np.mean(fold_recalls)),
        "recall_std": float(np.std(fold_recalls)),
        "f1": fold_f1s,
        "f1_mean": float(np.mean(fold_f1s)),
        "f1_std": float(np.std(fold_f1s)),
        "balanced_accuracy": fold_balanced_accs,
        "balanced_accuracy_mean": float(np.mean(fold_balanced_accs)),
        "balanced_accuracy_std": float(np.std(fold_balanced_accs)),
        "thresholds": fold_thresholds,
    }
    return results


def grid_search(
    y,
    x,
    k_indices,
    param_grid,
    make_train_fn,
    predict_fn,
    tune_threshold=False,
    threshold_candidates=None,
):
    """Exhaustive grid search across hyperparameter combinations using cross-validation.

    Args:
        y (np.ndarray): Labels array.
        x (np.ndarray): Feature matrix.
        k_indices (list): Cross-validation fold indices.
        param_grid (dict): Mapping of parameter names to lists of candidate values.
        make_train_fn (callable): Function (param_dict) -> train_fn.
        predict_fn (callable): Scoring function (model, x) -> predictions.
        tune_threshold (bool): Whether to calibrate threshold on each fold.
        threshold_candidates (np.ndarray, optional): Threshold candidates.

    Returns:
        tuple: (best_params, all_results)
            best_params: dict with parameter combination yielding highest mean F1.
            all_results: list of tuples (param_dict, cv_results).
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, prod)) for prod in itertools.product(*values)]

    best_params = None
    best_f1 = -1.0
    all_results = []

    for params in combinations:
        train_fn = make_train_fn(params)
        cv_res = cross_validate(
            y,
            x,
            k_indices,
            train_fn,
            predict_fn,
            tune_threshold=tune_threshold,
            threshold_candidates=threshold_candidates,
        )
        all_results.append((params, cv_res))

        mean_f1 = cv_res["f1_mean"]
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_params = params

    return best_params, all_results
