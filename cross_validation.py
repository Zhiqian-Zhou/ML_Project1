"""K-fold cross-validation and hyperparameter search.

STATUS: template. Based on labs/ex04/solution/ex04.ipynb (build_k_indices,
cross_validation, best_degree_selection), adapted to classification + F1.

Every number reported in the report (mean +/- std over folds) and every
hyperparameter choice (lambda_, gamma, max_iters, degree, threshold) must
come from here, never from the AIcrowd leaderboard.
"""

import numpy as np


def build_k_indices(y, k_fold, seed):
    """Split the sample indices into k_fold random folds.

    Prefer stratified folds (same positive ratio in every fold) because the
    classes are imbalanced. Do not drop the remainder samples.

    Args:
        y: numpy array of shape (N,), labels in {-1, 1}.
        k_fold: int, number of folds.
        seed: int, random seed for reproducibility.

    Returns:
        List of k_fold numpy arrays of indices.
    """
    raise NotImplementedError("Pending: build_k_indices")


def cross_validate(y, x, k_indices, train_fn, predict_fn):
    """Run k-fold cross-validation for one model / hyperparameter setting.

    For each fold k: preprocess with statistics of the training folds only,
    train with train_fn, score the held-out fold with predict_fn, and compute
    F1 and accuracy (tune the threshold on the training folds, not on the
    held-out fold).

    Args:
        y: numpy array of shape (N,), labels in {-1, 1}.
        x: numpy array of shape (N, D), features.
        k_indices: output of build_k_indices.
        train_fn: callable (y_tr, x_tr) -> model.
        predict_fn: callable (model, x_te) -> labels in {-1, 1}.

    Returns:
        dict with the per-fold and mean/std values of "f1" and "accuracy".
    """
    raise NotImplementedError("Pending: cross_validate")


def grid_search(y, x, k_indices, param_grid, make_train_fn, predict_fn):
    """Evaluate every combination of param_grid with cross_validate.

    Args:
        y: numpy array of shape (N,), labels in {-1, 1}.
        x: numpy array of shape (N, D), features.
        k_indices: output of build_k_indices.
        param_grid: dict name -> list of values, e.g. {"lambda_": [...]}.
        make_train_fn: callable (params) -> train_fn.
        predict_fn: callable (model, x_te) -> labels in {-1, 1}.

    Returns:
        best_params: dict.
        results: list of (params, cv_result) for plotting validation curves.
    """
    raise NotImplementedError("Pending: grid_search")
