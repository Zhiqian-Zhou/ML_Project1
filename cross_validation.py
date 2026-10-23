"""K-fold cross-validation and hyperparameter search.

Based on labs/ex04 (build_k_indices, cross_validation, best_degree_selection),
adapted to classification and the F1 score. Every number in the report
(mean +/- std over folds) and every hyperparameter choice comes from here,
never from the AIcrowd leaderboard.

Leakage rules: the preprocessor (fit_preprocessor) is fitted on the training
folds only and applied unchanged to the held-out fold, and the decision
threshold is chosen on the training-fold scores.
"""

import numpy as np

from implementations import reg_logistic_regression, ridge_regression, sigmoid
from metrics import accuracy, best_threshold, precision_recall_f1, predict_labels
from preprocessing import fit_preprocessor, merge_config, transform

METRICS = ("f1", "accuracy", "precision", "recall", "threshold", "f1_train")


def build_k_indices(y, k_fold, seed):
    """Split the row indices into k_fold stratified random folds.

    Positives and negatives are shuffled separately and each class is cut
    into k_fold nearly equal parts, so every fold has the same positive ratio
    (8.8% positives) and no row is dropped.

    Args:
        y: numpy array of shape (N,), labels in {-1, 1}.
        k_fold: int, number of folds.
        seed: int, random seed.

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
    names,
    k_indices,
    config,
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
