"""K-fold cross-validation and hyperparameter search.

Based on labs/ex04/solution/ex04.ipynb (build_k_indices, cross_validation,
best_degree_selection), adapted to classification + F1.

Every number reported in the report (mean +/- std over folds) and every
hyperparameter choice (lambda_, gamma, max_iters, degree, threshold) must
come from here, never from the AIcrowd leaderboard.

Leakage rules enforced here:
    * the preprocessor (fit_preprocessor) is fitted on the training folds
      only and applied unchanged to the held-out fold;
    * the decision threshold is chosen on the training-fold scores;
    * undersampling (optional) only removes rows of the training fold used
      to fit the model weights.
"""

import numpy as np

from implementations import reg_logistic_regression, ridge_regression, sigmoid
from metrics import accuracy, best_threshold, precision_recall_f1, predict_labels
from models import undersample_indices, weighted_reg_logistic_regression
from preprocessing import DEFAULT_CONFIG, fit_preprocessor, transform

# Config keys that change the preprocessed matrices (sections 4.5 and 5b of
# the design). Changing only other keys (model keys) reuses the cached folds.
PREP_KEYS = (
    "max_nan_ratio",
    "onehot_state",
    "include_leaky",
    "drop_features",
    "nan_indicators",
    "nan_indicator_min_share",
    "nan_indicator_min_effect",
    "indicator_max_corr",
    "rare_level_min_share",
    "log_counts",
    "clip_quantile",
    "poly_degree",
    "interactions",
)

METRICS = ("f1", "accuracy", "precision", "recall", "threshold", "f1_train")


def merge_config(config=None):
    """Merge a partial config into a copy of DEFAULT_CONFIG.

    Args:
        config: dict or None, keys overriding DEFAULT_CONFIG.

    Returns:
        dict, the merged config (DEFAULT_CONFIG is not modified).
    """
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config or {})
    return cfg


def build_k_indices(y, k_fold, seed):
    """Split the sample indices into k_fold stratified random folds.

    Positives and negatives are shuffled separately (np.random.default_rng)
    and each class is cut into k_fold nearly equal parts with
    np.array_split, so every fold has the same positive ratio (within one
    row per class) and no remainder sample is dropped.

    Args:
        y: numpy array of shape (N,), labels in {-1, 1}.
        k_fold: int, number of folds.
        seed: int, random seed for reproducibility.

    Returns:
        List of k_fold sorted int numpy arrays of indices. Every row is in
        exactly one fold.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    pos = rng.permutation(np.flatnonzero(y == 1))
    neg = rng.permutation(np.flatnonzero(y != 1))
    pos_parts = np.array_split(pos, k_fold)
    neg_parts = np.array_split(neg, k_fold)
    return [
        np.sort(np.concatenate([pos_parts[k], neg_parts[k]])).astype(np.int64)
        for k in range(k_fold)
    ]


def fold_split(k_indices, k):
    """Training and held-out indices of fold k.

    Args:
        k_indices: output of build_k_indices.
        k: int, index of the held-out fold.

    Returns:
        tr_idx: sorted int numpy array, the rows of all the other folds.
        te_idx: int numpy array, the rows of fold k.
    """
    te_idx = np.asarray(k_indices[k])
    tr_idx = np.sort(
        np.concatenate([k_indices[i] for i in range(len(k_indices)) if i != k])
    )
    return tr_idx, te_idx


def train_reg_logistic(y, tx, config):
    """Train regularized logistic regression (optionally class-weighted).

    Args:
        y: numpy array of shape (N,), labels in {-1, 1}.
        tx: numpy array of shape (N, D), preprocessed features with bias.
        config: dict with lambda_, gamma, max_iters and class_weight
            (None -> implementations.reg_logistic_regression, else
            models.weighted_reg_logistic_regression with that positive weight).

    Returns:
        w: numpy array of shape (D,).
    """
    cfg = merge_config(config)
    y01 = (np.asarray(y) == 1).astype(np.float64)
    initial_w = np.zeros(tx.shape[1])
    if cfg["class_weight"] is None:
        w, _ = reg_logistic_regression(
            y01, tx, cfg["lambda_"], initial_w, cfg["max_iters"], cfg["gamma"]
        )
    else:
        w, _ = weighted_reg_logistic_regression(
            y01,
            tx,
            cfg["lambda_"],
            initial_w,
            cfg["max_iters"],
            cfg["gamma"],
            cfg["class_weight"],
        )
    return w


def logistic_scores(w, tx):
    """Predicted probabilities of the positive class.

    Args:
        w: numpy array of shape (D,).
        tx: numpy array of shape (N, D).

    Returns:
        numpy array of shape (N,), sigmoid(tx @ w).
    """
    return sigmoid(tx @ w)


def train_ridge(y, tx, config):
    """Train ridge regression on the {-1, 1} labels.

    Args:
        y: numpy array of shape (N,), labels in {-1, 1}.
        tx: numpy array of shape (N, D).
        config: dict with lambda_.

    Returns:
        w: numpy array of shape (D,).
    """
    cfg = merge_config(config)
    w, _ = ridge_regression(np.asarray(y, dtype=np.float64), tx, cfg["lambda_"])
    return w


def linear_scores(w, tx):
    """Linear scores tx @ w (for ridge / least squares).

    Args:
        w: numpy array of shape (D,).
        tx: numpy array of shape (N, D).

    Returns:
        numpy array of shape (N,).
    """
    return tx @ w


def prepare_folds(y, xc, names, k_indices, config):
    """Fit the preprocessor on each training fold and transform both parts.

    Args:
        y: numpy array of shape (N,), labels in {-1, 1}.
        xc: numpy array of shape (N, D), CLEANED features (clean_data output).
        names: list of D column names of xc.
        k_indices: output of build_k_indices.
        config: dict, preprocessing options (merged with DEFAULT_CONFIG).

    Returns:
        List of k tuples (tx_tr, y_tr, tx_te, y_te, state).
    """
    cfg = merge_config(config)
    folds = []
    for k in range(len(k_indices)):
        tr_idx, te_idx = fold_split(k_indices, k)
        xc_tr = xc[tr_idx]
        state = fit_preprocessor(xc_tr, names, cfg, y=y[tr_idx])
        tx_tr = transform(xc_tr, state)
        del xc_tr
        tx_te = transform(xc[te_idx], state)
        folds.append((tx_tr, y[tr_idx], tx_te, y[te_idx], state))
    return folds


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
    """Run k-fold cross-validation for one preprocessing + model setting.

    For each fold k: preprocess with statistics of the training folds only,
    optionally undersample the negatives of the training fold (seed =
    config["seed"] + k), train with train_fn, choose the threshold that
    maximizes F1 on the scores of the FULL training fold (unless
    config["threshold"] is a float), then score the held-out fold.

    Args:
        y: numpy array of shape (N,), labels in {-1, 1}.
        xc: numpy array of shape (N, D), cleaned features (unused if folds).
        names: list of D column names of xc (unused if folds is given).
        k_indices: output of build_k_indices.
        config: dict, preprocessing and model options.
        train_fn: callable (y_tr, tx_tr, config) -> w.
        score_fn: callable (w, tx) -> scores of shape (N,).
        folds: optional output of prepare_folds for the same k_indices and
            preprocessing config, to skip the preprocessing.
        verbose: bool, print one line per fold.

    Returns:
        dict with numpy arrays of shape (k,) for "f1", "accuracy",
        "precision", "recall", "threshold", "f1_train", their "<m>_mean"
        and "<m>_std" floats, and "n_features" (int, width of tx).
    """
    cfg = merge_config(config)
    if folds is None:
        folds = prepare_folds(y, xc, names, k_indices, cfg)
    res = dict((m, []) for m in METRICS)
    for k, (tx_tr, y_tr, tx_te, y_te, _) in enumerate(folds):
        if cfg["undersample_ratio"] is not None:
            rows = undersample_indices(y_tr, cfg["undersample_ratio"], cfg["seed"] + k)
            w = train_fn(y_tr[rows], tx_tr[rows], cfg)
        else:
            w = train_fn(y_tr, tx_tr, cfg)
        scores_tr = score_fn(w, tx_tr)
        if cfg["threshold"] is None:
            thr, f1_tr = best_threshold(y_tr, scores_tr)
        else:
            thr = float(cfg["threshold"])
            f1_tr = precision_recall_f1(y_tr, predict_labels(scores_tr, thr))[2]
        pred = predict_labels(score_fn(w, tx_te), thr)
        prec, rec, f1 = precision_recall_f1(y_te, pred)
        res["f1"].append(f1)
        res["accuracy"].append(accuracy(y_te, pred))
        res["precision"].append(prec)
        res["recall"].append(rec)
        res["threshold"].append(thr)
        res["f1_train"].append(f1_tr)
        if verbose:
            print(
                "fold %d: F1 %.4f (train %.4f) acc %.4f thr %.4f"
                % (k, f1, f1_tr, res["accuracy"][-1], thr)
            )
    out = {}
    for m in METRICS:
        arr = np.array(res[m], dtype=np.float64)
        out[m] = arr
        out[m + "_mean"] = float(np.mean(arr))
        out[m + "_std"] = float(np.std(arr))
    out["n_features"] = int(folds[0][0].shape[1])
    return out


def _grid_combinations(param_grid):
    """All combinations of a parameter grid, last key varying fastest.

    Args:
        param_grid: dict name -> list of values.

    Returns:
        List of dicts, in grid order.
    """
    combos = [{}]
    for key in param_grid:
        combos = [dict(c, **{key: v}) for c in combos for v in param_grid[key]]
    return combos


def _prep_key(cfg):
    """Hashable cache key of the preprocessing part of a config.

    Args:
        cfg: dict, merged config.

    Returns:
        tuple of the PREP_KEYS values (lists turned into tuples).
    """

    def freeze(v):
        """Turn lists (also nested) into tuples."""
        if isinstance(v, (list, tuple)):
            return tuple(freeze(e) for e in v)
        return v

    return tuple(freeze(cfg[k]) for k in PREP_KEYS)


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
    """Evaluate every combination of param_grid with cross_validate.

    All combinations use the same folds. The preprocessed folds are computed
    once per distinct tuple of PREP_KEYS values: combinations that change
    only model keys reuse them. Only one set of folds is kept in memory at a
    time (combinations are evaluated grouped by preprocessing key; the
    results are returned in grid order).

    Args:
        y: numpy array of shape (N,), labels in {-1, 1}.
        xc: numpy array of shape (N, D), cleaned features.
        names: list of D column names of xc.
        k_indices: output of build_k_indices.
        config: dict, base config (merged with DEFAULT_CONFIG).
        param_grid: dict name -> list of values, e.g. {"lambda_": [...]}.
        train_fn: callable (y_tr, tx_tr, config) -> w.
        score_fn: callable (w, tx) -> scores.
        verbose: bool, print one line per combination.

    Returns:
        best_params: dict, the combination with the highest f1_mean (ties ->
            first in grid order).
        results: list of (params, cv_result) in grid order.
    """
    base = merge_config(config)
    combos = _grid_combinations(param_grid)
    keys = [_prep_key(dict(base, **p)) for p in combos]
    order = []
    for key in keys:
        if key not in order:
            order.append(key)
    cv_results = [None] * len(combos)
    for key in order:
        folds = None
        for i, p in enumerate(combos):
            if keys[i] != key:
                continue
            cfg = dict(base, **p)
            if folds is None:
                folds = prepare_folds(y, xc, names, k_indices, cfg)
            cv_results[i] = cross_validate(
                y, xc, names, k_indices, cfg, train_fn, score_fn, folds=folds
            )
            if verbose:
                print(
                    "%s: F1 %.4f +/- %.4f"
                    % (p, cv_results[i]["f1_mean"], cv_results[i]["f1_std"])
                )
        del folds
    results = list(zip(combos, cv_results))
    best = 0
    for i in range(1, len(results)):
        if results[i][1]["f1_mean"] > results[best][1]["f1_mean"]:
            best = i
    return results[best][0], results
