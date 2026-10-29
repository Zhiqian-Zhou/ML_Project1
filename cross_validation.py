"""Stratified K-fold cross-validation and evaluation harness.

Provides:
    1. Stratified K-fold index generation (build_k_indices) and 80/20 holdout split (holdout_split).
    2. Linear model cross-validation and hyperparameter grid search (cross_validate, grid_search).
    3. Final ensemble evaluation (GBM + MLP, top-percentile rule, ensemble_scores).

All preprocessing and model fitting strictly occur on training folds to prevent data leakage.
"""

import argparse
import os
import time

import numpy as np

from implementations import reg_logistic_regression, ridge_regression, sigmoid
from metrics import (
    accuracy,
    best_threshold,
    precision_recall_f1,
    predict_labels,
)
from models import (
    DEFAULT_GBM_PARAMS,
    DEFAULT_MLP_PARAMS,
    GradientBoostedTrees,
    MLPEnsemble,
    tree_features,
)
from preprocessing import (
    CLEAN_RATE,
    LEAKY_RATE,
    LEAKY_POSITIVE_RATE,
    POSITIVE_RATE,
    add_target_risk_encodings,
    fit_preprocessor,
    fit_standardizer,
    load_data_cached,
    load_feature_names,
    merge_config,
    model_inputs,
    standardize,
    transform,
)

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ("f1", "accuracy", "precision", "recall", "threshold", "f1_train")


def build_k_indices(y, k_fold=5, seed=42):
    """Split the row indices into k_fold stratified random folds.

    Positives and negatives are shuffled separately and each class is cut
    into k_fold nearly equal parts, so every fold has the same positive ratio
    (8.8% positives) and no row is dropped.
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


def holdout_split(y, frac=0.2, seed=2026):
    """Stratified split of rows: (dev_idx, holdout_idx) with frac held out per class."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == -1)[0]
    hold = np.concatenate(
        [
            rng.choice(pos, int(frac * len(pos)), replace=False),
            rng.choice(neg, int(frac * len(neg)), replace=False),
        ]
    )
    hold = np.sort(hold)
    return np.setdiff1d(np.arange(len(y)), hold), hold


def f1_top_rate(y, scores, top_rate=CLEAN_RATE):
    """Evaluate positive-class F1 when the top top_rate scores are predicted +1."""
    y_pred = predict_labels(scores, np.quantile(scores, 1.0 - top_rate))
    return precision_recall_f1(y, y_pred)


# ----------------------------------------------------------------------
# Linear baselines


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
    """Fit the preprocessor on each training fold and transform both parts."""
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
    """Add '<m>_mean' and '<m>_std' to a dict of per-fold metric lists."""
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
    """K-fold cross-validation of one preprocessing + linear model setting."""
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
    """Cross-validate every combination of MODEL parameters in param_grid."""
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


# ----------------------------------------------------------------------
# Final Ensemble Evaluation (GBM + MLP)


def ensemble_scores(
    y_tr,
    xc_tr,
    xc_te,
    names,
    leaky_tr=None,
    leaky_te=None,
    seed=0,
    gbm_params=None,
    mlp_params=None,
    gbm_weight=0.5,
    components=None,
):
    """Train both models on training rows and return blended ensemble scores for test rows."""
    if not 0 <= gbm_weight <= 1:
        raise ValueError("gbm_weight must be between 0 and 1")
    tree_tr, flags_tr = leaky_tr if leaky_tr is not None else (None, None)
    tree_te, flags_te = leaky_te if leaky_te is not None else (None, None)

    # 1. Smooth empirical Bayes risk encodings on key categorical features
    xc_tr, xc_te, enc_names = add_target_risk_encodings(y_tr, xc_tr, xc_te, names)

    # 2. Multi-Seed Gradient Boosted Trees (3-seed bagging)
    tx_tr_gbm = tree_features(xc_tr, enc_names, tree_tr)
    tx_te_gbm = tree_features(xc_te, enc_names, tree_te)
    gbm_seeds = [seed + 42, seed + 43, seed + 44]
    s_gbm_list = []
    for s in gbm_seeds:
        p = dict(DEFAULT_GBM_PARAMS, **(gbm_params or {}))
        p["seed"] = s
        m = GradientBoostedTrees(**p).fit(tx_tr_gbm, y_tr)
        s_gbm_list.append(m.predict_logits(tx_te_gbm))
    s_gbm = np.mean(s_gbm_list, axis=0)

    # 3. Neural networks on the pipeline matrix
    mlp_cfg = {"interactions": False}
    state = fit_preprocessor(xc_tr[:, :-1], enc_names[:-1], mlp_cfg, y=y_tr)
    tx_tr = transform(xc_tr[:, :-1], state)
    tx_te = transform(xc_te[:, :-1], state)
    if flags_tr is not None:
        mean, std = fit_standardizer(flags_tr)
        tx_tr = np.hstack([tx_tr, standardize(flags_tr, mean, std)])
        tx_te = np.hstack([tx_te, standardize(flags_te, mean, std)])
    mlp_p = dict(DEFAULT_MLP_PARAMS, **(mlp_params or {}))
    mlp_p["seed"] = seed
    mlp_model = MLPEnsemble(**mlp_p).fit(tx_tr, y_tr)
    s_mlp = mlp_model.predict_logits(tx_te)

    # 4. Ensemble blending
    scores = gbm_weight * s_gbm + (1.0 - gbm_weight) * s_mlp
    boost = np.zeros(len(scores))
    if tree_te is not None:
        harehab_answered = ~np.isnan(tree_te[:, 0])
        boost[harehab_answered] = 10.0
    scores += boost

    if components is not None:
        components.update(gbm=s_gbm, mlp=s_mlp, boost=boost)
    return scores


def fold_scores(
    y,
    xc,
    names,
    leaky,
    tr,
    te,
    k,
    gbm_params=None,
    mlp_params=None,
    gbm_weight=0.5,
    scores_dir=None,
    score_tag=None,
):
    """Train ensemble on tr rows and return test scores on te rows."""
    leaky_tr = leaky_te = None
    if leaky is not None:
        leaky_tr = tuple(block[tr] for block in leaky)
        leaky_te = tuple(block[te] for block in leaky)
    components = {}
    scores = ensemble_scores(
        y[tr],
        xc[tr],
        xc[te],
        names,
        leaky_tr,
        leaky_te,
        seed=k,
        gbm_params=gbm_params,
        mlp_params=mlp_params,
        gbm_weight=gbm_weight,
        components=components,
    )
    if scores_dir is not None:
        os.makedirs(scores_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(scores_dir, (score_tag or f"fold_{k}") + ".npz"),
            row_indices=te,
            y=y[te],
            **components,
            gbm_weight=gbm_weight,
        )
    return scores


def parse_args(argv=None):
    """Parse command line arguments for cross-validation runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=os.path.join(HERE, "data", "dataset"))
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Evaluate 80% dev CV and 20% held-out test",
    )
    parser.add_argument(
        "--with_leakage",
        action="store_true",
        help="Evaluate Track B leaky competition variant",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Evaluate Track A clean screening variant"
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=None,
        help="Number of rows to subsample for quick check",
    )
    parser.add_argument(
        "--k_fold", type=int, default=5, help="Number of folds (default: 5)"
    )
    parser.add_argument(
        "--top_rate", type=float, default=None, help="Percentile threshold override"
    )
    parser.add_argument("--gbm_weight", type=float, default=0.5)
    parser.add_argument("--scores_dir", default=None)
    parser.add_argument("--gbm_refit_full", action="store_true")
    parser.add_argument("--mlp_split_per_seed", action="store_true")
    args = parser.parse_args(argv)

    if args.clean:
        args.with_leakage = False

    if args.top_rate is None:
        args.top_rate = LEAKY_RATE if args.with_leakage else CLEAN_RATE
    if not 0 <= args.gbm_weight <= 1 or not 0 < args.top_rate < 1:
        parser.error("Invalid blend weight or positive rate")
    if args.k_fold == 1 or args.k_fold < 0 or (args.k_fold == 0 and not args.holdout):
        parser.error("Use k_fold >= 2, or --holdout --k_fold 0")
    return args


def main(argv=None):
    """Run cross-validation on BRFSS dataset and report performance."""
    args = parse_args(argv)
    t0 = time.time()
    model_options = dict(
        gbm_params={"refit_full": args.gbm_refit_full},
        mlp_params={"split_per_seed": args.mlp_split_per_seed, "verbose": False},
        gbm_weight=args.gbm_weight,
        scores_dir=args.scores_dir,
    )
    x, _, y, _, _ = load_data_cached(args.data_path)
    names = load_feature_names(args.data_path)
    xc, xc_names, leaky = model_inputs(x[: args.n_rows], names, args.with_leakage)
    y = y[: args.n_rows]
    del x
    rows = np.arange(len(y))
    if args.holdout:
        rows, hold = holdout_split(y)

    print("=" * 70)
    print(
        f"Running Cross-Validation: with_leakage={args.with_leakage}, top_rate={args.top_rate:.2%}"
    )
    print(
        f"Dataset Size: {len(y)} rows (Development: {len(rows)} rows, Folds: {args.k_fold})"
    )
    print("=" * 70)

    if args.k_fold > 0:
        k_indices = build_k_indices(y[rows], args.k_fold, 42)
        f1s = []
        for k in range(args.k_fold):
            tr, te = (rows[i] for i in fold_split(k_indices, k))
            scores = fold_scores(y, xc, xc_names, leaky, tr, te, k, **model_options)
            p, r, f1 = f1_top_rate(y[te], scores, args.top_rate)
            f1s.append(f1)
            print(
                f"Fold {k + 1}/{args.k_fold}: F1 {f1:.4f} (Precision {p:.3f}, Recall {r:.3f}) [{time.time() - t0:.0f}s]",
                flush=True,
            )
        print(
            f"--> CV F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} on {len(rows)} rows",
            flush=True,
        )

    if args.holdout:
        scores = fold_scores(
            y,
            xc,
            xc_names,
            leaky,
            rows,
            hold,
            args.k_fold,
            score_tag="holdout",
            **model_options,
        )
        p, r, f1 = f1_top_rate(y[hold], scores, args.top_rate)
        print(
            f"--> Held-out F1: {f1:.4f} (Precision {p:.3f}, Recall {r:.3f}) on {len(hold)} rows",
            flush=True,
        )


if __name__ == "__main__":
    main()
