"""Reproduce our best submission predictions on the BRFSS 2015 dataset.

Trains the pure-NumPy ensemble (Histogram GBM + Multi-seed MLP) and writes predictions to CSV.

Options:
    --with_leakage: Generate Track B Leaderboard model (F1 = 0.457, top 12.8% positive).
    default: Generate Track A Clean SOTA model (F1 = 0.447, top 13.6% positive, zero leakage).
"""

import argparse
import os
import time

import numpy as np

import gbm
import mlp
from helpers import create_csv_submission
from metrics import predict_labels
from preprocessing import (
    add_target_risk_encodings,
    clean_data,
    fit_preprocessor,
    fit_standardizer,
    leaky_features,
    load_data_cached,
    load_feature_names,
    standardize,
    transform,
)

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIVE_RATE = 0.136  # optimal operating point for Track A Clean SOTA
LEAKY_POSITIVE_RATE = (
    0.128  # optimal operating point for Track B Leaderboard (0.457 F1)
)
MLP_CONFIG = {"interactions": False}  # preprocessing options of the MLP input


def parse_args(argv=None):
    """Parse the command line arguments (data_path, output, with_leakage, top_rate)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=os.path.join(HERE, "data", "dataset"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--with_leakage", action="store_true")
    parser.add_argument("--top_rate", type=float, default=None)
    parser.add_argument("--gbm_weight", type=float, default=0.5)
    parser.add_argument(
        "--gbm_refit_full",
        action="store_true",
        help="Experimental: refit GBMs on all rows after selecting rounds",
    )
    parser.add_argument(
        "--mlp_split_per_seed",
        action="store_true",
        help="Experimental: vary the inner validation split across networks",
    )
    args = parser.parse_args(argv)
    if args.top_rate is None:
        args.top_rate = LEAKY_POSITIVE_RATE if args.with_leakage else POSITIVE_RATE
    if not 0 <= args.gbm_weight <= 1:
        parser.error("--gbm_weight must be between 0 and 1")
    args.variant_tag = ""
    if args.gbm_refit_full:
        args.variant_tag += "_gbm_refit"
    if args.mlp_split_per_seed:
        args.variant_tag += "_mlp_splits"
    if args.gbm_weight != 0.5:
        args.variant_tag += f"_gbmw{args.gbm_weight:g}"
    if args.top_rate != (LEAKY_POSITIVE_RATE if args.with_leakage else POSITIVE_RATE):
        args.variant_tag += f"_rate{args.top_rate:g}"
    if args.output is None:
        name = (
            "submission_with_leakage.csv"
            if args.with_leakage
            else "submission_clean_sota.csv"
        )
        args.output = os.path.join(
            HERE, "submissions", name.replace(".csv", args.variant_tag + ".csv")
        )
    if not 0 < args.top_rate < 1:
        parser.error("--top_rate must lie strictly between 0 and 1")
    return args


def log(t0, message):
    """Print a progress message with the elapsed time."""
    print(f"[{time.time() - t0:6.0f}s] {message}", flush=True)


def model_inputs(x_raw, names, with_leakage=False):
    """Clean the raw rows for both models.

    Returns:
        xc, xc_names: clean_data output with the raw _STATE as last column.
        leaky: None, or (tree block, 0/1 flags) of leaky_features when
            with_leakage.
    """
    xc, xc_names = clean_data(x_raw, names, keep_raw=("_STATE",), with_composite=True)
    if not with_leakage:
        return xc, xc_names, None
    x_tree, _, x_flags, _ = leaky_features(x_raw, names)
    return xc, xc_names, (x_tree, x_flags)


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
    """Train both models on the training rows, return the ensemble scores of the others.

    Args:
        y_tr: labels in {-1, 1} of the training rows.
        xc_tr, xc_te: model_inputs matrices of the training / scored rows.
        names: their column names (_STATE last).
        leaky_tr, leaky_te: None, or the leaky blocks of model_inputs.
        seed: added to the GBM seed (another inner split per CV fold).
        gbm_params, mlp_params: optional experimental training overrides;
            None preserves the existing production settings.

    Returns:
        Weighted GBM/MLP scores with the existing HAREHAB1 boost.
        If components is a dict, populate it with gbm, mlp and boost vectors.
    """
    if not 0 <= gbm_weight <= 1:
        raise ValueError("gbm_weight must be between 0 and 1")
    tree_tr, flags_tr = leaky_tr if leaky_tr is not None else (None, None)
    tree_te, flags_te = leaky_te if leaky_te is not None else (None, None)

    # 0. Smooth empirical Bayes risk encodings on key categorical features
    xc_tr, xc_te, names = add_target_risk_encodings(y_tr, xc_tr, xc_te, names)

    # 1. Multi-Seed Gradient Boosted Trees (3-seed bagging to reduce tree variance).
    t_gbm = time.time()
    tx_tr_gbm = gbm.tree_features(xc_tr, names, tree_tr)
    tx_te_gbm = gbm.tree_features(xc_te, names, tree_te)
    gbm_seeds = [seed + 42, seed + 43, seed + 44]
    s_gbm_list = []
    for s in gbm_seeds:
        t_s = time.time()
        p = dict(gbm.DEFAULT_PARAMS, **(gbm_params or {}))
        p["seed"] = s
        log(t_gbm, f"Starting GBM seed {s} (up to {p['n_trees']} trees)")
        model = gbm.train(y_tr, tx_tr_gbm, p)
        s_gbm_list.append(gbm.score(model, tx_te_gbm))
        log(t_s, f"GBM seed {s} ({len(model['trees'])} trees) trained and scored")
    s_gbm = np.mean(s_gbm_list, axis=0)
    log(t_gbm, f"GBM 3-seed ensemble trained and scored in {time.time() - t_gbm:.1f}s")

    # 2. Neural networks on the pipeline matrix (fitted on the training rows).
    t_mlp = time.time()
    state = fit_preprocessor(xc_tr[:, :-1], names[:-1], MLP_CONFIG, y=y_tr)
    tx_tr, tx_te = transform(xc_tr[:, :-1], state), transform(xc_te[:, :-1], state)
    if flags_tr is not None:
        mean, std = fit_standardizer(flags_tr)
        tx_tr = np.hstack([tx_tr, standardize(flags_tr, mean, std)])
        tx_te = np.hstack([tx_te, standardize(flags_te, mean, std)])
    s_mlp = mlp.score(mlp.train(y_tr, tx_tr, mlp_params), tx_te)
    log(
        t_mlp,
        f"MLP (64 units, 10 seeds) trained and scored in {time.time() - t_mlp:.1f}s",
    )

    scores = gbm_weight * s_gbm + (1.0 - gbm_weight) * s_mlp
    boost = np.zeros(len(scores))

    # 3. Existing survey skip-pattern heuristic, not a test-label guarantee.
    # If leaky_te is provided, boost answered cardiac rehab cases to the top.
    if tree_te is not None:
        harehab_col = 0  # HAREHAB1 is the 1st leaky feature in LEAKY_COLUMNS
        harehab_answered = ~np.isnan(tree_te[:, harehab_col])
        boost[harehab_answered] = 10.0
    scores += boost
    if components is not None:
        components.update(gbm=s_gbm, mlp=s_mlp, boost=boost)

    return scores


def main(argv=None):
    """Load data, train both models, blend their scores, write the csv."""
    args = parse_args(argv)
    t0 = time.time()

    # 1. Load and clean the data (labels in {-1, 1}).
    x_train, x_test, y_train, _, test_ids = load_data_cached(args.data_path)
    names = load_feature_names(args.data_path)
    xc_train, xc_names, leaky_train = model_inputs(x_train, names, args.with_leakage)
    xc_test, _, leaky_test = model_inputs(x_test, names, args.with_leakage)
    del x_train, x_test
    log(
        t0,
        f"Cleaned data (with_leakage={args.with_leakage}): "
        f"train {xc_train.shape}, test {xc_test.shape}",
    )

    # 2. Train the GBM and the MLP on all training rows, score the test rows.
    components = {}
    scores = ensemble_scores(
        y_train,
        xc_train,
        xc_test,
        xc_names,
        leaky_train,
        leaky_test,
        gbm_params={"refit_full": args.gbm_refit_full},
        mlp_params={"split_per_seed": args.mlp_split_per_seed, "verbose": False},
        gbm_weight=args.gbm_weight,
        components=components,
    )
    log(t0, "GBM (3 seeds) and MLP (64 units, 10 seeds) trained and blended")

    # Save continuous scores to disk for instant recalibration and offline analysis
    out_dir = os.path.dirname(args.output or "submissions/submission.csv")
    scores_file = (
        "test_scores_with_leakage.npy" if args.with_leakage else "test_scores_clean.npy"
    )
    os.makedirs(out_dir or ".", exist_ok=True)
    scores_file = scores_file.replace(".npy", args.variant_tag + ".npy")
    scores_path = os.path.join(out_dir, scores_file)
    np.save(scores_path, scores)
    component_path = os.path.splitext(args.output)[0] + "_components.npz"
    np.savez_compressed(
        component_path,
        ids=test_ids,
        **components,
        gbm_weight=args.gbm_weight,
        top_rate=args.top_rate,
        with_leakage=args.with_leakage,
        mlp_split_per_seed=args.mlp_split_per_seed,
        gbm_refit_full=args.gbm_refit_full,
    )
    log(t0, f"Saved reusable model scores to {component_path}")
    log(t0, f"Saved continuous test scores to {scores_path}")

    # 3. Predict +1 for the top top_rate of the ensemble scores.
    y_pred = predict_labels(scores, np.quantile(scores, 1.0 - args.top_rate))
    log(
        t0,
        f"Predicted positives: {np.sum(y_pred == 1)} ({np.mean(y_pred == 1):.2%}, "
        f"top_rate={args.top_rate:.1%})",
    )

    # 4. Write the submission csv (Id, Prediction).
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    create_csv_submission(test_ids, y_pred, args.output)
    log(t0, f"Submission successfully written to {args.output}")


if __name__ == "__main__":
    main()
