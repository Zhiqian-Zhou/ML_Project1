"""Reproduce our best AIcrowd submission from scratch.

Final model (chosen by 5-fold cross-validation on the training rows):
    * gbm.py: gradient boosted trees on the cleaned features + composite vascular
      multi-morbidity indicators, with cost-sensitive positive weighting (w_pos=2.0),
      the state of residence replaced by its smoothed prevalence of heart disease.
    * mlp.py: mean of 10 small neural networks (64 ReLU units) on the
      standardized pipeline features.
    * score = 0.5 * Rank(GBM) + 0.5 * Rank(MLP); the ~13.6% of test rows with
      the highest blended rank percentiles are predicted +1.
Every random step uses a fixed seed, so the csv is reproducible at every run.

--with_leakage: the variant that also uses the leaky questions, which are
answered only after a heart attack (preprocessing.LEAKY_COLUMNS). It shows
how much they leak; it is not our official clinical model, because these answers
do not exist for the undiagnosed people a screening model is meant for.

Usage (the default paths are relative to this file):
    python run.py [--data_path data/dataset] [--output submissions/submission_clean_sota.csv]
                  [--with_leakage] [--top_rate 0.136]
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
POSITIVE_RATE = (
    0.136  # share of test rows predicted +1 (validated via micro-benchmarks)
)
MLP_CONFIG = {"interactions": False}  # preprocessing options of the MLP input


def parse_args():
    """Parse the command line arguments (data_path, output, with_leakage, top_rate)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=os.path.join(HERE, "data", "dataset"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--with_leakage", action="store_true")
    parser.add_argument("--top_rate", type=float, default=POSITIVE_RATE)
    args = parser.parse_args()
    if args.output is None:
        name = (
            "submission_with_leakage.csv"
            if args.with_leakage
            else "submission_clean_sota.csv"
        )
        args.output = os.path.join(HERE, "submissions", name)
    return args


def log(t0, message):
    """Print a progress message with the elapsed time."""
    print(f"[{time.time() - t0:6.0f}s] {message}", flush=True)


def rank_normalize(scores):
    """Normalize continuous decision scores into uniform percentile ranks in [0, 1]."""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n <= 1:
        return np.zeros_like(scores)
    ranks = np.empty(n, dtype=np.float64)
    ranks[np.argsort(scores)] = np.linspace(0.0, 1.0, n)
    return ranks


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
    y_tr, xc_tr, xc_te, names, leaky_tr=None, leaky_te=None, seed=0, use_rank=True
):
    """Train both models on the training rows, return the ensemble scores of the others.

    Args:
        y_tr: labels in {-1, 1} of the training rows.
        xc_tr, xc_te: model_inputs matrices of the training / scored rows.
        names: their column names (_STATE last).
        leaky_tr, leaky_te: None, or the leaky blocks of model_inputs.
        seed: added to the GBM seed (another inner split per CV fold).
        use_rank: bool, whether to average percentile ranks (True) or raw logits (False).

    Returns:
        0.5 * Score(GBM) + 0.5 * Score(MLP) of the scored rows.
    """
    tree_tr, flags_tr = leaky_tr if leaky_tr is not None else (None, None)
    tree_te, flags_te = leaky_te if leaky_te is not None else (None, None)

    # 1. Gradient boosted trees on the cleaned matrix (cost-sensitive w_pos=2.0).
    params = {"seed": gbm.DEFAULT_PARAMS["seed"] + seed}
    model = gbm.train(y_tr, gbm.tree_features(xc_tr, names, tree_tr), params)
    s_gbm = gbm.score(model, gbm.tree_features(xc_te, names, tree_te))

    # 2. Neural networks on the pipeline matrix (fitted on the training rows).
    state = fit_preprocessor(xc_tr[:, :-1], names[:-1], MLP_CONFIG, y=y_tr)
    tx_tr, tx_te = transform(xc_tr[:, :-1], state), transform(xc_te[:, :-1], state)
    if flags_tr is not None:
        mean, std = fit_standardizer(flags_tr)
        tx_tr = np.hstack([tx_tr, standardize(flags_tr, mean, std)])
        tx_te = np.hstack([tx_te, standardize(flags_te, mean, std)])
    s_mlp = mlp.score(mlp.train(y_tr, tx_tr), tx_te)

    if use_rank:
        return 0.5 * rank_normalize(s_gbm) + 0.5 * rank_normalize(s_mlp)
    return 0.5 * s_gbm + 0.5 * s_mlp


def main():
    """Load data, train both models, blend their scores, write the csv."""
    args = parse_args()
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
    scores = ensemble_scores(
        y_train, xc_train, xc_test, xc_names, leaky_train, leaky_test
    )
    log(t0, "GBM (w_pos=2.0) and MLP (10 seeds) trained and blended")

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
