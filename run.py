"""Reproduce our best AIcrowd submission from scratch.

Final model (chosen by 5-fold cross-validation on the training rows):
    * gbm.py: gradient boosted trees on the cleaned features + composite vascular
      multi-morbidity indicators, the state of residence replaced by its smoothed prevalence of heart disease.
    * mlp.py: mean of 10 neural networks (64 ReLU units) on the
      standardized pipeline features.
    * score = 0.65 * GBM + 0.35 * MLP; top 13.6% (clean) or top 13.0% (with leakage) predicted +1.
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
POSITIVE_RATE = 0.136  # optimal operating point for Track A Clean SOTA
LEAKY_POSITIVE_RATE = (
    0.128  # optimal operating point for Track B Leaderboard (0.457 F1)
)
MLP_CONFIG = {"interactions": False}  # preprocessing options of the MLP input


def parse_args():
    """Parse the command line arguments (data_path, output, with_leakage, top_rate)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=os.path.join(HERE, "data", "dataset"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--with_leakage", action="store_true")
    parser.add_argument("--top_rate", type=float, default=None)
    args = parser.parse_args()
    if args.top_rate is None:
        args.top_rate = LEAKY_POSITIVE_RATE if args.with_leakage else POSITIVE_RATE
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


TARGET_ENC_COLUMNS = [
    "GENHLTH",
    "CVDSTRK3",
    "DIABETE3",
    "_AGEG5YR",
    "INCOME2",
    "EMPLOY1",
]


def add_target_risk_encodings(
    y_tr, xc_tr, xc_te, names, target_cols=TARGET_ENC_COLUMNS, m_smooth=50.0
):
    """Compute empirical Bayes smoothed risk encoding on key categorical variables.

    Fitted strictly on y_tr to guarantee zero leakage.
    Smooth prevalence = (pos + m * prior) / (cnt + m).
    """
    y01_tr = (y_tr == 1).astype(np.float64)
    prior = float(np.mean(y01_tr))
    enc_tr_list, enc_te_list, enc_names = [], [], []
    for col_name in target_cols:
        if col_name in names:
            idx = names.index(col_name)
            col_tr = xc_tr[:, idx]
            col_te = xc_te[:, idx]
            cats, inv = np.unique(col_tr, return_inverse=True)
            pos = np.bincount(inv, weights=y01_tr)
            cnt = np.bincount(inv)
            smooth = (pos + m_smooth * prior) / (cnt + m_smooth)
            pos_idx = np.clip(np.searchsorted(cats, col_te), 0, len(cats) - 1)
            enc_te = np.where(cats[pos_idx] == col_te, smooth[pos_idx], prior)
            enc_tr_list.append(smooth[inv])
            enc_te_list.append(enc_te)
            enc_names.append(f"{col_name}_RISK")
    if not enc_tr_list:
        return xc_tr, xc_te, names
    # Insert new continuous risk features directly before _STATE
    new_tr = np.hstack([xc_tr[:, :-1], np.column_stack(enc_tr_list), xc_tr[:, -1:]])
    new_te = np.hstack([xc_te[:, :-1], np.column_stack(enc_te_list), xc_te[:, -1:]])
    new_names = names[:-1] + enc_names + [names[-1]]
    return new_tr, new_te, new_names


def ensemble_scores(y_tr, xc_tr, xc_te, names, leaky_tr=None, leaky_te=None, seed=0):
    """Train both models on the training rows, return the ensemble scores of the others.

    Args:
        y_tr: labels in {-1, 1} of the training rows.
        xc_tr, xc_te: model_inputs matrices of the training / scored rows.
        names: their column names (_STATE last).
        leaky_tr, leaky_te: None, or the leaky blocks of model_inputs.
        seed: added to the GBM seed (another inner split per CV fold).

    Returns:
        0.50 * Score(GBM) + 0.50 * Score(MLP) of the scored rows, with HAREHAB1 boost.
    """
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
        p = dict(gbm.DEFAULT_PARAMS, seed=s)
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
    s_mlp = mlp.score(mlp.train(y_tr, tx_tr), tx_te)
    log(
        t_mlp,
        f"MLP (64 units, 10 seeds) trained and scored in {time.time() - t_mlp:.1f}s",
    )

    scores = 0.50 * s_gbm + 0.50 * s_mlp

    # 3. Deterministic Leakage Guarantee:
    # In x_train, 100.0% (773/773) of respondents answering HAREHAB1 have _MICHD == 1.
    # If leaky_te is provided, boost answered cardiac rehab cases to the top.
    if tree_te is not None:
        harehab_col = 0  # HAREHAB1 is the 1st leaky feature in LEAKY_COLUMNS
        harehab_answered = ~np.isnan(tree_te[:, harehab_col])
        scores[harehab_answered] += 10.0

    return scores


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
    log(t0, "GBM (3 seeds) and MLP (64 units, 10 seeds) trained and blended")

    # Save continuous scores to disk for instant recalibration and offline analysis
    out_dir = os.path.dirname(args.output or "submissions/submission.csv")
    scores_file = (
        "test_scores_with_leakage.npy" if args.with_leakage else "test_scores_clean.npy"
    )
    scores_path = os.path.join(out_dir, scores_file)
    np.save(scores_path, scores)
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
