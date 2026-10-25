"""Reproduce our best AIcrowd submission from scratch.

Final model (chosen by 5-fold cross-validation on the training rows):
    * gbm.py: gradient boosted trees on the cleaned features, the state of
      residence replaced by its smoothed prevalence of heart disease.
    * mlp.py: mean of 10 small neural networks (64 ReLU units) on the
      standardized pipeline features.
    * score = 0.5 * GBM logit + 0.5 * MLP logit; the 14% of test rows with
      the highest scores are predicted +1 (the share of positives that
      maximized F1 in cross-validation).
Every random step uses a fixed seed, so the csv is the same at every run.

--with_leakage: the variant that also uses the leaky questions, which are
answered only after a heart attack (preprocessing.LEAKY_COLUMNS). It shows
how much they leak; it is not our model, because these answers do not exist
for the undiagnosed people a screening model is meant for. Same models and
parameters; only the predicted share is 13% (chosen in cross-validation).

Usage (the default paths are relative to this file):
    python run.py [--data_path data/dataset] [--output submissions/submission.csv]
                  [--with_leakage]
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
POSITIVE_RATE = 0.14  # share of test rows predicted +1
LEAKY_POSITIVE_RATE = 0.13  # the same with the leaky questions
MLP_CONFIG = {"interactions": False}  # preprocessing options of the MLP input


def parse_args():
    """Parse the command line arguments (data_path, output, with_leakage)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=os.path.join(HERE, "data", "dataset"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--with_leakage", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        name = "submission_with_leakage.csv" if args.with_leakage else "submission.csv"
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
    xc, xc_names = clean_data(x_raw, names, keep_raw=("_STATE",))
    if not with_leakage:
        return xc, xc_names, None
    x_tree, _, x_flags, _ = leaky_features(x_raw, names)
    return xc, xc_names, (x_tree, x_flags)


def ensemble_scores(y_tr, xc_tr, xc_te, names, leaky_tr=None, leaky_te=None, seed=0):
    """Train both models on the training rows, return the logits of the others.

    Args:
        y_tr: labels in {-1, 1} of the training rows.
        xc_tr, xc_te: model_inputs matrices of the training / scored rows.
        names: their column names (_STATE last).
        leaky_tr, leaky_te: None, or the leaky blocks of model_inputs.
        seed: added to the GBM seed (another inner split per CV fold).

    Returns:
        0.5 * GBM logit + 0.5 * MLP logit of the scored rows.
    """
    tree_tr, flags_tr = leaky_tr if leaky_tr is not None else (None, None)
    tree_te, flags_te = leaky_te if leaky_te is not None else (None, None)
    # Gradient boosted trees on the cleaned matrix.
    params = {"seed": gbm.DEFAULT_PARAMS["seed"] + seed}
    model = gbm.train(y_tr, gbm.tree_features(xc_tr, names, tree_tr), params)
    s_gbm = gbm.score(model, gbm.tree_features(xc_te, names, tree_te))
    # Neural networks on the pipeline matrix (fitted on the training rows;
    # without its last column xc is the default clean_data output).
    state = fit_preprocessor(xc_tr[:, :-1], names[:-1], MLP_CONFIG, y=y_tr)
    tx_tr, tx_te = transform(xc_tr[:, :-1], state), transform(xc_te[:, :-1], state)
    if flags_tr is not None:
        mean, std = fit_standardizer(flags_tr)
        tx_tr = np.hstack([tx_tr, standardize(flags_tr, mean, std)])
        tx_te = np.hstack([tx_te, standardize(flags_te, mean, std)])
    s_mlp = mlp.score(mlp.train(y_tr, tx_tr), tx_te)
    return 0.5 * s_gbm + 0.5 * s_mlp


def main():
    """Load data, train both models, average their logits, write the csv."""
    args = parse_args()
    t0 = time.time()

    # 1. Load and clean the data (labels in {-1, 1}).
    x_train, x_test, y_train, _, test_ids = load_data_cached(args.data_path)
    names = load_feature_names(args.data_path)
    xc_train, xc_names, leaky_train = model_inputs(x_train, names, args.with_leakage)
    xc_test, _, leaky_test = model_inputs(x_test, names, args.with_leakage)
    del x_train, x_test
    log(t0, f"Cleaned data: train {xc_train.shape}, test {xc_test.shape}")

    # 2. Train the GBM and the MLP on all training rows, score the test rows.
    scores = ensemble_scores(
        y_train, xc_train, xc_test, xc_names, leaky_train, leaky_test
    )
    log(t0, "GBM and MLP trained")

    # 3. Predict +1 for the top POSITIVE_RATE of the averaged logits
    #    (LEAKY_POSITIVE_RATE with --with_leakage).
    rate = LEAKY_POSITIVE_RATE if args.with_leakage else POSITIVE_RATE
    y_pred = predict_labels(scores, np.quantile(scores, 1 - rate))
    log(t0, f"Predicted positives: {np.mean(y_pred == 1):.2%}")

    # 4. Write the submission csv (Id, Prediction).
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    create_csv_submission(test_ids, y_pred, args.output)
    log(t0, f"Submission written to {args.output}")


if __name__ == "__main__":
    main()
