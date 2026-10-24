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

Usage (the default paths are relative to this file):
    python run.py [--data_path data/dataset] [--output submissions/submission.csv]
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
    load_data_cached,
    load_feature_names,
    transform,
)

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIVE_RATE = 0.14  # share of test rows predicted +1
MLP_CONFIG = {"interactions": False}  # preprocessing options of the MLP input


def parse_args():
    """Parse the command line arguments (data_path, output)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=os.path.join(HERE, "data", "dataset"))
    parser.add_argument(
        "--output", default=os.path.join(HERE, "submissions", "submission.csv")
    )
    return parser.parse_args()


def log(t0, message):
    """Print a progress message with the elapsed time."""
    print(f"[{time.time() - t0:6.0f}s] {message}", flush=True)


def main():
    """Load data, train both models, average their logits, write the csv."""
    args = parse_args()
    t0 = time.time()

    # 1. Load and clean the data (labels in {-1, 1}). The raw state code
    # _STATE is kept as the last column for the trees.
    x_train, x_test, y_train, _, test_ids = load_data_cached(args.data_path)
    names = load_feature_names(args.data_path)
    xc_train, xc_names = clean_data(x_train, names, keep_raw=("_STATE",))
    xc_test, _ = clean_data(x_test, names, keep_raw=("_STATE",))
    del x_train, x_test
    log(t0, f"Cleaned data: train {xc_train.shape}, test {xc_test.shape}")

    # 2. Gradient boosted trees on the cleaned matrix.
    model = gbm.train(y_train, gbm.tree_features(xc_train, xc_names))
    gbm_test = gbm.score(model, gbm.tree_features(xc_test, xc_names))
    log(t0, f"GBM trained: {len(model['trees'])} trees")

    # 3. Neural networks on the pipeline matrix (fitted on the training rows;
    # without its last column xc is the default clean_data output).
    state = fit_preprocessor(xc_train[:, :-1], xc_names[:-1], MLP_CONFIG, y=y_train)
    model = mlp.train(y_train, transform(xc_train[:, :-1], state))
    mlp_test = mlp.score(model, transform(xc_test[:, :-1], state))
    log(t0, f"MLP trained: {len(model['nets'])} networks")

    # 4. Average the two logits and predict +1 for the top POSITIVE_RATE.
    scores = 0.5 * gbm_test + 0.5 * mlp_test
    y_pred = predict_labels(scores, np.quantile(scores, 1 - POSITIVE_RATE))
    log(t0, f"Predicted positives: {np.mean(y_pred == 1):.2%}")

    # 5. Write the submission csv (Id, Prediction).
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    create_csv_submission(test_ids, y_pred, args.output)
    log(t0, f"Submission written to {args.output}")


if __name__ == "__main__":
    main()
