"""Reproduce our best AIcrowd submission from scratch.

STATUS: baseline (final preprocessing, model not tuned yet). `python run.py`
must write exactly the csv we submitted. No step is random: gradient descent
starts from zeros and the threshold is chosen on the training scores.

Usage (the default paths are relative to this file):
    python run.py [--data_path data/dataset] [--output submissions/submission.csv]
"""

import argparse
import os

import numpy as np

from helpers import create_csv_submission
from implementations import reg_logistic_regression, sigmoid
from metrics import best_threshold, predict_labels
from preprocessing import (
    DEFAULT_CONFIG,
    load_data_cached,
    load_feature_names,
    preprocess,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# Final options and hyperparameters, chosen by cross-validation (see the
# report): the single source of truth is preprocessing.DEFAULT_CONFIG.
CONFIG = DEFAULT_CONFIG


def parse_args():
    """Parse the command line arguments (data_path, output)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=os.path.join(HERE, "data", "dataset"))
    parser.add_argument(
        "--output", default=os.path.join(HERE, "submissions", "submission.csv")
    )
    return parser.parse_args()


def main():
    """Load data, preprocess, train the final model and write the submission."""
    args = parse_args()

    # 1. Load data (labels in {-1, 1}).
    x_train, x_test, y_train, _, test_ids = load_data_cached(args.data_path)
    feature_names = load_feature_names(args.data_path)

    # 2. Preprocess: fit on train, apply to train and test. The labels are
    # needed to choose the missing indicators, as inside cross-validation.
    tx_train, tx_test = preprocess(
        x_train, x_test, feature_names, CONFIG, y_train=y_train
    )
    print(f"Design matrix: train {tx_train.shape}, test {tx_test.shape}")

    # 3. Train the final model (logistic regression needs labels in {0, 1}).
    y01 = (y_train + 1) / 2
    initial_w = np.zeros(tx_train.shape[1])
    w, loss = reg_logistic_regression(
        y01,
        tx_train,
        CONFIG["lambda_"],
        initial_w,
        CONFIG["max_iters"],
        CONFIG["gamma"],
    )
    print(f"Final training loss: {loss:.5f}")

    # 4. Choose the threshold that maximizes F1 on the training scores (as in
    # cross-validation), then predict labels in {-1, 1}.
    threshold, f1_train = best_threshold(y_train, sigmoid(tx_train @ w))
    print(f"Threshold {threshold:.3f} (training F1 {f1_train:.4f})")
    y_pred = predict_labels(sigmoid(tx_test @ w), threshold)

    # 5. Write the submission csv (Id, Prediction).
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    create_csv_submission(test_ids, y_pred, args.output)
    print(f"Submission written to {args.output}")


if __name__ == "__main__":
    main()
