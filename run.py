"""Reproduce our best AIcrowd submission from scratch.

STATUS: baseline (final preprocessing, model not tuned yet). When finished,
`python run.py` must write EXACTLY the same csv we submitted for our best
AIcrowd score (fix every random seed).

Usage:
    python run.py --data_path data/dataset --output submissions/submission.csv
"""

import argparse

import numpy as np

from helpers import create_csv_submission
from implementations import reg_logistic_regression, sigmoid
from metrics import best_threshold, predict_labels
from preprocessing import load_data_cached, load_feature_names, preprocess

SEED = 42

# Final hyperparameters, chosen by cross-validation (see notebooks/ and the
# report). Keys not listed here take the values of
# preprocessing.DEFAULT_CONFIG. Update them together with the report.
CONFIG = {
    "lambda_": 1e-4,
    "gamma": 0.5,
    "max_iters": 1000,
}


def parse_args():
    """Parse the command line arguments.

    Returns:
        argparse.Namespace with data_path and output.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="data/dataset")
    parser.add_argument("--output", default="submissions/submission.csv")
    return parser.parse_args()


def main():
    """Load data, preprocess, train the final model and write the submission."""
    args = parse_args()
    np.random.seed(SEED)

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
    create_csv_submission(test_ids, y_pred, args.output)
    print(f"Submission written to {args.output}")


if __name__ == "__main__":
    main()
