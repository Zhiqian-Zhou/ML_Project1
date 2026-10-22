"""Reproduce our best AIcrowd submission from scratch.

Pipeline:
1. CDC survey code sanitation & missingness handling.
2. Informative missingness indicator masks (MNAR, 5% to 85%).
3. Nominal categorical one-hot encoding (MARITAL, EMPLOY1, _RACE).
4. Clinically grounded synergistic cross-terms (Framingham risk multiplication).
5. L2 Regularized Logistic Regression with calibrated threshold tau*.

Usage:
    python run.py --data_path data/dataset --output submissions/submission_feature_engineered.csv
"""

import argparse
import json
import os

import numpy as np

from helpers import create_csv_submission
from implementations import reg_logistic_regression
from metrics import best_threshold, predict_labels
from preprocessing import load_data_cached, load_feature_names, preprocess

SEED = 42

# Final hyperparameters, validated through 5-Fold Stratified Cross-Validation
CONFIG = {
    "max_nan_ratio": 0.85,
    "add_nan_indicators": True,
    "one_hot_nominal": True,
    "clinical_interactions": True,
    "drop_first_nominal": True,
    "poly_degree": 1,
    "lambda_": 1e-4,
    "gamma": 0.15,
    "max_iters": 400,
    "threshold": 0.210,
}


def parse_args():
    """Parse the command line arguments.

    Returns:
        argparse.Namespace with data_path and output.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="data/dataset")
    parser.add_argument(
        "--output", default="submissions/submission_feature_engineered.csv"
    )
    return parser.parse_args()


def main():
    """Load data, preprocess, train the final model and write the submission."""
    args = parse_args()
    np.random.seed(SEED)

    # Fast binary cache path detection
    repo_root = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(repo_root, "data", "processed")
    cache_xtr = os.path.join(processed_dir, "x_train.npy")
    cache_xte = os.path.join(processed_dir, "x_test.npy")
    cache_ytr = os.path.join(processed_dir, "y_train.npy")
    cache_te_id = os.path.join(processed_dir, "test_ids.npy")
    cache_feat = os.path.join(processed_dir, "features_selected.json")

    if (
        os.path.exists(cache_xtr)
        and os.path.exists(cache_xte)
        and os.path.exists(cache_ytr)
        and os.path.exists(cache_te_id)
        and os.path.exists(cache_feat)
    ):
        print(f"Loading pre-sanitized dataset from cache ({processed_dir})...")
        x_train = np.load(cache_xtr)
        x_test = np.load(cache_xte)
        y_train = np.load(cache_ytr)
        test_ids = np.load(cache_te_id)
        with open(cache_feat, "r") as f:
            feature_names = json.load(f)
    else:
        print(f"Loading raw dataset from {args.data_path}...")
        x_train, x_test, y_train, _, test_ids = load_data_cached(args.data_path)
        feature_names = load_feature_names(args.data_path)

    print(
        f"Input dimensions: x_train={x_train.shape}, x_test={x_test.shape}, "
        f"positive ratio={np.mean(y_train == 1):.4%}"
    )

    # 2. Preprocess: fit feature engineering on train, apply to train and test
    print("Applying hypothesis-driven feature engineering pipeline...")
    tx_train, tx_test = preprocess(x_train, x_test, feature_names, CONFIG)
    print(
        f"Engineered design matrix: tx_train={tx_train.shape}, tx_test={tx_test.shape}"
    )

    # 3. Train final model with regularized logistic regression
    y01 = (y_train + 1) / 2
    initial_w = np.zeros(tx_train.shape[1])
    print(
        f"Training Regularized Logistic Regression (lambda={CONFIG['lambda_']}, "
        f"gamma={CONFIG['gamma']}, max_iters={CONFIG['max_iters']})..."
    )
    w, loss = reg_logistic_regression(
        y01,
        tx_train,
        CONFIG["lambda_"],
        initial_w,
        CONFIG["max_iters"],
        CONFIG["gamma"],
    )
    print(f"Final training loss: {loss:.5f}")

    # 4. Calibrate decision threshold tau* on training set
    logits_tr = np.clip(tx_train @ w, -250, 250)
    probs_tr = 1 / (1 + np.exp(-logits_tr))
    tau, train_f1 = best_threshold(y_train, probs_tr, np.linspace(0.05, 0.45, 81))
    print(f"Calibrated decision threshold: tau*={tau:.4f} (Train F1={train_f1:.4f})")

    # 5. Predict on test set
    logits_te = np.clip(tx_test @ w, -250, 250)
    probs_te = 1 / (1 + np.exp(-logits_te))
    y_pred = predict_labels(probs_te, tau)
    print(
        f"Test set predictions: total={len(y_pred)}, "
        f"predicted positive (+1)={np.sum(y_pred == 1)} ({np.mean(y_pred == 1):.2%}), "
        f"predicted negative (-1)={np.sum(y_pred == -1)} ({np.mean(y_pred == -1):.2%})"
    )

    # 6. Write submission CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    create_csv_submission(test_ids, y_pred, args.output)
    print(f"Successfully generated submission: {args.output}")


if __name__ == "__main__":
    main()
