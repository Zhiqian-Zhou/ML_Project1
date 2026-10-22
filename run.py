"""Reproduce our best AIcrowd submission from scratch.

Pipeline (SOTA Ensemble):
1. CDC survey code sanitation & leak-free median imputation.
2. 5 Clinically grounded synergistic cross-terms (Framingham epidemiology).
3. 2 Quadratic physiological risk curves (_BMI5^2, _AGEG5YR^2).
4. Empirical Bayesian Log-Odds Encoding (WoE) for exponential risk factors.
5. Multi-Model Heterogeneous Rank-Order Ensemble Blend:
   - Model 1: Cost-Sensitive Weighted Ridge Regression (w_pos = 3.0, weight = 0.45)
   - Model 2: Regularized Logistic Regression (lambda = 1e-4, weight = 0.35)
   - Model 3: Balanced Undersampling Bagging Ensemble (M = 10 bags, weight = 0.20)
6. Dynamic decision threshold calibration tau* on blended percentile ranks.

Usage:
    python run.py --data_path data/dataset --output submissions/submission_sota.csv
"""

import argparse
import json
import os
import time

import numpy as np

from cross_validation import weighted_ridge_regression
from feature_engineering import build_streamlined_features
from helpers import create_csv_submission
from implementations import reg_logistic_regression, ridge_regression, sigmoid
from metrics import best_threshold, predict_labels
from preprocessing import load_data_cached, load_feature_names

SEED = 42

CONFIG = {
    "lambda_": 1e-4,
    "w_pos": 3.0,
    "gamma": 0.15,
    "max_iters": 350,
    "n_bags": 10,
    "blend_weights": (0.45, 0.35, 0.20),
}


def rank_normalize(scores):
    """Map real-valued continuous predictions to uniform percentile ranks in [0, 1]."""
    return np.argsort(np.argsort(scores)) / float(len(scores))


def compute_woe_features(x_tr, x_te, feature_names, alpha=20.0, y_tr=None):
    """Compute empirical Bayesian Laplace-smoothed Log-Odds (WoE) on training set and apply to test."""
    woe_cols = ["_AGEG5YR", "GENHLTH", "_SMOKER3", "INCOME2"]
    woe_indices = [feature_names.index(c) for c in woe_cols if c in feature_names]

    pos_p = np.mean(y_tr == 1)
    neg_p = 1.0 - pos_p
    prior_lo = np.log(pos_p / neg_p)

    woe_tr_list, woe_te_list = [], []
    for idx in woe_indices:
        col_tr = x_tr[:, idx]
        col_te = x_te[:, idx]
        cats = np.unique(col_tr[~np.isnan(col_tr)])
        mapping = {}
        for c in cats:
            m_c = col_tr == c
            n_pos = np.sum(y_tr[m_c] == 1)
            n_neg = np.sum(y_tr[m_c] == -1)
            mapping[c] = (
                np.log((n_pos + alpha * pos_p) / (n_neg + alpha * neg_p)) - prior_lo
            )

        woe_tr_list.append(np.array([mapping.get(v, 0.0) for v in col_tr]))
        woe_te_list.append(np.array([mapping.get(v, 0.0) for v in col_te]))

    w_mat_tr = np.column_stack(woe_tr_list)
    w_mat_te = np.column_stack(woe_te_list)

    w_mean = np.mean(w_mat_tr, axis=0)
    w_std = np.std(w_mat_tr, axis=0)
    w_std[w_std == 0.0] = 1.0

    return (w_mat_tr - w_mean) / w_std, (w_mat_te - w_mean) / w_std


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="data/dataset")
    parser.add_argument("--output", default="submissions/submission_sota.csv")
    parser.add_argument(
        "--model",
        choices=["sota_ensemble", "weighted_ridge", "logistic"],
        default="sota_ensemble",
        help="Model architecture: 'sota_ensemble' (default best), 'weighted_ridge', or 'logistic'.",
    )
    return parser.parse_args()


def main():
    """Load data, preprocess, train the final model and write the submission."""
    args = parse_args()
    np.random.seed(SEED)

    # 1. Fast binary cache path detection
    repo_root = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(repo_root, "data", "processed")
    cache_xtr = os.path.join(processed_dir, "x_train.npy")
    cache_xte = os.path.join(processed_dir, "x_test.npy")
    cache_ytr = os.path.join(processed_dir, "y_train.npy")
    cache_te_id = os.path.join(processed_dir, "test_ids.npy")
    cache_feat = os.path.join(processed_dir, "features_selected.json")

    t_load = time.time()
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
        f"positive ratio={np.mean(y_train == 1):.4%} (loaded in {time.time() - t_load:.2f}s)"
    )

    # 2. Build streamlined 205-dimensional design matrix
    print("Building streamlined high-signal design matrix...")
    tx_train, tx_test, _ = build_streamlined_features(x_train, x_test, feature_names)

    # 3. Append empirical Bayes WoE log-odds features
    print("Computing out-of-fold empirical Bayesian WoE log-odds features...")
    woe_tr, woe_te = compute_woe_features(
        x_train, x_test, feature_names, alpha=20.0, y_tr=y_train
    )
    tx_train = np.c_[tx_train, woe_tr]
    tx_test = np.c_[tx_test, woe_te]
    print(
        f"Final design matrix shape: tx_train={tx_train.shape}, tx_test={tx_test.shape}"
    )

    # 4. Model Training & Prediction
    t_train = time.time()
    if args.model == "sota_ensemble":
        print(
            f"Training SOTA Heterogeneous Ensemble Blend ({CONFIG['blend_weights'][0]:.2f} Ridge + "
            f"{CONFIG['blend_weights'][1]:.2f} Logistic + {CONFIG['blend_weights'][2]:.2f} Bagging)..."
        )
        # Model 1: Cost-Sensitive Weighted Ridge
        w_ridge, _ = weighted_ridge_regression(
            y_train.astype(float),
            tx_train,
            lambda_=CONFIG["lambda_"],
            w_pos=CONFIG["w_pos"],
        )
        s1_tr = tx_train @ w_ridge
        s1_te = tx_test @ w_ridge

        # Model 2: Regularized Logistic Regression
        y01 = (y_train + 1) / 2
        w0 = np.zeros(tx_train.shape[1])
        w_log, _ = reg_logistic_regression(
            y01,
            tx_train,
            CONFIG["lambda_"],
            initial_w=w0,
            max_iters=CONFIG["max_iters"],
            gamma=CONFIG["gamma"],
        )
        s2_tr = sigmoid(tx_train @ w_log)
        s2_te = sigmoid(tx_test @ w_log)

        # Model 3: Balanced Undersampling Bagging Ensemble
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == -1)[0]
        n_pos = len(pos_idx)
        rng = np.random.default_rng(SEED)

        bag_tr = np.zeros(len(tx_train))
        bag_te = np.zeros(len(tx_test))
        for m in range(CONFIG["n_bags"]):
            sub_neg = rng.choice(neg_idx, size=n_pos, replace=False)
            sub_idx = np.concatenate([pos_idx, sub_neg])
            w_bag, _ = ridge_regression(
                y_train[sub_idx].astype(float),
                tx_train[sub_idx],
                lambda_=CONFIG["lambda_"],
            )
            bag_tr += tx_train @ w_bag
            bag_te += tx_test @ w_bag
        s3_tr = bag_tr / float(CONFIG["n_bags"])
        s3_te = bag_te / float(CONFIG["n_bags"])

        # Rank-Order Percentile Normalization
        r1_tr, r1_te = rank_normalize(s1_tr), rank_normalize(s1_te)
        r2_tr, r2_te = rank_normalize(s2_tr), rank_normalize(s2_te)
        r3_tr, r3_te = rank_normalize(s3_tr), rank_normalize(s3_te)

        w1, w2, w3 = CONFIG["blend_weights"]
        blend_tr = w1 * r1_tr + w2 * r2_tr + w3 * r3_tr
        blend_te = w1 * r1_te + w2 * r2_te + w3 * r3_te

        tau, train_f1 = best_threshold(y_train, blend_tr, np.linspace(0.65, 0.95, 61))
        print(
            f"Ensemble calibrated decision threshold: tau*={tau:.4f} (Train F1={train_f1:.4f}, "
            f"trained in {time.time() - t_train:.2f}s)"
        )
        y_pred = predict_labels(blend_te, tau)

    elif args.model == "weighted_ridge":
        print(
            f"Training Cost-Sensitive Weighted Ridge (lambda={CONFIG['lambda_']}, w_pos={CONFIG['w_pos']})..."
        )
        w, loss = weighted_ridge_regression(
            y_train.astype(float),
            tx_train,
            lambda_=CONFIG["lambda_"],
            w_pos=CONFIG["w_pos"],
        )
        scores_tr = tx_train @ w
        tau, train_f1 = best_threshold(y_train, scores_tr, np.linspace(-1.0, 1.0, 201))
        print(
            f"Calibrated decision threshold: tau*={tau:.4f} (Train F1={train_f1:.4f})"
        )
        y_pred = predict_labels(tx_test @ w, tau)

    else:
        print(
            f"Training Regularized Logistic Regression (lambda={CONFIG['lambda_']}, gamma={CONFIG['gamma']})..."
        )
        y01 = (y_train + 1) / 2
        w0 = np.zeros(tx_train.shape[1])
        w, loss = reg_logistic_regression(
            y01,
            tx_train,
            CONFIG["lambda_"],
            initial_w=w0,
            max_iters=CONFIG["max_iters"],
            gamma=CONFIG["gamma"],
        )
        probs_tr = sigmoid(tx_train @ w)
        tau, train_f1 = best_threshold(y_train, probs_tr, np.linspace(0.05, 0.45, 81))
        print(
            f"Calibrated decision threshold: tau*={tau:.4f} (Train F1={train_f1:.4f})"
        )
        y_pred = predict_labels(sigmoid(tx_test @ w), tau)

    # 5. Output Diagnostics & Submission File Write
    print(
        f"Test set predictions: total={len(y_pred)}, "
        f"predicted positive (+1)={np.sum(y_pred == 1)} ({np.mean(y_pred == 1):.2%}), "
        f"predicted negative (-1)={np.sum(y_pred == -1)} ({np.mean(y_pred == -1):.2%})"
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    create_csv_submission(test_ids, y_pred, args.output)
    print(f"Successfully generated submission: {args.output}")


if __name__ == "__main__":
    main()
