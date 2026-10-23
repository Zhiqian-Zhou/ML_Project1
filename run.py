"""Reproduce our best AIcrowd submission from scratch.

Pipeline (Milestone 6.5 SOTA Super Tri-Blend):
1. CDC survey code sanitation & leak-free median imputation.
2. 5 Clinically grounded synergistic cross-terms (Framingham epidemiology) + 2 quadratics.
3. Extended Empirical Bayesian Log-Odds Encoding (WoE) across 8 key demographic/clinical features.
4. 10 Clinical subgroup bias-correcting and risk-linearizing terms (223 dimensions total).
5. Super Tri-Blend of 3 heterogeneous sub-models with Simplex-Optimized Weights:
   - Model 1: Cost-Sensitive Weighted Ridge (lambda = 1e-4, w_pos = 3.0, weight = 0.35)
   - Model 2: Stratified Hard-Negative Balanced Logistic Bagging (M = 15 models, 50% hard + 50% easy negatives, weight = 0.40)
   - Model 3: Differentiable Soft-F1 Metric Alignment via Adam (max_iters = 250, lr = 0.03, weight = 0.25)
6. Dynamic decision threshold calibration tau* on blended uniform percentile ranks.

Usage:
    python run.py --data_path data/dataset --output submissions/submission_sota_m6_5.csv
"""

import argparse
import json
import os
import time

import numpy as np

from cross_validation import weighted_ridge_regression
from feature_engineering import build_advanced_features, train_soft_f1_optimizer
from helpers import create_csv_submission
from implementations import reg_logistic_regression, ridge_regression, sigmoid
from metrics import best_threshold, predict_labels
from preprocessing import load_data_cached, load_feature_names

SEED = 42

CONFIG = {
    "lambda_": 1e-4,
    "w_pos": 3.0,
    "gamma_logistic": 0.30,
    "max_iters_logistic": 200,
    "n_bags": 15,
    "lr_soft_f1": 0.03,
    "max_iters_soft_f1": 250,
    "blend_weights": (0.35, 0.40, 0.25),
}


def rank_normalize(scores):
    """Map real-valued continuous predictions to uniform percentile ranks in [0, 1]."""
    return np.argsort(np.argsort(scores)) / float(len(scores))


def compute_clinical_risk_scores(x_arr, feature_names):
    """Compute composite clinical vulnerability score (0 to 7) for negative stratification."""
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    age = x_arr[:, name_to_idx["_AGEG5YR"]]
    hbp = x_arr[:, name_to_idx["_RFHYPE5"]]
    chol = x_arr[:, name_to_idx["_RFCHOL"]]
    smoker = x_arr[:, name_to_idx["_SMOKER3"]]
    diabete = x_arr[:, name_to_idx["DIABETE3"]]
    stroke = x_arr[:, name_to_idx["CVDSTRK3"]]
    genhlth = x_arr[:, name_to_idx["GENHLTH"]]

    risk_score = (
        (age >= 10.0).astype(int)
        + (hbp == 2.0).astype(int)
        + (chol == 2.0).astype(int)
        + ((smoker == 1.0) | (smoker == 2.0)).astype(int)
        + (diabete == 1.0).astype(int)
        + (stroke == 1.0).astype(int)
        + (genhlth >= 4.0).astype(int)
    )
    return risk_score


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="data/dataset")
    parser.add_argument("--output", default="submissions/submission_sota_m6_5.csv")
    parser.add_argument(
        "--model",
        choices=["super_blend", "weighted_ridge", "logistic"],
        default="super_blend",
        help="Model architecture: 'super_blend' (default best M6.5), 'weighted_ridge', or 'logistic'.",
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

    # 2. Build rich, clinical 223-dimensional design matrix
    print("Building advanced 223-dimensional clinical design matrix...")
    t_feat = time.time()
    tx_train, tx_test, feature_out_names = build_advanced_features(
        x_train, x_test, feature_names=feature_names, y_tr=y_train, alpha=20.0
    )
    print(
        f"Design matrix constructed: tx_train={tx_train.shape}, tx_test={tx_test.shape} "
        f"({len(feature_out_names)} features, computed in {time.time() - t_feat:.2f}s)"
    )

    # 3. Model Training & Prediction
    t_train = time.time()
    if args.model == "super_blend":
        w1, w2, w3 = CONFIG["blend_weights"]
        print(
            f"Training Milestone 6.5 Super Tri-Blend ({w1:.2f} Weighted Ridge + "
            f"{w2:.2f} Hard-Negative Logistic Bagging + {w3:.2f} Soft-F1 Optimizer)..."
        )

        # Model 1: Cost-Sensitive Weighted Ridge
        print("  [1/3] Training Cost-Sensitive Weighted Ridge (w_pos=3.0)...")
        t0 = time.time()
        w_ridge, _ = weighted_ridge_regression(
            y_train.astype(float),
            tx_train,
            lambda_=CONFIG["lambda_"],
            w_pos=CONFIG["w_pos"],
        )
        s1_tr = tx_train @ w_ridge
        s1_te = tx_test @ w_ridge
        print(f"        M1 finished in {time.time() - t0:.2f}s")

        # Model 2: Stratified Hard-Negative Balanced Logistic Bagging
        print(
            f"  [2/3] Training Stratified Hard-Negative Logistic Bagging (M={CONFIG['n_bags']} bags, max_iters={CONFIG['max_iters_logistic']})..."
        )
        t0 = time.time()
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == -1)[0]
        n_pos = len(pos_idx)

        risk_scores_tr = compute_clinical_risk_scores(x_train, feature_names)
        hard_neg_idx = neg_idx[risk_scores_tr[neg_idx] >= 2]
        easy_neg_idx = neg_idx[risk_scores_tr[neg_idx] <= 1]
        print(
            f"        Clinical negative stratification: {len(hard_neg_idx)} hard negatives (score >= 2), "
            f"{len(easy_neg_idx)} easy negatives (score <= 1)"
        )

        rng = np.random.default_rng(SEED)
        bag_tr = np.zeros(len(tx_train))
        bag_te = np.zeros(len(tx_test))
        n_half = n_pos // 2

        for m in range(CONFIG["n_bags"]):
            sub_hard = rng.choice(hard_neg_idx, size=n_half, replace=False)
            sub_easy = rng.choice(easy_neg_idx, size=n_pos - n_half, replace=False)
            sub_idx = np.concatenate([pos_idx, sub_hard, sub_easy])
            y_sub_01 = (y_train[sub_idx] == 1).astype(float)
            w0 = np.zeros(tx_train.shape[1])
            w_bag, _ = reg_logistic_regression(
                y_sub_01,
                tx_train[sub_idx],
                lambda_=CONFIG["lambda_"],
                initial_w=w0,
                max_iters=CONFIG["max_iters_logistic"],
                gamma=CONFIG["gamma_logistic"],
            )
            bag_tr += sigmoid(tx_train @ w_bag)
            bag_te += sigmoid(tx_test @ w_bag)
        s2_tr = bag_tr / float(CONFIG["n_bags"])
        s2_te = bag_te / float(CONFIG["n_bags"])
        print(f"        M2 finished in {time.time() - t0:.2f}s")

        # Model 3: Differentiable Soft-F1 Metric Alignment via Adam
        print(
            f"  [3/3] Training Differentiable Soft-F1 Optimizer (Adam lr={CONFIG['lr_soft_f1']}, max_iters={CONFIG['max_iters_soft_f1']})..."
        )
        t0 = time.time()
        y01 = (y_train == 1).astype(float)
        w_soft, _ = train_soft_f1_optimizer(
            tx_train,
            y01,
            max_iters=CONFIG["max_iters_soft_f1"],
            lr=CONFIG["lr_soft_f1"],
            lambda_reg=1e-5,
        )
        s3_tr = sigmoid(tx_train @ w_soft)
        s3_te = sigmoid(tx_test @ w_soft)
        print(f"        M3 finished in {time.time() - t0:.2f}s")

        # Percentile Rank-Order Normalization
        print("Normalizing sub-model scores to uniform percentile ranks...")
        r1_tr, r1_te = rank_normalize(s1_tr), rank_normalize(s1_te)
        r2_tr, r2_te = rank_normalize(s2_tr), rank_normalize(s2_te)
        r3_tr, r3_te = rank_normalize(s3_tr), rank_normalize(s3_te)

        blend_tr = w1 * r1_tr + w2 * r2_tr + w3 * r3_tr
        blend_te = w1 * r1_te + w2 * r2_te + w3 * r3_te

        tau, train_f1 = best_threshold(y_train, blend_tr, np.linspace(0.65, 0.95, 61))
        print(
            f"Ensemble calibrated decision threshold: tau*={tau:.4f} (Train F1={train_f1:.4f}, "
            f"all models trained in {time.time() - t_train:.2f}s)"
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
            f"Training Regularized Logistic Regression (lambda={CONFIG['lambda_']}, gamma={CONFIG['gamma_logistic']})..."
        )
        y01 = (y_train + 1) / 2
        w0 = np.zeros(tx_train.shape[1])
        w, loss = reg_logistic_regression(
            y01,
            tx_train,
            CONFIG["lambda_"],
            initial_w=w0,
            max_iters=CONFIG["max_iters_logistic"],
            gamma=CONFIG["gamma_logistic"],
        )
        probs_tr = sigmoid(tx_train @ w)
        tau, train_f1 = best_threshold(y_train, probs_tr, np.linspace(0.05, 0.45, 81))
        print(
            f"Calibrated decision threshold: tau*={tau:.4f} (Train F1={train_f1:.4f})"
        )
        y_pred = predict_labels(sigmoid(tx_test @ w), tau)

    # 4. Output Diagnostics & Submission File Write
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
