"""Milestone 6: 5-Fold Stratified Cross-Validation Benchmark for Super Tri-Blend.

Evaluates the 223-dimensional feature space combining:
- 205 Base Streamlined features
- 8 Extended Empirical Bayesian WoE features
- 10 Clinical Subgroup Bias-Correcting terms

Models evaluated in the Super Tri-Blend:
- Model 1: Cost-Sensitive Weighted Ridge (lambda=1e-4, w_pos=3.0, weight=0.35)
- Model 2: Balanced Logistic Bagging (M=15 bags, lambda=1e-4, gamma=0.3, max_iters=200, weight=0.35)
- Model 3: Differentiable Soft-F1 Optimizer (Adam lr=0.03, max_iters=250, weight=0.30)
"""

import json
import os
import time
import numpy as np

from cross_validation import build_k_indices, weighted_ridge_regression
from feature_engineering import build_advanced_features, train_soft_f1_optimizer
from implementations import reg_logistic_regression, sigmoid
from metrics import (
    accuracy,
    best_threshold,
    confusion_matrix,
    precision_recall_f1,
    predict_labels,
)

SEED = 42


def rank_normalize(scores):
    """Map continuous predictions to uniform percentile ranks in [0, 1]."""
    return np.argsort(np.argsort(scores)) / float(len(scores))


def run_benchmark():
    print("=" * 115)
    print("Milestone 6: Super Tri-Blend 5-Fold Stratified Cross-Validation Benchmark")
    print("=" * 115)

    with open("data/processed/features_selected.json") as f:
        feat_names = json.load(f)

    x = np.load("data/processed/x_train.npy")
    y = np.load("data/processed/y_train.npy")
    k_indices = build_k_indices(y, k_fold=5, seed=SEED)

    print(f"Dataset: {x.shape[0]} samples, {len(feat_names)} raw features.")
    print("Evaluating 5 stratified folds with zero data leakage...\n")

    acc_list = []
    prec_list = []
    rec_list = []
    bacc_list = []
    f1_list = []
    tau_list = []
    fold_times = []

    m1_f1_list = []
    m2_f1_list = []
    m3_f1_list = []

    t_start = time.time()

    for k in range(5):
        t_fold = time.time()
        val_idx = k_indices[k]
        tr_idx = np.concatenate([k_indices[i] for i in range(5) if i != k])
        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        # 1. Advanced Feature Engineering (223 dimensions)
        tx_tr, tx_val, feat_out = build_advanced_features(
            x_tr, x_val, feature_names=feat_names, y_tr=y_tr, alpha=20.0
        )
        assert tx_tr.shape[1] == 223, f"Expected 223 dims, got {tx_tr.shape[1]}"

        # 2. Model 1: Cost-Sensitive Weighted Ridge (w_pos=3.0, lambda=1e-4)
        t0 = time.time()
        w_ridge, _ = weighted_ridge_regression(
            y_tr.astype(float), tx_tr, lambda_=1e-4, w_pos=3.0
        )
        s1_tr = tx_tr @ w_ridge
        s1_val = tx_val @ w_ridge
        t_m1 = time.time() - t0

        # Model 1 individual F1
        tau1, _ = best_threshold(y_tr, s1_tr, np.linspace(-1.0, 1.0, 101))
        _, _, f1_m1 = precision_recall_f1(y_val, predict_labels(s1_val, tau1))
        m1_f1_list.append(f1_m1)

        # 3. Model 2: Balanced Logistic Bagging (M=15 models, max_iters=200, gamma=0.3)
        t0 = time.time()
        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == -1)[0]
        n_pos = len(pos_idx)
        rng = np.random.default_rng(SEED + k * 100)

        bag_tr = np.zeros(len(tx_tr))
        bag_val = np.zeros(len(tx_val))
        n_bags = 15
        for m in range(n_bags):
            sub_neg = rng.choice(neg_idx, size=n_pos, replace=False)
            sub_idx = np.concatenate([pos_idx, sub_neg])
            y_sub_01 = (y_tr[sub_idx] == 1).astype(float)
            w0 = np.zeros(tx_tr.shape[1])
            w_bag, _ = reg_logistic_regression(
                y_sub_01,
                tx_tr[sub_idx],
                lambda_=1e-4,
                initial_w=w0,
                max_iters=200,
                gamma=0.3,
            )
            bag_tr += sigmoid(tx_tr @ w_bag)
            bag_val += sigmoid(tx_val @ w_bag)
        s2_tr = bag_tr / float(n_bags)
        s2_val = bag_val / float(n_bags)
        t_m2 = time.time() - t0

        # Model 2 individual F1
        tau2, _ = best_threshold(y_tr, s2_tr, np.linspace(0.1, 0.9, 81))
        _, _, f1_m2 = precision_recall_f1(y_val, predict_labels(s2_val, tau2))
        m2_f1_list.append(f1_m2)

        # 4. Model 3: Differentiable Soft-F1 Optimizer via Adam
        t0 = time.time()
        y_tr_01 = (y_tr == 1).astype(float)
        w_soft, _ = train_soft_f1_optimizer(
            tx_tr, y_tr_01, max_iters=250, lr=0.03, lambda_reg=1e-5
        )
        s3_tr = sigmoid(tx_tr @ w_soft)
        s3_val = sigmoid(tx_val @ w_soft)
        t_m3 = time.time() - t0

        # Model 3 individual F1
        tau3, _ = best_threshold(y_tr, s3_tr, np.linspace(0.1, 0.9, 81))
        _, _, f1_m3 = precision_recall_f1(y_val, predict_labels(s3_val, tau3))
        m3_f1_list.append(f1_m3)

        # 5. Percentile Rank Normalization & Tri-Blend
        r1_tr, r1_val = rank_normalize(s1_tr), rank_normalize(s1_val)
        r2_tr, r2_val = rank_normalize(s2_tr), rank_normalize(s2_val)
        r3_tr, r3_val = rank_normalize(s3_tr), rank_normalize(s3_val)

        blend_tr = 0.35 * r1_tr + 0.35 * r2_tr + 0.30 * r3_tr
        blend_val = 0.35 * r1_val + 0.35 * r2_val + 0.30 * r3_val

        # 6. Threshold Calibration strictly on training fold
        tau, train_f1 = best_threshold(y_tr, blend_tr, np.linspace(0.65, 0.95, 61))
        tau_list.append(tau)

        # 7. Validation Evaluation
        pred_val = predict_labels(blend_val, tau)
        acc = accuracy(y_val, pred_val)
        p, r, f1 = precision_recall_f1(y_val, pred_val)
        cm = confusion_matrix(y_val, pred_val)
        bacc = cm["balanced_accuracy"]

        acc_list.append(acc)
        prec_list.append(p)
        rec_list.append(r)
        bacc_list.append(bacc)
        f1_list.append(f1)
        elapsed_fold = time.time() - t_fold
        fold_times.append(elapsed_fold)

        print(
            f"Fold {k+1}/5: Val F1={f1:.4f} (Prec={p:.4f}, Rec={r:.4f}, BAcc={bacc:.4f}, Acc={acc:.4f}) | "
            f"tau*={tau:.4f} (Train F1={train_f1:.4f}) | Sub-models: M1={f1_m1:.4f}, M2={f1_m2:.4f}, M3={f1_m3:.4f} | "
            f"Time: {elapsed_fold:.1f}s (M1={t_m1:.1f}s, M2={t_m2:.1f}s, M3={t_m3:.1f}s)"
        )

    total_time = time.time() - t_start
    results = {
        "model": "Milestone 6 Super Tri-Blend (223 Dims)",
        "dims": 223,
        "blend_weights": [0.35, 0.35, 0.30],
        "accuracy_mean": float(np.mean(acc_list)),
        "accuracy_std": float(np.std(acc_list)),
        "precision_mean": float(np.mean(prec_list)),
        "precision_std": float(np.std(prec_list)),
        "recall_mean": float(np.mean(rec_list)),
        "recall_std": float(np.std(rec_list)),
        "balanced_accuracy_mean": float(np.mean(bacc_list)),
        "balanced_accuracy_std": float(np.std(bacc_list)),
        "f1_mean": float(np.mean(f1_list)),
        "f1_std": float(np.std(f1_list)),
        "threshold_mean": float(np.mean(tau_list)),
        "m1_weighted_ridge_f1": float(np.mean(m1_f1_list)),
        "m2_balanced_logistic_f1": float(np.mean(m2_f1_list)),
        "m3_soft_f1_f1": float(np.mean(m3_f1_list)),
        "total_time_seconds": float(total_time),
    }

    print("\n" + "=" * 115)
    print("Super Tri-Blend 5-Fold Stratified CV Summary:")
    print(f"  Val F1-Score:      {results['f1_mean']:.4f} +/- {results['f1_std']:.4f}")
    print(
        f"  Val Precision:     {results['precision_mean']:.4f} +/- {results['precision_std']:.4f}"
    )
    print(
        f"  Val Recall:        {results['recall_mean']:.4f} +/- {results['recall_std']:.4f}"
    )
    print(
        f"  Balanced Accuracy: {results['balanced_accuracy_mean']:.4f} +/- {results['balanced_accuracy_std']:.4f}"
    )
    print(
        f"  Val Accuracy:      {results['accuracy_mean']:.4f} +/- {results['accuracy_std']:.4f}"
    )
    print(f"  Threshold tau*:    {results['threshold_mean']:.4f}")
    print(
        f"  Sub-models:        M1 (Ridge)={results['m1_weighted_ridge_f1']:.4f}, M2 (Bagging)={results['m2_balanced_logistic_f1']:.4f}, M3 (Soft-F1)={results['m3_soft_f1_f1']:.4f}"
    )
    print(f"  Total CV Time:     {total_time:.1f}s")
    print("=" * 115)

    os.makedirs("report", exist_ok=True)
    with open("report/m6_super_blend_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved report/m6_super_blend_benchmark.json successfully.\n")

    return results


if __name__ == "__main__":
    run_benchmark()
