import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Milestone 6.6: 5-Fold Stratified Cross-Validation Benchmark for 233-Dimensional Feature Space.

Compares:
1. Architecture A: Direct 233-dim Super Tri-Blend (0.35 Ridge + 0.40 Hard-Neg Bagging + 0.25 Soft-F1)
2. Architecture B: Two-Stage Cascade Screening
   - Stage 1 (Triage): Fast Ridge filters bottom 30% lowest-risk samples (< 30th percentile -> -1)
   - Stage 2 (Specialist): Specialist ensemble trained on the top 70% high-risk cohort (~13% positive prevalence)
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


def run_cascade_benchmark():
    print("=" * 115)
    print(
        "Milestone 6.6: 233-Dim Vascular Synergy & Two-Stage Cascade 5-Fold CV Benchmark"
    )
    print("=" * 115)

    with open("data/processed/features_selected.json") as f:
        feat_names = json.load(f)

    x = np.load("data/processed/x_train.npy")
    y = np.load("data/processed/y_train.npy")
    k_indices = build_k_indices(y, k_fold=5, seed=SEED)

    print(f"Dataset: {x.shape[0]} samples, {len(feat_names)} raw features.")
    print("Evaluating 5 stratified folds with zero data leakage...\n")

    # Metrics containers
    metrics_a = {"f1": [], "prec": [], "rec": [], "bacc": [], "acc": [], "tau": []}
    metrics_b = {"f1": [], "prec": [], "rec": [], "bacc": [], "acc": []}

    t_start = time.time()

    for k in range(5):
        t_fold = time.time()
        val_idx = k_indices[k]
        tr_idx = np.concatenate([k_indices[i] for i in range(5) if i != k])
        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        # 1. Feature Engineering (233 dimensions)
        tx_tr, tx_val, out_names = build_advanced_features(
            x_tr, x_val, feature_names=feat_names, y_tr=y_tr, alpha=20.0
        )
        assert tx_tr.shape[1] == 233, f"Expected 233 dims, got {tx_tr.shape[1]}"

        # =========================================================================
        # Architecture A: Direct 233-dim Super Tri-Blend (0.35 Ridge + 0.40 Bagging + 0.25 Soft-F1)
        # =========================================================================
        # Model 1: Cost-Sensitive Weighted Ridge
        w_ridge, _ = weighted_ridge_regression(
            y_tr.astype(float), tx_tr, lambda_=1e-4, w_pos=3.0
        )
        s1_tr = tx_tr @ w_ridge
        s1_val = tx_val @ w_ridge

        # Model 2: Stratified Hard-Negative Balanced Logistic Bagging (M=15)
        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == -1)[0]
        n_pos = len(pos_idx)

        risk_scores_tr = compute_clinical_risk_scores(x_tr, feat_names)
        hard_neg_idx = neg_idx[risk_scores_tr[neg_idx] >= 2]
        easy_neg_idx = neg_idx[risk_scores_tr[neg_idx] <= 1]

        rng = np.random.default_rng(SEED + k * 100)
        bag_tr = np.zeros(len(tx_tr))
        bag_val = np.zeros(len(tx_val))
        n_half = n_pos // 2

        for m in range(15):
            sub_hard = rng.choice(hard_neg_idx, size=n_half, replace=False)
            sub_easy = rng.choice(easy_neg_idx, size=n_pos - n_half, replace=False)
            sub_idx = np.concatenate([pos_idx, sub_hard, sub_easy])
            y_sub = (y_tr[sub_idx] == 1).astype(float)
            w0 = np.zeros(tx_tr.shape[1])
            w_bag, _ = reg_logistic_regression(
                y_sub,
                tx_tr[sub_idx],
                lambda_=1e-4,
                initial_w=w0,
                max_iters=200,
                gamma=0.3,
            )
            bag_tr += sigmoid(tx_tr @ w_bag)
            bag_val += sigmoid(tx_val @ w_bag)
        s2_tr = bag_tr / 15.0
        s2_val = bag_val / 15.0

        # Model 3: Differentiable Soft-F1 Optimizer
        y_tr_01 = (y_tr == 1).astype(float)
        w_soft, _ = train_soft_f1_optimizer(
            tx_tr, y_tr_01, max_iters=250, lr=0.03, lambda_reg=1e-5
        )
        s3_tr = sigmoid(tx_tr @ w_soft)
        s3_val = sigmoid(tx_val @ w_soft)

        # Percentile Rank Normalization & Blend A
        r1_tr, r1_val = rank_normalize(s1_tr), rank_normalize(s1_val)
        r2_tr, r2_val = rank_normalize(s2_tr), rank_normalize(s2_val)
        r3_tr, r3_val = rank_normalize(s3_tr), rank_normalize(s3_val)

        blend_a_tr = 0.35 * r1_tr + 0.40 * r2_tr + 0.25 * r3_tr
        blend_a_val = 0.35 * r1_val + 0.40 * r2_val + 0.25 * r3_val

        tau_a, tr_f1_a = best_threshold(y_tr, blend_a_tr, np.linspace(0.65, 0.95, 61))
        pred_a = predict_labels(blend_a_val, tau_a)
        acc_a = accuracy(y_val, pred_a)
        p_a, r_a, f1_a = precision_recall_f1(y_val, pred_a)
        bacc_a = confusion_matrix(y_val, pred_a)["balanced_accuracy"]

        metrics_a["f1"].append(f1_a)
        metrics_a["prec"].append(p_a)
        metrics_a["rec"].append(r_a)
        metrics_a["bacc"].append(bacc_a)
        metrics_a["acc"].append(acc_a)
        metrics_a["tau"].append(tau_a)

        # =========================================================================
        # Architecture B: Two-Stage Cascade Screening
        # =========================================================================
        # Stage 1 Triage Cutoff (30th percentile of Stage 1 Ridge scores)
        tau_triage = np.percentile(s1_tr, 30.0)
        ambig_tr_mask = s1_tr >= tau_triage
        ambig_val_mask = s1_val >= tau_triage
        tx_tr_spec = tx_tr[ambig_tr_mask]
        y_tr_spec = y_tr[ambig_tr_mask]

        # Stage 2 Specialist Model 1: Specialist Ridge
        w_spec_ridge, _ = weighted_ridge_regression(
            y_tr_spec.astype(float), tx_tr_spec, lambda_=1e-4, w_pos=2.5
        )
        s_spec1_tr = tx_tr_spec @ w_spec_ridge
        s_spec1_val_ambig = tx_val[ambig_val_mask] @ w_spec_ridge

        # Stage 2 Specialist Model 2: Specialist Bagging (M=10)
        pos_spec_idx = np.where(y_tr_spec == 1)[0]
        neg_spec_idx = np.where(y_tr_spec == -1)[0]
        n_pos_spec = len(pos_spec_idx)
        bag_spec_tr = np.zeros(len(tx_tr_spec))
        bag_spec_val_ambig = np.zeros(np.sum(ambig_val_mask))

        for m in range(10):
            sub_neg_spec = rng.choice(neg_spec_idx, size=n_pos_spec, replace=False)
            sub_idx_spec = np.concatenate([pos_spec_idx, sub_neg_spec])
            y_sub_spec = (y_tr_spec[sub_idx_spec] == 1).astype(float)
            w0_spec = np.zeros(tx_tr_spec.shape[1])
            w_bag_spec, _ = reg_logistic_regression(
                y_sub_spec,
                tx_tr_spec[sub_idx_spec],
                lambda_=1e-4,
                initial_w=w0_spec,
                max_iters=200,
                gamma=0.3,
            )
            bag_spec_tr += sigmoid(tx_tr_spec @ w_bag_spec)
            bag_spec_val_ambig += sigmoid(tx_val[ambig_val_mask] @ w_bag_spec)
        s_spec2_tr = bag_spec_tr / 10.0
        s_spec2_val_ambig = bag_spec_val_ambig / 10.0

        # Rank normalize within ambiguous cohort
        r_spec1_tr = rank_normalize(s_spec1_tr)
        r_spec1_val_ambig = rank_normalize(s_spec1_val_ambig)
        r_spec2_tr = rank_normalize(s_spec2_tr)
        r_spec2_val_ambig = rank_normalize(s_spec2_val_ambig)

        blend_spec_tr = 0.50 * r_spec1_tr + 0.50 * r_spec2_tr
        blend_spec_val_ambig = 0.50 * r_spec1_val_ambig + 0.50 * r_spec2_val_ambig

        tau_spec, _ = best_threshold(
            y_tr_spec, blend_spec_tr, np.linspace(0.40, 0.90, 101)
        )

        # Cascade Decision Rule:
        # If Stage 1 score < tau_triage -> Predict -1
        # Else -> Predict based on Stage 2 Specialist
        pred_b = np.full(len(tx_val), -1, dtype=int)
        pred_b[ambig_val_mask] = predict_labels(blend_spec_val_ambig, tau_spec)

        acc_b = accuracy(y_val, pred_b)
        p_b, r_b, f1_b = precision_recall_f1(y_val, pred_b)
        bacc_b = confusion_matrix(y_val, pred_b)["balanced_accuracy"]

        metrics_b["f1"].append(f1_b)
        metrics_b["prec"].append(p_b)
        metrics_b["rec"].append(r_b)
        metrics_b["bacc"].append(bacc_b)
        metrics_b["acc"].append(acc_b)

        print(
            f"Fold {k+1}/5 ({time.time() - t_fold:.1f}s) | "
            f"Arch A (233d Super Tri-Blend): F1={f1_a:.4f} (Prec={p_a:.4f}, Rec={r_a:.4f}, BAcc={bacc_a:.4f}) | "
            f"Arch B (Two-Stage Cascade): F1={f1_b:.4f} (Prec={p_b:.4f}, Rec={r_b:.4f}, BAcc={bacc_b:.4f})"
        )

    total_time = time.time() - t_start

    results = {
        "Architecture_A_233d_Super_Tri_Blend": {
            "f1_mean": float(np.mean(metrics_a["f1"])),
            "f1_std": float(np.std(metrics_a["f1"])),
            "precision_mean": float(np.mean(metrics_a["prec"])),
            "precision_std": float(np.std(metrics_a["prec"])),
            "recall_mean": float(np.mean(metrics_a["rec"])),
            "recall_std": float(np.std(metrics_a["rec"])),
            "balanced_accuracy_mean": float(np.mean(metrics_a["bacc"])),
            "balanced_accuracy_std": float(np.std(metrics_a["bacc"])),
            "accuracy_mean": float(np.mean(metrics_a["acc"])),
            "accuracy_std": float(np.std(metrics_a["acc"])),
            "threshold_mean": float(np.mean(metrics_a["tau"])),
        },
        "Architecture_B_Two_Stage_Cascade": {
            "f1_mean": float(np.mean(metrics_b["f1"])),
            "f1_std": float(np.std(metrics_b["f1"])),
            "precision_mean": float(np.mean(metrics_b["prec"])),
            "precision_std": float(np.std(metrics_b["prec"])),
            "recall_mean": float(np.mean(metrics_b["rec"])),
            "recall_std": float(np.std(metrics_b["rec"])),
            "balanced_accuracy_mean": float(np.mean(metrics_b["bacc"])),
            "balanced_accuracy_std": float(np.std(metrics_b["bacc"])),
            "accuracy_mean": float(np.mean(metrics_b["acc"])),
            "accuracy_std": float(np.std(metrics_b["acc"])),
        },
        "total_time_seconds": float(total_time),
    }

    print("\n" + "=" * 115)
    print("5-Fold Cross-Validation Comparison Summary:")
    print("-" * 115)
    res_a = results["Architecture_A_233d_Super_Tri_Blend"]
    res_b = results["Architecture_B_Two_Stage_Cascade"]
    print(
        f"Arch A (233d Super Tri-Blend): Val F1={res_a['f1_mean']:.4f} +/- {res_a['f1_std']:.4f} | "
        f"Prec={res_a['precision_mean']:.4f} | Rec={res_a['recall_mean']:.4f} | BAcc={res_a['balanced_accuracy_mean']:.4f} | Acc={res_a['accuracy_mean']:.4f}"
    )
    print(
        f"Arch B (Two-Stage Cascade):    Val F1={res_b['f1_mean']:.4f} +/- {res_b['f1_std']:.4f} | "
        f"Prec={res_b['precision_mean']:.4f} | Rec={res_b['recall_mean']:.4f} | BAcc={res_b['balanced_accuracy_mean']:.4f} | Acc={res_b['accuracy_mean']:.4f}"
    )
    print("=" * 115)

    os.makedirs("report", exist_ok=True)
    with open("report/m6_6_cascade_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved report/m6_6_cascade_benchmark.json successfully.\n")

    return results


if __name__ == "__main__":
    run_cascade_benchmark()
