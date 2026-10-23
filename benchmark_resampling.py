"""Milestone 6.5: Advanced Algorithmic & Resampling Research Benchmark.

Evaluates 3 high-leverage hypotheses:
1. Hypothesis 1: Stratified Hard-Negative Balanced Logistic Bagging
   - Clinical risk score partitioning: Hard negatives (score >= 2) vs Easy negatives (score <= 1).
   - 50% hard + 50% easy negative sampling per balanced bag (M=15).
2. Hypothesis 2: Pure-NumPy Continuous-Subspace Borderline-SMOTE
   - Identifies positive samples near the negative boundary.
   - Interpolates exclusively along continuous/WoE dimensions, preserving categorical semantics.
3. Hypothesis 3: Simplex Coordinate Search for Optimal Ensemble Weights
   - Systematically searches w1 + w2 + w3 = 1.0 on a 0.05 step grid to maximize cross-validated F1.
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


def continuous_borderline_smote(
    tx_tr, y_tr, cont_indices, k_neighbors=5, n_synthetic=5000, seed=42
):
    """Vectorized pure-NumPy Borderline-SMOTE interpolating strictly on continuous dimensions."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == -1)[0]

    X_pos = tx_tr[pos_idx]

    # Subsample 5000 negatives to compute boundary distance efficiently
    n_neg_sub = min(5000, len(neg_idx))
    sub_neg_idx = rng.choice(neg_idx, size=n_neg_sub, replace=False)
    X_neg_sub = tx_tr[sub_neg_idx]

    # Euclidean distance from each positive to nearest negative
    pos_sq = np.sum(X_pos**2, axis=1, keepdims=True)
    neg_sq = np.sum(X_neg_sub**2, axis=1, keepdims=True).T
    dists = pos_sq + neg_sq - 2.0 * (X_pos @ X_neg_sub.T)
    min_dist_to_neg = np.min(dists, axis=1)

    # Borderline positives are those closest to negatives (top 50%)
    n_border = max(10, len(pos_idx) // 2)
    border_idx_in_pos = np.argsort(min_dist_to_neg)[:n_border]
    X_border = X_pos[border_idx_in_pos]

    # Find k nearest positive neighbors for each borderline point
    border_sq = np.sum(X_border**2, axis=1, keepdims=True)
    pos_sq_all = np.sum(X_pos**2, axis=1, keepdims=True).T
    pos_dists = border_sq + pos_sq_all - 2.0 * (X_border @ X_pos.T)
    k_nn_idx = np.argsort(pos_dists, axis=1)[:, 1 : k_neighbors + 1]

    # Generate synthetic points
    base_indices = rng.choice(len(X_border), size=n_synthetic, replace=True)
    neighbor_choices = rng.integers(0, k_neighbors, size=n_synthetic)
    target_pos_indices = k_nn_idx[base_indices, neighbor_choices]

    lambdas = rng.uniform(0.0, 1.0, size=(n_synthetic, 1)).astype(tx_tr.dtype)

    X_syn = X_border[base_indices].copy()
    diff = (
        X_pos[target_pos_indices][:, cont_indices]
        - X_border[base_indices][:, cont_indices]
    )
    X_syn[:, cont_indices] += lambdas * diff

    y_syn = np.ones(n_synthetic, dtype=y_tr.dtype)
    tx_aug = np.vstack([tx_tr, X_syn])
    y_aug = np.concatenate([y_tr, y_syn])

    return tx_aug, y_aug


def run_benchmark():
    print("=" * 115)
    print("Milestone 6.5: Advanced Algorithmic & Resampling Benchmark (Pure NumPy)")
    print("=" * 115)

    with open("data/processed/features_selected.json") as f:
        feat_names = json.load(f)

    x = np.load("data/processed/x_train.npy")
    y = np.load("data/processed/y_train.npy")
    k_indices = build_k_indices(y, k_fold=5, seed=SEED)

    print(f"Dataset: {x.shape[0]} samples, {len(feat_names)} raw features.")
    print("Executing 5-fold stratified cross-validation with zero data leakage...\n")

    # Storage for out-of-fold predictions
    oof_y = []
    oof_s_ridge = []
    oof_s_bag_uniform = []
    oof_s_bag_hard = []
    oof_s_soft_f1 = []

    # Metrics collectors
    metrics = {
        "H1_bag_uniform": {"f1": [], "prec": [], "rec": [], "bacc": []},
        "H1_bag_hard_neg": {"f1": [], "prec": [], "rec": [], "bacc": []},
        "H2_smote_ridge": {"f1": [], "prec": [], "rec": [], "bacc": []},
        "ridge_base": {"f1": [], "prec": [], "rec": [], "bacc": []},
        "soft_f1_base": {"f1": [], "prec": [], "rec": [], "bacc": []},
    }

    t_start = time.time()

    for k in range(5):
        t_fold = time.time()
        val_idx = k_indices[k]
        tr_idx = np.concatenate([k_indices[i] for i in range(5) if i != k])
        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        # 1. Feature Engineering (223 dims)
        tx_tr, tx_val, out_names = build_advanced_features(
            x_tr, x_val, feature_names=feat_names, y_tr=y_tr, alpha=20.0
        )
        cont_indices = np.array(
            [
                i
                for i, n in enumerate(out_names)
                if any(
                    k in n
                    for k in [
                        "BMI",
                        "PHYSHLTH",
                        "MENTHLTH",
                        "AGE",
                        "HTM4",
                        "WTKG3",
                        "WOE",
                        "LOG",
                    ]
                )
            ]
        )

        # ---------------------------------------------------------
        # Model 1: Cost-Sensitive Weighted Ridge (Baseline on fold)
        # ---------------------------------------------------------
        w_ridge, _ = weighted_ridge_regression(
            y_tr.astype(float), tx_tr, lambda_=1e-4, w_pos=3.0
        )
        s1_tr = tx_tr @ w_ridge
        s1_val = tx_val @ w_ridge
        tau1, _ = best_threshold(y_tr, s1_tr, np.linspace(-1.0, 1.0, 101))
        p, r, f1 = precision_recall_f1(y_val, predict_labels(s1_val, tau1))
        bacc = confusion_matrix(y_val, predict_labels(s1_val, tau1))[
            "balanced_accuracy"
        ]
        metrics["ridge_base"]["f1"].append(f1)
        metrics["ridge_base"]["prec"].append(p)
        metrics["ridge_base"]["rec"].append(r)
        metrics["ridge_base"]["bacc"].append(bacc)

        # ---------------------------------------------------------
        # Hypothesis 2: Continuous Borderline-SMOTE + Ridge
        # ---------------------------------------------------------
        tx_aug, y_aug = continuous_borderline_smote(
            tx_tr,
            y_tr,
            cont_indices=cont_indices,
            k_neighbors=5,
            n_synthetic=5000,
            seed=SEED + k,
        )
        w_ridge_smote, _ = weighted_ridge_regression(
            y_aug.astype(float), tx_aug, lambda_=1e-4, w_pos=3.0
        )
        s_smote_tr = tx_aug @ w_ridge_smote
        s_smote_val = tx_val @ w_ridge_smote
        tau_smote, _ = best_threshold(y_aug, s_smote_tr, np.linspace(-1.0, 1.0, 101))
        p, r, f1 = precision_recall_f1(y_val, predict_labels(s_smote_val, tau_smote))
        bacc = confusion_matrix(y_val, predict_labels(s_smote_val, tau_smote))[
            "balanced_accuracy"
        ]
        metrics["H2_smote_ridge"]["f1"].append(f1)
        metrics["H2_smote_ridge"]["prec"].append(p)
        metrics["H2_smote_ridge"]["rec"].append(r)
        metrics["H2_smote_ridge"]["bacc"].append(bacc)

        # ---------------------------------------------------------
        # Hypothesis 1: Hard-Negative Bagging vs Uniform Bagging
        # ---------------------------------------------------------
        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == -1)[0]
        n_pos = len(pos_idx)

        # Compute clinical vulnerability risk score strictly on training fold
        risk_tr = compute_clinical_risk_scores(x_tr, feat_names)
        hard_neg_idx = neg_idx[risk_tr[neg_idx] >= 2]
        easy_neg_idx = neg_idx[risk_tr[neg_idx] <= 1]

        rng = np.random.default_rng(SEED + k * 100)
        n_bags = 15

        # 1a. Standard Uniform Bagging
        bag_unif_tr = np.zeros(len(tx_tr))
        bag_unif_val = np.zeros(len(tx_val))
        for m in range(n_bags):
            sub_neg = rng.choice(neg_idx, size=n_pos, replace=False)
            sub_idx = np.concatenate([pos_idx, sub_neg])
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
            bag_unif_tr += sigmoid(tx_tr @ w_bag)
            bag_unif_val += sigmoid(tx_val @ w_bag)
        s_unif_tr = bag_unif_tr / float(n_bags)
        s_unif_val = bag_unif_val / float(n_bags)

        tau_unif, _ = best_threshold(y_tr, s_unif_tr, np.linspace(0.1, 0.9, 81))
        p, r, f1 = precision_recall_f1(y_val, predict_labels(s_unif_val, tau_unif))
        bacc = confusion_matrix(y_val, predict_labels(s_unif_val, tau_unif))[
            "balanced_accuracy"
        ]
        metrics["H1_bag_uniform"]["f1"].append(f1)
        metrics["H1_bag_uniform"]["prec"].append(p)
        metrics["H1_bag_uniform"]["rec"].append(r)
        metrics["H1_bag_uniform"]["bacc"].append(bacc)

        # 1b. Stratified Hard-Negative Bagging (50% hard + 50% easy)
        bag_hard_tr = np.zeros(len(tx_tr))
        bag_hard_val = np.zeros(len(tx_val))
        n_half = n_pos // 2
        for m in range(n_bags):
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
            bag_hard_tr += sigmoid(tx_tr @ w_bag)
            bag_hard_val += sigmoid(tx_val @ w_bag)
        s_hard_tr = bag_hard_tr / float(n_bags)
        s_hard_val = bag_hard_val / float(n_bags)

        tau_hard, _ = best_threshold(y_tr, s_hard_tr, np.linspace(0.1, 0.9, 81))
        p, r, f1 = precision_recall_f1(y_val, predict_labels(s_hard_val, tau_hard))
        bacc = confusion_matrix(y_val, predict_labels(s_hard_val, tau_hard))[
            "balanced_accuracy"
        ]
        metrics["H1_bag_hard_neg"]["f1"].append(f1)
        metrics["H1_bag_hard_neg"]["prec"].append(p)
        metrics["H1_bag_hard_neg"]["rec"].append(r)
        metrics["H1_bag_hard_neg"]["bacc"].append(bacc)

        # ---------------------------------------------------------
        # Model 3: Differentiable Soft-F1 Optimizer
        # ---------------------------------------------------------
        y_tr_01 = (y_tr == 1).astype(float)
        w_soft, _ = train_soft_f1_optimizer(
            tx_tr, y_tr_01, max_iters=250, lr=0.03, lambda_reg=1e-5
        )
        s3_tr = sigmoid(tx_tr @ w_soft)
        s3_val = sigmoid(tx_val @ w_soft)
        tau3, _ = best_threshold(y_tr, s3_tr, np.linspace(0.1, 0.9, 81))
        p, r, f1 = precision_recall_f1(y_val, predict_labels(s3_val, tau3))
        bacc = confusion_matrix(y_val, predict_labels(s3_val, tau3))[
            "balanced_accuracy"
        ]
        metrics["soft_f1_base"]["f1"].append(f1)
        metrics["soft_f1_base"]["prec"].append(p)
        metrics["soft_f1_base"]["rec"].append(r)
        metrics["soft_f1_base"]["bacc"].append(bacc)

        # Collect out-of-fold percentile ranks for simplex optimization
        oof_y.append(y_val)
        oof_s_ridge.append(rank_normalize(s1_val))
        oof_s_bag_uniform.append(rank_normalize(s_unif_val))
        oof_s_bag_hard.append(rank_normalize(s_hard_val))
        oof_s_soft_f1.append(rank_normalize(s3_val))

        print(
            f"Fold {k+1}/5 finished in {time.time() - t_fold:.1f}s | "
            f"Ridge={metrics['ridge_base']['f1'][-1]:.4f}, "
            f"SMOTE-Ridge={metrics['H2_smote_ridge']['f1'][-1]:.4f}, "
            f"BagUnif={metrics['H1_bag_uniform']['f1'][-1]:.4f}, "
            f"BagHard={metrics['H1_bag_hard_neg']['f1'][-1]:.4f}, "
            f"SoftF1={metrics['soft_f1_base']['f1'][-1]:.4f}"
        )

    # Concatenate full out-of-fold arrays
    oof_y = np.concatenate(oof_y)
    oof_r1 = np.concatenate(oof_s_ridge)
    oof_r2_unif = np.concatenate(oof_s_bag_uniform)
    oof_r2_hard = np.concatenate(oof_s_bag_hard)
    oof_r3 = np.concatenate(oof_s_soft_f1)

    print("\n" + "=" * 115)
    print("Hypothesis 1 & 2 Comparison (5-Fold Stratified CV):")
    print("-" * 115)
    for model_name, m in metrics.items():
        print(
            f"{model_name:<25} | Val F1: {np.mean(m['f1']):.4f} +/- {np.std(m['f1']):.4f} | "
            f"Prec: {np.mean(m['prec']):.4f} | Rec: {np.mean(m['rec']):.4f} | BAcc: {np.mean(m['bacc']):.4f}"
        )

    # -------------------------------------------------------------
    # Task 3: Simplex Coordinate Search for Optimal Ensemble Weights
    # -------------------------------------------------------------
    print("\n" + "=" * 115)
    print("Task 3: Simplex Grid Search for Optimal Blend Weights (step=0.05)")
    print("=" * 115)

    candidates = np.linspace(0.65, 0.95, 61)

    # Simplex Search A: Using Standard Uniform Bagging (R1 + R2_unif + R3)
    best_f1_unif = -1.0
    best_w_unif = None
    best_tau_unif = None

    for w1 in np.arange(0.0, 1.01, 0.05):
        for w2 in np.arange(0.0, 1.01 - w1 + 1e-6, 0.05):
            w3 = max(0.0, 1.0 - w1 - w2)
            blend_oof = w1 * oof_r1 + w2 * oof_r2_unif + w3 * oof_r3
            tau, f1 = best_threshold(oof_y, blend_oof, candidates)
            if f1 > best_f1_unif:
                best_f1_unif = f1
                best_w_unif = (
                    round(float(w1), 2),
                    round(float(w2), 2),
                    round(float(w3), 2),
                )
                best_tau_unif = float(tau)

    # Simplex Search B: Using Hard-Negative Bagging (R1 + R2_hard + R3)
    best_f1_hard = -1.0
    best_w_hard = None
    best_tau_hard = None

    for w1 in np.arange(0.0, 1.01, 0.05):
        for w2 in np.arange(0.0, 1.01 - w1 + 1e-6, 0.05):
            w3 = max(0.0, 1.0 - w1 - w2)
            blend_oof = w1 * oof_r1 + w2 * oof_r2_hard + w3 * oof_r3
            tau, f1 = best_threshold(oof_y, blend_oof, candidates)
            if f1 > best_f1_hard:
                best_f1_hard = f1
                best_w_hard = (
                    round(float(w1), 2),
                    round(float(w2), 2),
                    round(float(w3), 2),
                )
                best_tau_hard = float(tau)

    print(
        f"Simplex Result (Uniform Bagging): Best Weights={best_w_unif}, "
        f"OOF F1={best_f1_unif:.4f}, tau*={best_tau_unif:.4f}"
    )
    print(
        f"Simplex Result (Hard-Negative Bagging): Best Weights={best_w_hard}, "
        f"OOF F1={best_f1_hard:.4f}, tau*={best_tau_hard:.4f}"
    )

    # Evaluate the winning configuration fold by fold
    use_hard = best_f1_hard >= best_f1_unif
    winning_weights = best_w_hard if use_hard else best_w_unif
    chosen_r2 = oof_s_bag_hard if use_hard else oof_s_bag_uniform

    winning_f1_list = []
    winning_p_list = []
    winning_r_list = []
    winning_bacc_list = []
    winning_acc_list = []

    for k in range(5):
        w1, w2, w3 = winning_weights
        val_idx = k_indices[k]
        y_val = y[val_idx]
        b_val = w1 * oof_s_ridge[k] + w2 * chosen_r2[k] + w3 * oof_s_soft_f1[k]

        tau_k, _ = best_threshold(
            oof_y,
            w1 * oof_r1 + w2 * (oof_r2_hard if use_hard else oof_r2_unif) + w3 * oof_r3,
            candidates,
        )
        pred_k = predict_labels(b_val, tau_k)
        acc = accuracy(y_val, pred_k)
        p, r, f1 = precision_recall_f1(y_val, pred_k)
        cm = confusion_matrix(y_val, pred_k)

        winning_f1_list.append(f1)
        winning_p_list.append(p)
        winning_r_list.append(r)
        winning_bacc_list.append(cm["balanced_accuracy"])
        winning_acc_list.append(acc)

    winning_results = {
        "strategy": (
            "Hard-Negative Bagging Tri-Blend"
            if use_hard
            else "Uniform Bagging Tri-Blend"
        ),
        "optimal_weights": list(winning_weights),
        "accuracy_mean": float(np.mean(winning_acc_list)),
        "accuracy_std": float(np.std(winning_acc_list)),
        "precision_mean": float(np.mean(winning_p_list)),
        "precision_std": float(np.std(winning_p_list)),
        "recall_mean": float(np.mean(winning_r_list)),
        "recall_std": float(np.std(winning_r_list)),
        "balanced_accuracy_mean": float(np.mean(winning_bacc_list)),
        "balanced_accuracy_std": float(np.std(winning_bacc_list)),
        "f1_mean": float(np.mean(winning_f1_list)),
        "f1_std": float(np.std(winning_f1_list)),
        "optimal_tau": float(best_tau_hard if use_hard else best_tau_unif),
        "hypotheses_comparison": {
            "H1_Hard_Negative_Bagging": {
                "f1_mean": float(np.mean(metrics["H1_bag_hard_neg"]["f1"])),
                "f1_std": float(np.std(metrics["H1_bag_hard_neg"]["f1"])),
                "prec_mean": float(np.mean(metrics["H1_bag_hard_neg"]["prec"])),
                "rec_mean": float(np.mean(metrics["H1_bag_hard_neg"]["rec"])),
            },
            "H1_Uniform_Bagging": {
                "f1_mean": float(np.mean(metrics["H1_bag_uniform"]["f1"])),
                "f1_std": float(np.std(metrics["H1_bag_uniform"]["f1"])),
                "prec_mean": float(np.mean(metrics["H1_bag_uniform"]["prec"])),
                "rec_mean": float(np.mean(metrics["H1_bag_uniform"]["rec"])),
            },
            "H2_Continuous_Borderline_SMOTE": {
                "f1_mean": float(np.mean(metrics["H2_smote_ridge"]["f1"])),
                "f1_std": float(np.std(metrics["H2_smote_ridge"]["f1"])),
                "prec_mean": float(np.mean(metrics["H2_smote_ridge"]["prec"])),
                "rec_mean": float(np.mean(metrics["H2_smote_ridge"]["rec"])),
            },
            "Ridge_Base": {
                "f1_mean": float(np.mean(metrics["ridge_base"]["f1"])),
                "f1_std": float(np.std(metrics["ridge_base"]["f1"])),
                "prec_mean": float(np.mean(metrics["ridge_base"]["prec"])),
                "rec_mean": float(np.mean(metrics["ridge_base"]["rec"])),
            },
            "Soft_F1_Base": {
                "f1_mean": float(np.mean(metrics["soft_f1_base"]["f1"])),
                "f1_std": float(np.std(metrics["soft_f1_base"]["f1"])),
                "prec_mean": float(np.mean(metrics["soft_f1_base"]["prec"])),
                "rec_mean": float(np.mean(metrics["soft_f1_base"]["rec"])),
            },
        },
        "total_elapsed_seconds": float(time.time() - t_start),
    }

    print("\n" + "=" * 115)
    print(f"Optimal Super Tri-Blend ({winning_results['strategy']}):")
    print(f"  Weights:           {winning_results['optimal_weights']}")
    print(
        f"  Val F1-Score:      {winning_results['f1_mean']:.4f} +/- {winning_results['f1_std']:.4f}"
    )
    print(
        f"  Val Precision:     {winning_results['precision_mean']:.4f} +/- {winning_results['precision_std']:.4f}"
    )
    print(
        f"  Val Recall:        {winning_results['recall_mean']:.4f} +/- {winning_results['recall_std']:.4f}"
    )
    print(
        f"  Balanced Accuracy: {winning_results['balanced_accuracy_mean']:.4f} +/- {winning_results['balanced_accuracy_std']:.4f}"
    )
    print(
        f"  Val Accuracy:      {winning_results['accuracy_mean']:.4f} +/- {winning_results['accuracy_std']:.4f}"
    )
    print(f"  Calibrated tau*:   {winning_results['optimal_tau']:.4f}")
    print(f"  Total Time:        {winning_results['total_elapsed_seconds']:.1f}s")
    print("=" * 115)

    os.makedirs("report", exist_ok=True)
    with open("report/m6_5_resampling_benchmark.json", "w") as f:
        json.dump(winning_results, f, indent=2)
    print("Saved report/m6_5_resampling_benchmark.json successfully.\n")

    return winning_results


if __name__ == "__main__":
    run_benchmark()
