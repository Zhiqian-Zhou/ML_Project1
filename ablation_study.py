"""Progressive Feature Engineering Ablation Study (Milestone 4).

Generates the complete quantitative evaluation for Table 2 of the final report:
- Step 0: Baseline (197 base features)
- Step 1: Step 0 + Informative Missingness Indicators (MNAR, 5% to 85%)
- Step 2: Step 1 + Nominal One-Hot Encodings (MARITAL, EMPLOY1, _RACE)
- Step 3: Step 2 + Clinical Non-linear Interaction Terms
- Step 4: Step 3 + Regularization & Learning Rate Re-tuning (lambda*, gamma*)

Strict pure-NumPy evaluation with Stratified 5-Fold Cross-Validation (seed=42)
and leak-free threshold calibration on training folds.
"""

import json
import time
import numpy as np

from cross_validation import build_k_indices, best_threshold
from feature_engineering import (
    engineer_features_fold,
    build_clinical_interactions,
)
from implementations import reg_logistic_regression, sigmoid
from metrics import accuracy, precision_recall_f1, confusion_matrix, predict_labels


def evaluate_step(
    x,
    y,
    k_indices,
    feature_names,
    step_name,
    add_missing_indicators=False,
    one_hot_nominal=False,
    clinical_interactions=False,
    lambda_=1e-4,
    gamma=0.10,
    max_iters=300,
    threshold_candidates=None,
    active_interactions=None,
):
    """Run leak-free 5-fold cross-validation for a given ablation step."""
    if threshold_candidates is None:
        threshold_candidates = np.linspace(0.05, 0.50, 46)

    k_fold = len(k_indices)
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    balanced_accs = []
    thresholds = []
    dims = []

    t0 = time.time()

    for k in range(k_fold):
        val_idx = k_indices[k]
        tr_idx = np.concatenate([k_indices[i] for i in range(k_fold) if i != k])

        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        tx_tr, tx_val, assembled_names = engineer_features_fold(
            x_tr=x_tr,
            x_te=x_val,
            feature_names=feature_names,
            add_missing_indicators=add_missing_indicators,
            one_hot_nominal=one_hot_nominal,
            clinical_interactions=clinical_interactions,
            min_nan_ratio=0.05,
            max_nan_ratio=0.85,
            drop_first_nominal=True,
            standardize_features=True,
            add_bias=True,
        )

        dims.append(tx_tr.shape[1])

        # Train model on training fold
        y_tr_01 = (y_tr == 1).astype(float)
        w0 = np.zeros(tx_tr.shape[1])
        w, final_loss = reg_logistic_regression(
            y_tr_01,
            tx_tr,
            lambda_=lambda_,
            initial_w=w0,
            max_iters=max_iters,
            gamma=gamma,
        )

        # Calibrate decision threshold tau* on training fold predictions
        tr_scores = sigmoid(tx_tr @ w)
        tau, _ = best_threshold(y_tr, tr_scores, threshold_candidates)
        thresholds.append(tau)

        # Evaluate on held-out validation fold
        val_scores = sigmoid(tx_val @ w)
        y_pred = predict_labels(val_scores, tau)

        acc = accuracy(y_val, y_pred)
        prec, rec, f1 = precision_recall_f1(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        balanced_accs.append(cm["balanced_accuracy"])

    elapsed = time.time() - t0

    res = {
        "step_name": step_name,
        "features_dim": int(np.mean(dims)),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "precision_mean": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions)),
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "balanced_accuracy_mean": float(np.mean(balanced_accs)),
        "balanced_accuracy_std": float(np.std(balanced_accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "threshold_mean": float(np.mean(thresholds)),
        "elapsed_seconds": elapsed,
    }

    print(
        f"{step_name:<38} | D={res['features_dim']:<3} | Acc: {res['accuracy_mean']:.4f} | "
        f"Prec: {res['precision_mean']:.4f} | Rec: {res['recall_mean']:.4f} | "
        f"BAcc: {res['balanced_accuracy_mean']:.4f} | F1: {res['f1_mean']:.4f} +/- {res['f1_std']:.4f} | "
        f"tau*: {res['threshold_mean']:.3f} ({elapsed:.1f}s)"
    )
    return res


def main():
    print("=" * 105)
    print("EPFL CS-433 Project 1: Milestone 4 Progressive Ablation Study")
    print("=" * 105)

    # 1. Load data
    with open("data/processed/features_selected.json", "r") as f:
        feature_names = json.load(f)

    x = np.load("data/processed/x_train.npy")
    y = np.load("data/processed/y_train.npy")
    print(f"Loaded dataset: {x.shape[0]} samples, {len(feature_names)} base features")

    # 2. Stratified 5-Fold split (seed=42)
    k_indices = build_k_indices(y, k_fold=5, seed=42)
    print(f"Constructed 5 stratified folds (seed=42)\n")

    print("-" * 105)
    print(
        f"{'Ablation Step / Hypothesis':<38} | {'Dims':<5} | {'Val Acc':<8} | {'Prec':<7} | "
        f"{'Rec':<7} | {'BAcc':<7} | {'Val F1 (+1)':<18} | {'tau*':<5} (Time)"
    )
    print("-" * 105)

    ablation_results = []

    # Step 0: Baseline (197 features)
    res_s0 = evaluate_step(
        x=x,
        y=y,
        k_indices=k_indices,
        feature_names=feature_names,
        step_name="Step 0: Baseline (197 features)",
        add_missing_indicators=False,
        one_hot_nominal=False,
        clinical_interactions=False,
        lambda_=1e-4,
        gamma=0.10,
        max_iters=300,
    )
    ablation_results.append(res_s0)

    # Step 1: Step 0 + Missingness Indicators (5% to 85%)
    res_s1 = evaluate_step(
        x=x,
        y=y,
        k_indices=k_indices,
        feature_names=feature_names,
        step_name="Step 1: + Missingness Indicators",
        add_missing_indicators=True,
        one_hot_nominal=False,
        clinical_interactions=False,
        lambda_=1e-4,
        gamma=0.10,
        max_iters=300,
    )
    ablation_results.append(res_s1)

    # Step 2: Step 1 + Nominal One-Hot Encodings (MARITAL, EMPLOY1, _RACE)
    res_s2 = evaluate_step(
        x=x,
        y=y,
        k_indices=k_indices,
        feature_names=feature_names,
        step_name="Step 2: + Nominal One-Hot Encodings",
        add_missing_indicators=True,
        one_hot_nominal=True,
        clinical_interactions=False,
        lambda_=1e-4,
        gamma=0.10,
        max_iters=300,
    )
    ablation_results.append(res_s2)

    # Step 3: Step 2 + Clinical Interaction Cross-Terms
    res_s3 = evaluate_step(
        x=x,
        y=y,
        k_indices=k_indices,
        feature_names=feature_names,
        step_name="Step 3: + Clinical Interactions",
        add_missing_indicators=True,
        one_hot_nominal=True,
        clinical_interactions=True,
        lambda_=1e-4,
        gamma=0.10,
        max_iters=300,
    )
    ablation_results.append(res_s3)

    # Step 4: Step 3 + Tuned Regularization (lambda*, gamma*)
    res_s4 = evaluate_step(
        x=x,
        y=y,
        k_indices=k_indices,
        feature_names=feature_names,
        step_name="Step 4: + Tuned Hyperparameters",
        add_missing_indicators=True,
        one_hot_nominal=True,
        clinical_interactions=True,
        lambda_=1e-4,
        gamma=0.15,
        max_iters=400,
    )
    ablation_results.append(res_s4)

    print("-" * 105)

    # Save summary table as json
    with open("report/ablation_table_2.json", "w") as f:
        json.dump(ablation_results, f, indent=2)
    print("\nSaved report/ablation_table_2.json successfully.")


if __name__ == "__main__":
    main()
