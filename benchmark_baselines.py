"""Empirical validation of baseline models on CDC BRFSS 2015 dataset.

Evaluates 5-fold stratified cross-validation performance across:
1. Baseline 0: Dummy all-negative (-1) majority classifier.
2. Baseline 1a: Ridge Regression (lambda=1e-4) with default threshold (tau=0.0).
3. Baseline 1b: Ridge Regression (lambda=1e-4) with calibrated threshold (tau tuned on train).
4. Baseline 2a: Regularized Logistic Regression (lambda=1e-4) with default threshold (tau=0.5).
5. Baseline 2b: Regularized Logistic Regression (lambda=1e-4) with calibrated threshold (tau tuned on train).
"""

import time
import numpy as np

from cross_validation import build_k_indices, cross_validate
from implementations import ridge_regression, reg_logistic_regression, sigmoid


def main():
    print("=" * 80)
    print("EPFL CS-433 Project 1: Milestone 3 Baseline Validation Benchmark")
    print("=" * 80)

    # 1. Load sanitized data from processed cache
    t0 = time.time()
    x = np.load("data/processed/x_train.npy")
    y = np.load("data/processed/y_train.npy")
    print(
        f"Data loaded in {time.time() - t0:.3f}s: x={x.shape}, y={y.shape}, "
        f"Positive ratio={np.mean(y == 1):.4%}"
    )

    # 2. Build 5 stratified folds
    k_fold = 5
    seed = 42
    k_indices = build_k_indices(y, k_fold=k_fold, seed=seed)
    print(f"Constructed {k_fold} stratified folds (seed={seed})")

    # --------------------------------------------------------------------------
    # Baseline 0: Dummy All-Negative (-1) Predictor
    # --------------------------------------------------------------------------
    print("\n[Evaluating Baseline 0: All-Negative (-1) Majority Class Predictor]...")
    res_b0 = cross_validate(
        y=y,
        x=x,
        k_indices=k_indices,
        train_fn=lambda y_tr, x_tr: None,
        predict_fn=lambda model, x_te: np.full(len(x_te), -1),
        tune_threshold=False,
        preprocess=False,
    )
    print(
        f"Baseline 0 -> Accuracy: {res_b0['accuracy_mean']:.4f} +/- {res_b0['accuracy_std']:.4f} | "
        f"Precision: {res_b0['precision_mean']:.4f} | Recall: {res_b0['recall_mean']:.4f} | "
        f"Balanced Acc: {res_b0['balanced_accuracy_mean']:.4f} | F1: {res_b0['f1_mean']:.4f}"
    )

    # --------------------------------------------------------------------------
    # Baseline 1a: Ridge Regression (lambda=1e-4, default tau=0.0)
    # --------------------------------------------------------------------------
    print("\n[Evaluating Baseline 1a: Ridge Regression (tau=0.0 default)]...")
    lambda_ridge = 1e-4

    def train_ridge(y_tr, tx_tr):
        w, _ = ridge_regression(y_tr.astype(float), tx_tr, lambda_=lambda_ridge)
        return w

    def predict_ridge_scores(model, tx_te):
        return tx_te @ model

    t_r0 = time.time()
    res_b1a = cross_validate(
        y=y,
        x=x,
        k_indices=k_indices,
        train_fn=train_ridge,
        predict_fn=predict_ridge_scores,
        tune_threshold=False,
        preprocess=True,
    )
    print(
        f"Ridge (tau=0.0) [{time.time() - t_r0:.2f}s] -> "
        f"Accuracy: {res_b1a['accuracy_mean']:.4f} +/- {res_b1a['accuracy_std']:.4f} | "
        f"Precision: {res_b1a['precision_mean']:.4f} +/- {res_b1a['precision_std']:.4f} | "
        f"Recall: {res_b1a['recall_mean']:.4f} +/- {res_b1a['recall_std']:.4f} | "
        f"Balanced Acc: {res_b1a['balanced_accuracy_mean']:.4f} +/- {res_b1a['balanced_accuracy_std']:.4f} | "
        f"F1: {res_b1a['f1_mean']:.4f} +/- {res_b1a['f1_std']:.4f}"
    )

    # --------------------------------------------------------------------------
    # Baseline 1b: Ridge Regression (lambda=1e-4, calibrated tau)
    # --------------------------------------------------------------------------
    print("\n[Evaluating Baseline 1b: Ridge Regression (Calibrated Threshold)]...")
    t_r1 = time.time()
    res_b1b = cross_validate(
        y=y,
        x=x,
        k_indices=k_indices,
        train_fn=train_ridge,
        predict_fn=predict_ridge_scores,
        tune_threshold=True,
        threshold_candidates=np.linspace(-0.9, 0.1, 100),
        preprocess=True,
    )
    mean_tau_ridge = np.mean(res_b1b["thresholds"])
    print(
        f"Ridge (Calibrated tau={mean_tau_ridge:.3f}) [{time.time() - t_r1:.2f}s] -> "
        f"Accuracy: {res_b1b['accuracy_mean']:.4f} +/- {res_b1b['accuracy_std']:.4f} | "
        f"Precision: {res_b1b['precision_mean']:.4f} +/- {res_b1b['precision_std']:.4f} | "
        f"Recall: {res_b1b['recall_mean']:.4f} +/- {res_b1b['recall_std']:.4f} | "
        f"Balanced Acc: {res_b1b['balanced_accuracy_mean']:.4f} +/- {res_b1b['balanced_accuracy_std']:.4f} | "
        f"F1: {res_b1b['f1_mean']:.4f} +/- {res_b1b['f1_std']:.4f}"
    )

    # --------------------------------------------------------------------------
    # Baseline 2a: Regularized Logistic Regression (default tau=0.5)
    # --------------------------------------------------------------------------
    print(
        "\n[Evaluating Baseline 2a: Regularized Logistic Regression (tau=0.5 default)]..."
    )
    gamma_log = 0.1
    max_iters_log = 300
    lambda_log = 1e-4

    def train_log(y_tr, tx_tr):
        y_01 = (y_tr == 1).astype(float)
        w0 = np.zeros(tx_tr.shape[1])
        w, _ = reg_logistic_regression(
            y_01,
            tx_tr,
            lambda_=lambda_log,
            initial_w=w0,
            max_iters=max_iters_log,
            gamma=gamma_log,
        )
        return w

    def predict_log_probs(model, tx_te):
        return sigmoid(tx_te @ model)

    def predict_log_default(model, tx_te):
        probs = sigmoid(tx_te @ model)
        return np.where(probs >= 0.5, 1, -1)

    t_l0 = time.time()
    res_b2a = cross_validate(
        y=y,
        x=x,
        k_indices=k_indices,
        train_fn=train_log,
        predict_fn=predict_log_default,
        tune_threshold=False,
        preprocess=True,
    )
    print(
        f"Logistic (tau=0.5) [{time.time() - t_l0:.2f}s] -> "
        f"Accuracy: {res_b2a['accuracy_mean']:.4f} +/- {res_b2a['accuracy_std']:.4f} | "
        f"Precision: {res_b2a['precision_mean']:.4f} +/- {res_b2a['precision_std']:.4f} | "
        f"Recall: {res_b2a['recall_mean']:.4f} +/- {res_b2a['recall_std']:.4f} | "
        f"Balanced Acc: {res_b2a['balanced_accuracy_mean']:.4f} +/- {res_b2a['balanced_accuracy_std']:.4f} | "
        f"F1: {res_b2a['f1_mean']:.4f} +/- {res_b2a['f1_std']:.4f}"
    )

    # --------------------------------------------------------------------------
    # Baseline 2b: Regularized Logistic Regression (calibrated tau)
    # --------------------------------------------------------------------------
    print(
        "\n[Evaluating Baseline 2b: Regularized Logistic Regression (Calibrated Threshold)]..."
    )
    t_l1 = time.time()
    res_b2b = cross_validate(
        y=y,
        x=x,
        k_indices=k_indices,
        train_fn=train_log,
        predict_fn=predict_log_probs,
        tune_threshold=True,
        threshold_candidates=np.linspace(0.05, 0.50, 46),
        preprocess=True,
    )
    mean_tau_log = np.mean(res_b2b["thresholds"])
    print(
        f"Logistic (Calibrated tau={mean_tau_log:.3f}) [{time.time() - t_l1:.2f}s] -> "
        f"Accuracy: {res_b2b['accuracy_mean']:.4f} +/- {res_b2b['accuracy_std']:.4f} | "
        f"Precision: {res_b2b['precision_mean']:.4f} +/- {res_b2b['precision_std']:.4f} | "
        f"Recall: {res_b2b['recall_mean']:.4f} +/- {res_b2b['recall_std']:.4f} | "
        f"Balanced Acc: {res_b2b['balanced_accuracy_mean']:.4f} +/- {res_b2b['balanced_accuracy_std']:.4f} | "
        f"F1: {res_b2b['f1_mean']:.4f} +/- {res_b2b['f1_std']:.4f}"
    )

    print("\n" + "=" * 80)
    print("Baseline Cross-Validation Summary Table:")
    print("=" * 80)
    print(
        f"{'Model / Baseline':<32} | {'Val Acc':<8} | {'Precision':<10} | {'Recall':<8} | {'Balanced Acc':<12} | {'Val F1 (+1)':<12}"
    )
    print("-" * 94)
    print(
        f"{'B0: All-Negative Dummy':<32} | {res_b0['accuracy_mean']:.4f}   | {res_b0['precision_mean']:.4f}     | {res_b0['recall_mean']:.4f}   | {res_b0['balanced_accuracy_mean']:.4f}       | {res_b0['f1_mean']:.4f}"
    )
    print(
        f"{'B1a: Ridge (tau=0.0)':<32} | {res_b1a['accuracy_mean']:.4f}   | {res_b1a['precision_mean']:.4f}     | {res_b1a['recall_mean']:.4f}   | {res_b1a['balanced_accuracy_mean']:.4f}       | {res_b1a['f1_mean']:.4f} +/- {res_b1a['f1_std']:.4f}"
    )
    print(
        f"{'B1b: Ridge (Tuned tau)':<32} | {res_b1b['accuracy_mean']:.4f}   | {res_b1b['precision_mean']:.4f}     | {res_b1b['recall_mean']:.4f}   | {res_b1b['balanced_accuracy_mean']:.4f}       | {res_b1b['f1_mean']:.4f} +/- {res_b1b['f1_std']:.4f}"
    )
    print(
        f"{'B2a: Logistic (tau=0.5)':<32} | {res_b2a['accuracy_mean']:.4f}   | {res_b2a['precision_mean']:.4f}     | {res_b2a['recall_mean']:.4f}   | {res_b2a['balanced_accuracy_mean']:.4f}       | {res_b2a['f1_mean']:.4f} +/- {res_b2a['f1_std']:.4f}"
    )
    print(
        f"{'B2b: Logistic (Tuned tau)':<32} | {res_b2b['accuracy_mean']:.4f}   | {res_b2b['precision_mean']:.4f}     | {res_b2b['recall_mean']:.4f}   | {res_b2b['balanced_accuracy_mean']:.4f}       | {res_b2b['f1_mean']:.4f} +/- {res_b2b['f1_std']:.4f}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
