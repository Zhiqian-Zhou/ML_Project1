"""Cost-Sensitive / Class-Weighted Benchmark on Streamlined Feature Space (205 Dims).

Evaluates 5-fold stratified cross-validation performance across:
1. Baseline Unweighted Ridge (w_pos = 1.0)
2. Cost-Sensitive Ridge (w_pos = 1.5)
3. Cost-Sensitive Ridge (w_pos = 2.0)
4. Cost-Sensitive Ridge (w_pos = 2.5)
5. Cost-Sensitive Ridge (w_pos = 3.0)
6. Cost-Sensitive Ridge (w_pos = 5.0)
7. Regularized Logistic Regression (lambda = 1e-4, gamma = 0.15, max_iters = 350)
"""

import json
import time
import numpy as np

from cross_validation import (
    build_k_indices,
    cross_validate_weighted_ridge,
    best_threshold,
)
from feature_engineering import build_streamlined_features
from implementations import reg_logistic_regression, sigmoid
from metrics import accuracy, precision_recall_f1, confusion_matrix, predict_labels


def main():
    print("=" * 105)
    print(
        "Cost-Sensitive Learning Benchmark (Streamlined 205-Dimensional Representation)"
    )
    print("=" * 105)

    with open("data/processed/features_selected.json") as f:
        feat_names = json.load(f)

    x = np.load("data/processed/x_train.npy")
    y = np.load("data/processed/y_train.npy")
    k_indices = build_k_indices(y, k_fold=5, seed=42)

    print(f"Dataset: {x.shape[0]} samples, {len(feat_names)} base features")
    print(f"Constructed 5 stratified folds (seed=42)\n")

    print("-" * 105)
    print(
        f"{'Model / Architecture':<34} | {'w_pos':<6} | {'Val Acc':<8} | {'Prec':<7} | "
        f"{'Rec':<7} | {'BAcc':<7} | {'Val F1 (+1)':<18} | {'tau*':<7} | {'Time':<6}"
    )
    print("-" * 105)

    results = []

    # Sweep w_pos for Weighted Ridge Regression
    for w_pos in [1.0, 1.5, 2.0, 2.5, 3.0, 5.0]:
        t0 = time.time()
        res = cross_validate_weighted_ridge(
            y=y,
            x=x,
            k_indices=k_indices,
            feature_names=feat_names,
            lambda_=1e-4,
            w_pos=w_pos,
            threshold_candidates=np.linspace(-1.0, 1.0, 101),
        )
        elapsed = time.time() - t0
        res["model"] = f"Weighted Ridge (w_pos={w_pos})"
        res["w_pos"] = w_pos
        res["elapsed"] = elapsed
        results.append(res)

        print(
            f"{res['model']:<34} | {w_pos:<6.1f} | {res['accuracy_mean']:.4f}   | "
            f"{res['precision_mean']:.4f}  | {res['recall_mean']:.4f}  | "
            f"{res['balanced_accuracy_mean']:.4f}  | {res['f1_mean']:.4f} +/- {res['f1_std']:.4f} | "
            f"{res['threshold_mean']:+.4f} | {elapsed:.2f}s"
        )

    # Compare against Regularized Logistic Regression
    t0 = time.time()
    f1_list, p_list, r_list, bacc_list, acc_list, tau_list = [], [], [], [], [], []
    for k in range(5):
        val_idx = k_indices[k]
        tr_idx = np.concatenate([k_indices[i] for i in range(5) if i != k])
        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        tx_tr, tx_val, _ = build_streamlined_features(
            x_tr=x_tr, x_te=x_val, feature_names=feat_names
        )

        y_tr_01 = (y_tr == 1).astype(float)
        w0 = np.zeros(tx_tr.shape[1])
        w, _ = reg_logistic_regression(
            y_tr_01, tx_tr, lambda_=1e-4, initial_w=w0, max_iters=350, gamma=0.15
        )

        tr_scores = sigmoid(tx_tr @ w)
        val_scores = sigmoid(tx_val @ w)
        tau, _ = best_threshold(y_tr, tr_scores, np.linspace(0.05, 0.50, 46))
        tau_list.append(tau)

        pred = predict_labels(val_scores, tau)
        acc = accuracy(y_val, pred)
        prec, rec, f1 = precision_recall_f1(y_val, pred)
        cm = confusion_matrix(y_val, pred)

        acc_list.append(acc)
        p_list.append(prec)
        r_list.append(rec)
        f1_list.append(f1)
        bacc_list.append(cm["balanced_accuracy"])

    elapsed_log = time.time() - t0
    res_log = {
        "model": "Regularized Logistic Regression",
        "w_pos": 1.0,
        "accuracy_mean": float(np.mean(acc_list)),
        "precision_mean": float(np.mean(p_list)),
        "recall_mean": float(np.mean(r_list)),
        "balanced_accuracy_mean": float(np.mean(bacc_list)),
        "f1_mean": float(np.mean(f1_list)),
        "f1_std": float(np.std(f1_list)),
        "threshold_mean": float(np.mean(tau_list)),
        "elapsed": elapsed_log,
    }
    results.append(res_log)

    print(
        f"{res_log['model']:<34} | {'1.0':<6} | {res_log['accuracy_mean']:.4f}   | "
        f"{res_log['precision_mean']:.4f}  | {res_log['recall_mean']:.4f}  | "
        f"{res_log['balanced_accuracy_mean']:.4f}  | {res_log['f1_mean']:.4f} +/- {res_log['f1_std']:.4f} | "
        f"{res_log['threshold_mean']:+.4f} | {elapsed_log:.2f}s"
    )
    print("-" * 105)

    with open("report/cost_sensitive_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved report/cost_sensitive_benchmark.json successfully.")


if __name__ == "__main__":
    main()
