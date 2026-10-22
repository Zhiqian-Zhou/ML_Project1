"""Comprehensive 5-Fold Stratified Cross-Validation Benchmark for 4 Bold Hypotheses.

Hypothesis A: Empirical Bayesian Log-Odds Encoding (WoE)
Hypothesis B: Multi-Model Heterogeneous Rank Blending (Ridge + Logistic + Bagging)
Hypothesis C: Pure-NumPy 2-Layer Neural Network (MLP from Scratch)
Hypothesis D: Clinical Multi-morbidity Stacking Score
"""

import json
import os
import time
import numpy as np

from cross_validation import build_k_indices, weighted_ridge_regression
from feature_engineering import build_streamlined_features
from implementations import reg_logistic_regression, ridge_regression, sigmoid
from metrics import (
    accuracy,
    best_threshold,
    confusion_matrix,
    precision_recall_f1,
    predict_labels,
)

SEED = 42


def compute_woe_features(x_tr, x_val, feature_names, alpha=20.0, y_tr=None):
    """Compute empirical Bayesian Laplace-smoothed Log-Odds (WoE) out-of-fold."""
    woe_cols = ["_AGEG5YR", "GENHLTH", "_SMOKER3", "INCOME2"]
    woe_indices = [feature_names.index(c) for c in woe_cols if c in feature_names]

    pos_p = np.mean(y_tr == 1)
    neg_p = 1.0 - pos_p
    prior_lo = np.log(pos_p / neg_p)

    woe_tr_list, woe_val_list = [], []
    for idx in woe_indices:
        col_tr = x_tr[:, idx]
        col_val = x_val[:, idx]
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
        woe_val_list.append(np.array([mapping.get(v, 0.0) for v in col_val]))

    w_mat_tr = np.column_stack(woe_tr_list)
    w_mat_val = np.column_stack(woe_val_list)

    w_mean = np.mean(w_mat_tr, axis=0)
    w_std = np.std(w_mat_tr, axis=0)
    w_std[w_std == 0.0] = 1.0

    return (w_mat_tr - w_mean) / w_std, (w_mat_val - w_mean) / w_std


def compute_multimorbidity_score(x_mat, feature_names):
    """Compute composite cardiovascular clinical severity index (0 to 7)."""
    score = np.zeros(len(x_mat))

    def col_eq(col_name, val):
        if col_name in feature_names:
            idx = feature_names.index(col_name)
            return x_mat[:, idx] == val
        return np.zeros(len(x_mat), dtype=bool)

    # 1. High Blood Pressure
    score += (col_eq("_RFHYPE5", 2) | col_eq("BPHIGH4", 1)).astype(float)
    # 2. High Cholesterol
    score += (col_eq("_RFCHOL", 2) | col_eq("TOLDHI2", 1)).astype(float)
    # 3. Diabetes
    score += col_eq("DIABETE3", 1).astype(float)
    # 4. Stroke
    score += col_eq("CVDSTRK3", 1).astype(float)
    # 5. Smoker
    score += (col_eq("_SMOKER3", 1) | col_eq("_SMOKER3", 2)).astype(float)
    # 6. Kidney Disease
    score += col_eq("CHCKIDNY", 1).astype(float)
    # 7. COPD
    score += col_eq("CHCCOPD1", 1).astype(float)

    return score[:, None]


class PureNumpyMLP:
    """Lightweight 2-layer Neural Network (MLP) implemented in pure NumPy.
    Architecture: D -> H (ReLU) -> 1 (Sigmoid)
    Loss: Weighted Binary Cross-Entropy with L2 weight decay.
    Optimization: Mini-batch Adam.
    """

    def __init__(self, in_dim, hidden_dim=64, w_pos=2.5, l2_reg=1e-4, seed=42):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.w_pos = w_pos
        self.l2_reg = l2_reg
        rng = np.random.default_rng(seed)

        # He / Kaiming initialization
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), (in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        # Xavier initialization
        self.W2 = rng.normal(0.0, np.sqrt(1.0 / hidden_dim), (hidden_dim, 1))
        self.b2 = np.zeros(1)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0.0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        # Numerically stable sigmoid
        self.p = 1.0 / (1.0 + np.exp(-np.clip(self.z2, -30.0, 30.0)))
        return self.p.ravel()

    def fit(self, X, y01, epochs=10, batch_size=1024, lr=0.003):
        N = len(X)
        mW1, vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        mb1, vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        mW2, vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        mb2, vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        y_col = y01[:, None]
        # Sample weights: w_pos for 1, 1.0 for 0
        weights = np.where(y_col == 1.0, self.w_pos, 1.0)

        rng = np.random.default_rng(SEED)
        indices = np.arange(N)

        for epoch in range(epochs):
            rng.shuffle(indices)
            for start in range(0, N, batch_size):
                t += 1
                b_idx = indices[start : start + batch_size]
                xb = X[b_idx]
                yb = y_col[b_idx]
                wb = weights[b_idx]
                B = len(xb)

                # Forward
                z1 = xb @ self.W1 + self.b1
                a1 = np.maximum(0.0, z1)
                z2 = a1 @ self.W2 + self.b2
                p = 1.0 / (1.0 + np.exp(-np.clip(z2, -30.0, 30.0)))

                # Weighted BCE gradient: dL/dz2 = wb * (p - y) / B
                dz2 = wb * (p - yb) / float(B)
                dW2 = a1.T @ dz2 + self.l2_reg * self.W2
                db2 = np.sum(dz2, axis=0)

                # Backprop into hidden layer
                da1 = dz2 @ self.W2.T
                dz1 = da1 * (z1 > 0).astype(float)
                dW1 = xb.T @ dz1 + self.l2_reg * self.W1
                db1 = np.sum(dz1, axis=0)

                # Adam updates
                for param, grad, m, v in [
                    (self.W1, dW1, mW1, vW1),
                    (self.b1, db1, mb1, vb1),
                    (self.W2, dW2, mW2, vW2),
                    (self.b2, db2, mb2, vb2),
                ]:
                    m[:] = beta1 * m + (1.0 - beta1) * grad
                    v[:] = beta2 * v + (1.0 - beta2) * (grad**2)
                    m_hat = m / (1.0 - beta1**t)
                    v_hat = v / (1.0 - beta2**t)
                    param -= lr * m_hat / (np.sqrt(v_hat) + eps)


def rank_normalize(scores):
    """Percentile rank normalization to uniform [0, 1]."""
    return np.argsort(np.argsort(scores)) / float(len(scores))


def run_hypotheses_cv():
    print("=" * 110)
    print("Running Stratified 5-Fold CV for 4 Bold Hypotheses (Pure NumPy)")
    print("=" * 110)

    with open("data/processed/features_selected.json") as f:
        feat_names = json.load(f)

    x = np.load("data/processed/x_train.npy")
    y = np.load("data/processed/y_train.npy")
    k_indices = build_k_indices(y, k_fold=5, seed=42)

    results = {}

    # -------------------------------------------------------------
    # Hypothesis A: Empirical Bayesian Log-Odds Encoding (WoE)
    # -------------------------------------------------------------
    print(
        "\n[1/4] Evaluating Hypothesis A: Empirical Bayesian Log-Odds (WoE) on 209 Dims..."
    )
    t0 = time.time()
    acc_l, p_l, r_l, bacc_l, f1_l, tau_l = [], [], [], [], [], []
    for k in range(5):
        val_idx = k_indices[k]
        tr_idx = np.concatenate([k_indices[i] for i in range(5) if i != k])
        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        tx_tr, tx_val, _ = build_streamlined_features(x_tr, x_val, feat_names)
        woe_tr, woe_val = compute_woe_features(
            x_tr, x_val, feat_names, alpha=20.0, y_tr=y_tr
        )
        tx_tr = np.c_[tx_tr, woe_tr]
        tx_val = np.c_[tx_val, woe_val]

        w, _ = weighted_ridge_regression(
            y_tr.astype(float), tx_tr, lambda_=1e-4, w_pos=3.0
        )
        s_tr = tx_tr @ w
        s_val = tx_val @ w

        tau, _ = best_threshold(y_tr, s_tr, np.linspace(-1.0, 1.0, 101))
        tau_l.append(tau)
        pred = predict_labels(s_val, tau)

        acc_l.append(accuracy(y_val, pred))
        p, r, f1 = precision_recall_f1(y_val, pred)
        p_l.append(p)
        r_l.append(r)
        f1_l.append(f1)
        bacc_l.append(confusion_matrix(y_val, pred)["balanced_accuracy"])

    results["Hypothesis A: WoE Target Encoding"] = {
        "dims": 209,
        "accuracy_mean": float(np.mean(acc_l)),
        "accuracy_std": float(np.std(acc_l)),
        "precision_mean": float(np.mean(p_l)),
        "precision_std": float(np.std(p_l)),
        "recall_mean": float(np.mean(r_l)),
        "recall_std": float(np.std(r_l)),
        "balanced_accuracy_mean": float(np.mean(bacc_l)),
        "balanced_accuracy_std": float(np.std(bacc_l)),
        "f1_mean": float(np.mean(f1_l)),
        "f1_std": float(np.std(f1_l)),
        "threshold_mean": float(np.mean(tau_l)),
        "elapsed": time.time() - t0,
    }
    res = results["Hypothesis A: WoE Target Encoding"]
    print(
        f"  -> Val F1: {res['f1_mean']:.4f} +/- {res['f1_std']:.4f} (Prec: {res['precision_mean']:.4f}, Rec: {res['recall_mean']:.4f}, Time: {res['elapsed']:.2f}s)"
    )

    # -------------------------------------------------------------
    # Hypothesis B: Multi-Model Rank Blending (Heterogeneous Ensemble)
    # -------------------------------------------------------------
    print(
        "\n[2/4] Evaluating Hypothesis B: Multi-Model Heterogeneous Rank Blend + WoE..."
    )
    t0 = time.time()
    acc_l, p_l, r_l, bacc_l, f1_l, tau_l = [], [], [], [], [], []
    for k in range(5):
        val_idx = k_indices[k]
        tr_idx = np.concatenate([k_indices[i] for i in range(5) if i != k])
        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        tx_tr, tx_val, _ = build_streamlined_features(x_tr, x_val, feat_names)
        woe_tr, woe_val = compute_woe_features(
            x_tr, x_val, feat_names, alpha=20.0, y_tr=y_tr
        )
        tx_tr = np.c_[tx_tr, woe_tr]
        tx_val = np.c_[tx_val, woe_val]

        # Model 1: Cost-Sensitive Ridge (0.45)
        w_ridge, _ = weighted_ridge_regression(
            y_tr.astype(float), tx_tr, lambda_=1e-4, w_pos=3.0
        )
        s1_tr = tx_tr @ w_ridge
        s1_val = tx_val @ w_ridge

        # Model 2: Regularized Logistic (0.35)
        y_tr_01 = (y_tr == 1).astype(float)
        w0 = np.zeros(tx_tr.shape[1])
        w_log, _ = reg_logistic_regression(
            y_tr_01, tx_tr, lambda_=1e-4, initial_w=w0, max_iters=350, gamma=0.15
        )
        s2_tr = sigmoid(tx_tr @ w_log)
        s2_val = sigmoid(tx_val @ w_log)

        # Model 3: Balanced Undersampling Bagging Ensemble (0.20, M=10)
        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == -1)[0]
        n_pos = len(pos_idx)
        rng = np.random.default_rng(SEED + k)
        bag_tr = np.zeros(len(tx_tr))
        bag_val = np.zeros(len(tx_val))
        for _ in range(10):
            sub_neg = rng.choice(neg_idx, size=n_pos, replace=False)
            sub_idx = np.concatenate([pos_idx, sub_neg])
            w_bag, _ = ridge_regression(
                y_tr[sub_idx].astype(float), tx_tr[sub_idx], lambda_=1e-4
            )
            bag_tr += tx_tr @ w_bag
            bag_val += tx_val @ w_bag
        s3_tr = bag_tr / 10.0
        s3_val = bag_val / 10.0

        # Percentile Rank Normalization
        r1_tr, r1_val = rank_normalize(s1_tr), rank_normalize(s1_val)
        r2_tr, r2_val = rank_normalize(s2_tr), rank_normalize(s2_val)
        r3_tr, r3_val = rank_normalize(s3_tr), rank_normalize(s3_val)

        blend_tr = 0.45 * r1_tr + 0.35 * r2_tr + 0.20 * r3_tr
        blend_val = 0.45 * r1_val + 0.35 * r2_val + 0.20 * r3_val

        tau, _ = best_threshold(y_tr, blend_tr, np.linspace(0.65, 0.95, 61))
        tau_l.append(tau)
        pred = predict_labels(blend_val, tau)

        acc_l.append(accuracy(y_val, pred))
        p, r, f1 = precision_recall_f1(y_val, pred)
        p_l.append(p)
        r_l.append(r)
        f1_l.append(f1)
        bacc_l.append(confusion_matrix(y_val, pred)["balanced_accuracy"])

    results["Hypothesis B: Heterogeneous Rank Blend + WoE"] = {
        "dims": 209,
        "accuracy_mean": float(np.mean(acc_l)),
        "accuracy_std": float(np.std(acc_l)),
        "precision_mean": float(np.mean(p_l)),
        "precision_std": float(np.std(p_l)),
        "recall_mean": float(np.mean(r_l)),
        "recall_std": float(np.std(r_l)),
        "balanced_accuracy_mean": float(np.mean(bacc_l)),
        "balanced_accuracy_std": float(np.std(bacc_l)),
        "f1_mean": float(np.mean(f1_l)),
        "f1_std": float(np.std(f1_l)),
        "threshold_mean": float(np.mean(tau_l)),
        "elapsed": time.time() - t0,
    }
    res = results["Hypothesis B: Heterogeneous Rank Blend + WoE"]
    print(
        f"  -> Val F1: {res['f1_mean']:.4f} +/- {res['f1_std']:.4f} (Prec: {res['precision_mean']:.4f}, Rec: {res['recall_mean']:.4f}, Time: {res['elapsed']:.2f}s)"
    )

    # -------------------------------------------------------------
    # Hypothesis C: Pure-NumPy 2-Layer Neural Network (MLP)
    # -------------------------------------------------------------
    print("\n[3/4] Evaluating Hypothesis C: Pure-NumPy 2-Layer Neural Network (MLP)...")
    t0 = time.time()
    acc_l, p_l, r_l, bacc_l, f1_l, tau_l, tr_f1_l = [], [], [], [], [], [], []
    for k in range(5):
        val_idx = k_indices[k]
        tr_idx = np.concatenate([k_indices[i] for i in range(5) if i != k])
        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        tx_tr, tx_val, _ = build_streamlined_features(x_tr, x_val, feat_names)
        # Standardize features without bias column for MLP
        X_tr = tx_tr[:, 1:]
        X_val = tx_val[:, 1:]
        y_tr_01 = (y_tr == 1).astype(float)

        mlp = PureNumpyMLP(
            in_dim=X_tr.shape[1], hidden_dim=64, w_pos=2.5, l2_reg=1e-4, seed=SEED + k
        )
        mlp.fit(X_tr, y_tr_01, epochs=10, batch_size=1024, lr=0.003)

        p_tr = mlp.forward(X_tr)
        p_val = mlp.forward(X_val)

        tau, tr_f1 = best_threshold(y_tr, p_tr, np.linspace(0.1, 0.7, 61))
        tau_l.append(tau)
        tr_f1_l.append(tr_f1)
        pred = predict_labels(p_val, tau)

        acc_l.append(accuracy(y_val, pred))
        p, r, f1 = precision_recall_f1(y_val, pred)
        p_l.append(p)
        r_l.append(r)
        f1_l.append(f1)
        bacc_l.append(confusion_matrix(y_val, pred)["balanced_accuracy"])

    results["Hypothesis C: Pure-NumPy 2-Layer MLP"] = {
        "dims": 205,
        "accuracy_mean": float(np.mean(acc_l)),
        "accuracy_std": float(np.std(acc_l)),
        "precision_mean": float(np.mean(p_l)),
        "precision_std": float(np.std(p_l)),
        "recall_mean": float(np.mean(r_l)),
        "recall_std": float(np.std(r_l)),
        "balanced_accuracy_mean": float(np.mean(bacc_l)),
        "balanced_accuracy_std": float(np.std(bacc_l)),
        "f1_mean": float(np.mean(f1_l)),
        "f1_std": float(np.std(f1_l)),
        "train_f1_mean": float(np.mean(tr_f1_l)),
        "threshold_mean": float(np.mean(tau_l)),
        "elapsed": time.time() - t0,
    }
    res = results["Hypothesis C: Pure-NumPy 2-Layer MLP"]
    print(
        f"  -> Val F1: {res['f1_mean']:.4f} +/- {res['f1_std']:.4f} (Train F1: {res['train_f1_mean']:.4f}, Prec: {res['precision_mean']:.4f}, Rec: {res['recall_mean']:.4f}, Time: {res['elapsed']:.2f}s)"
    )

    # -------------------------------------------------------------
    # Hypothesis D: Clinical Multi-morbidity Stacking Score
    # -------------------------------------------------------------
    print("\n[4/4] Evaluating Hypothesis D: Clinical Multi-morbidity Stacking Score...")
    t0 = time.time()
    acc_l, p_l, r_l, bacc_l, f1_l, tau_l = [], [], [], [], [], []
    for k in range(5):
        val_idx = k_indices[k]
        tr_idx = np.concatenate([k_indices[i] for i in range(5) if i != k])
        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        tx_tr, tx_val, _ = build_streamlined_features(x_tr, x_val, feat_names)
        mm_tr = compute_multimorbidity_score(x_tr, feat_names)
        mm_val = compute_multimorbidity_score(x_val, feat_names)

        # Standardize multimorbidity score
        mm_m = np.mean(mm_tr)
        mm_s = np.std(mm_tr)
        mm_tr = (mm_tr - mm_m) / mm_s
        mm_val = (mm_val - mm_m) / mm_s

        tx_tr = np.c_[tx_tr, mm_tr]
        tx_val = np.c_[tx_val, mm_val]

        w, _ = weighted_ridge_regression(
            y_tr.astype(float), tx_tr, lambda_=1e-4, w_pos=3.0
        )
        s_tr = tx_tr @ w
        s_val = tx_val @ w

        tau, _ = best_threshold(y_tr, s_tr, np.linspace(-1.0, 1.0, 101))
        tau_l.append(tau)
        pred = predict_labels(s_val, tau)

        acc_l.append(accuracy(y_val, pred))
        p, r, f1 = precision_recall_f1(y_val, pred)
        p_l.append(p)
        r_l.append(r)
        f1_l.append(f1)
        bacc_l.append(confusion_matrix(y_val, pred)["balanced_accuracy"])

    results["Hypothesis D: Multi-morbidity Stacking"] = {
        "dims": 206,
        "accuracy_mean": float(np.mean(acc_l)),
        "accuracy_std": float(np.std(acc_l)),
        "precision_mean": float(np.mean(p_l)),
        "precision_std": float(np.std(p_l)),
        "recall_mean": float(np.mean(r_l)),
        "recall_std": float(np.std(r_l)),
        "balanced_accuracy_mean": float(np.mean(bacc_l)),
        "balanced_accuracy_std": float(np.std(bacc_l)),
        "f1_mean": float(np.mean(f1_l)),
        "f1_std": float(np.std(f1_l)),
        "threshold_mean": float(np.mean(tau_l)),
        "elapsed": time.time() - t0,
    }
    res = results["Hypothesis D: Multi-morbidity Stacking"]
    print(
        f"  -> Val F1: {res['f1_mean']:.4f} +/- {res['f1_std']:.4f} (Prec: {res['precision_mean']:.4f}, Rec: {res['recall_mean']:.4f}, Time: {res['elapsed']:.2f}s)"
    )

    # Save to report/hypotheses_benchmark.json
    os.makedirs("report", exist_ok=True)
    with open("report/hypotheses_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nBenchmark completed. Results saved to report/hypotheses_benchmark.json")

    print("\n" + "=" * 115)
    print(
        f"{'Hypothesis':<42} | {'Dims':<5} | {'Val Acc':<8} | {'Prec':<7} | {'Rec':<7} | {'BAcc':<7} | {'Val F1 (+1)':<18} | {'Time':<6}"
    )
    print("-" * 115)
    for name, r in results.items():
        print(
            f"{name:<42} | {r['dims']:<5} | {r['accuracy_mean']:.4f}   | "
            f"{r['precision_mean']:.4f}  | {r['recall_mean']:.4f}  | "
            f"{r['balanced_accuracy_mean']:.4f}  | {r['f1_mean']:.4f} +/- {r['f1_std']:.4f} | "
            f"{r['elapsed']:.1f}s"
        )
    print("=" * 115)


if __name__ == "__main__":
    run_hypotheses_cv()
