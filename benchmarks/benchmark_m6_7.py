import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Milestone 6.7: Non-Linear & Probability Calibration 5-Fold Stratified Cross-Validation Benchmark.

Evaluates on the 233-dimensional feature space:
1. Baseline: 233-dim Super Tri-Blend via Uniform Percentile Rank Normalization (Milestone 6.6)
2. Innovation 1: 233-dim Super Tri-Blend via Pure-NumPy Isotonic PAVA Probability Calibration
3. Innovation 2: 4-Model Quad-Blend adding Pure-NumPy FastHistBoostedTrees (Depth=2)
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


class IsotonicCalibrator:
    """Non-parametric monotonic probability calibration via Pool Adjacent Violators Algorithm (PAVA)."""

    def __init__(self):
        self.knots_s = None
        self.knots_p = None

    def fit(self, scores, y):
        """Fit a monotonically increasing step function from continuous scores to probabilities."""
        scores = np.asarray(scores, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        order = np.argsort(scores)
        sorted_s = scores[order]
        sorted_y = y[order]

        # Vectorized/stack PAVA
        stack = []  # stores [block_weight, block_weighted_y]
        for yi in sorted_y:
            cur_w = 1.0
            cur_wy = yi
            while stack and (stack[-1][1] / stack[-1][0]) >= (cur_wy / cur_w):
                prev_w, prev_wy = stack.pop()
                cur_w += prev_w
                cur_wy += prev_wy
            stack.append((cur_w, cur_wy))

        # Construct knot coordinates for monotonic piecewise linear interpolation
        knots_s = []
        knots_p = []
        cum_idx = 0
        for block_w, block_wy in stack:
            n_samples = int(round(block_w))
            block_s = np.mean(sorted_s[cum_idx : cum_idx + n_samples])
            knots_s.append(block_s)
            knots_p.append(block_wy / block_w)
            cum_idx += n_samples

        self.knots_s = np.array(knots_s, dtype=np.float64)
        self.knots_p = np.array(knots_p, dtype=np.float64)
        return self

    def predict_proba(self, scores):
        """Predict calibrated probabilities in [0, 1] via monotonic interpolation."""
        scores = np.asarray(scores, dtype=np.float64)
        p = np.interp(
            scores,
            self.knots_s,
            self.knots_p,
            left=self.knots_p[0],
            right=self.knots_p[-1],
        )
        return np.clip(p, 0.0, 1.0)


class FastHistBoostedTrees:
    """Fast histogram-based gradient boosted decision trees (Depth=2) in pure NumPy."""

    def __init__(
        self,
        n_estimators=35,
        learning_rate=0.15,
        l2_reg=1.0,
        n_bins=32,
        max_features=50,
        w_pos=2.5,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.n_bins = n_bins
        self.max_features = max_features
        self.w_pos = w_pos
        self.trees = []
        self.bin_edges = []
        self.feature_indices = None
        self.base_score = 0.0

    def fit(self, X, y):
        N, D = X.shape
        y01 = y.astype(np.float64)
        sample_weights = np.where(y01 == 1.0, self.w_pos, 1.0)

        # Select top features by variance
        if self.max_features < D:
            var = np.var(X, axis=0)
            self.feature_indices = np.argsort(var)[-self.max_features :]
            X_sub = X[:, self.feature_indices]
        else:
            self.feature_indices = np.arange(D)
            X_sub = X

        K = len(self.feature_indices)
        self.bin_edges = []
        X_binned = np.empty((N, K), dtype=np.uint8)
        percentiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]

        for j in range(K):
            edges = np.unique(np.percentile(X_sub[:, j], percentiles))
            self.bin_edges.append(edges)
            X_binned[:, j] = np.digitize(X_sub[:, j], edges).astype(np.uint8)

        p_mean = np.clip(
            np.sum(sample_weights * y01) / np.sum(sample_weights), 1e-4, 1 - 1e-4
        )
        self.base_score = float(np.log(p_mean / (1.0 - p_mean)))
        F = np.full(N, self.base_score, dtype=np.float64)
        self.trees = []
        lam = self.l2_reg

        for t in range(self.n_estimators):
            p = 1.0 / (1.0 + np.exp(-F))
            g = sample_weights * (p - y01)
            h = sample_weights * np.maximum(p * (1.0 - p), 1e-6)

            best_feat, best_bin, best_gain = 0, 0, -1.0
            G_tot, H_tot = np.sum(g), np.sum(h)
            root_score = (G_tot**2) / (H_tot + lam)

            for j in range(K):
                bj = X_binned[:, j]
                n_b = len(self.bin_edges[j]) + 1
                g_b = np.bincount(bj, weights=g, minlength=n_b)
                h_b = np.bincount(bj, weights=h, minlength=n_b)
                G_L = np.cumsum(g_b[:-1])
                H_L = np.cumsum(h_b[:-1])
                G_R = G_tot - G_L
                H_R = H_tot - H_L
                gains = 0.5 * (
                    (G_L**2) / (H_L + lam) + (G_R**2) / (H_R + lam) - root_score
                )
                max_g = np.max(gains)
                if max_g > best_gain:
                    best_gain, best_feat, best_bin = (
                        max_g,
                        j,
                        int(np.argmax(gains)),
                    )

            left_mask = X_binned[:, best_feat] <= best_bin
            right_mask = ~left_mask

            def find_child_split(mask):
                if np.sum(mask) < 20:
                    g_sum, h_sum = np.sum(g[mask]), np.sum(h[mask])
                    w = -g_sum / (h_sum + lam)
                    return None, None, w, w
                g_m, h_m = g[mask], h[mask]
                G_c, H_c = np.sum(g_m), np.sum(h_m)
                score_c = (G_c**2) / (H_c + lam)
                c_feat, c_bin, c_gain = 0, 0, -1.0
                for j in range(K):
                    bj = X_binned[mask, j]
                    n_b = len(self.bin_edges[j]) + 1
                    g_b = np.bincount(bj, weights=g_m, minlength=n_b)
                    h_b = np.bincount(bj, weights=h_m, minlength=n_b)
                    G_L = np.cumsum(g_b[:-1])
                    H_L = np.cumsum(h_b[:-1])
                    G_R = G_c - G_L
                    H_R = H_c - H_L
                    gains = 0.5 * (
                        (G_L**2) / (H_L + lam) + (G_R**2) / (H_R + lam) - score_c
                    )
                    max_g = np.max(gains)
                    if max_g > c_gain:
                        c_gain, c_feat, c_bin = (
                            max_g,
                            j,
                            int(np.argmax(gains)),
                        )
                l_m = X_binned[mask, c_feat] <= c_bin
                r_m = ~l_m
                w_L = (
                    -np.sum(g_m[l_m]) / (np.sum(h_m[l_m]) + lam)
                    if np.sum(l_m) > 0
                    else 0.0
                )
                w_R = (
                    -np.sum(g_m[r_m]) / (np.sum(h_m[r_m]) + lam)
                    if np.sum(r_m) > 0
                    else 0.0
                )
                return c_feat, c_bin, w_L, w_R

            j_L, b_L, w_LL, w_LR = find_child_split(left_mask)
            j_R, b_R, w_RL, w_RR = find_child_split(right_mask)

            update = np.zeros(N, dtype=np.float64)
            if j_L is not None:
                m_LL = left_mask & (X_binned[:, j_L] <= b_L)
                m_LR = left_mask & (~(X_binned[:, j_L] <= b_L))
                update[m_LL], update[m_LR] = w_LL, w_LR
            else:
                update[left_mask] = w_LL
            if j_R is not None:
                m_RL = right_mask & (X_binned[:, j_R] <= b_R)
                m_RR = right_mask & (~(X_binned[:, j_R] <= b_R))
                update[m_RL], update[m_RR] = w_RL, w_RR
            else:
                update[right_mask] = w_RL

            F += self.learning_rate * update
            self.trees.append(
                {
                    "root_feat": best_feat,
                    "root_bin": best_bin,
                    "j_L": j_L,
                    "b_L": b_L,
                    "w_LL": w_LL,
                    "w_LR": w_LR,
                    "j_R": j_R,
                    "b_R": b_R,
                    "w_RL": w_RL,
                    "w_RR": w_RR,
                }
            )
        return self

    def predict_raw(self, X):
        X_sub = X[:, self.feature_indices]
        N, K = X_sub.shape
        X_binned = np.empty((N, K), dtype=np.uint8)
        for j in range(K):
            X_binned[:, j] = np.digitize(X_sub[:, j], self.bin_edges[j]).astype(
                np.uint8
            )

        F = np.full(N, self.base_score, dtype=np.float64)
        for tree in self.trees:
            update = np.zeros(N, dtype=np.float64)
            left_mask = X_binned[:, tree["root_feat"]] <= tree["root_bin"]
            right_mask = ~left_mask
            if tree["j_L"] is not None:
                m_LL = left_mask & (X_binned[:, tree["j_L"]] <= tree["b_L"])
                m_LR = left_mask & (~(X_binned[:, tree["j_L"]] <= tree["b_L"]))
                update[m_LL], update[m_LR] = tree["w_LL"], tree["w_LR"]
            else:
                update[left_mask] = tree["w_LL"]
            if tree["j_R"] is not None:
                m_RL = right_mask & (X_binned[:, tree["j_R"]] <= tree["b_R"])
                m_RR = right_mask & (~(X_binned[:, tree["j_R"]] <= tree["b_R"]))
                update[m_RL], update[m_RR] = tree["w_RL"], tree["w_RR"]
            else:
                update[right_mask] = tree["w_RL"]
            F += self.learning_rate * update
        return F


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


def run_benchmark_m6_7():
    print("=" * 115)
    print(
        "Milestone 6.7: Non-Linear & Isotonic Calibration 5-Fold Stratified CV Benchmark"
    )
    print("=" * 115)

    with open("data/processed/features_selected.json") as f:
        feat_names = json.load(f)

    x = np.load("data/processed/x_train.npy")
    y = np.load("data/processed/y_train.npy")
    k_indices = build_k_indices(y, k_fold=5, seed=SEED)

    print(f"Dataset: {x.shape[0]} samples, {len(feat_names)} raw features.")
    print("Evaluating 5 stratified folds with zero data leakage...\n")

    metrics_rank = {
        "f1": [],
        "prec": [],
        "rec": [],
        "bacc": [],
        "acc": [],
        "tau": [],
    }
    metrics_pava = {
        "f1": [],
        "prec": [],
        "rec": [],
        "bacc": [],
        "acc": [],
        "tau": [],
    }
    metrics_quad = {
        "f1": [],
        "prec": [],
        "rec": [],
        "bacc": [],
        "acc": [],
        "tau": [],
    }

    t_start = time.time()

    for k in range(5):
        t_fold = time.time()
        val_idx = k_indices[k]
        tr_idx = np.concatenate([k_indices[i] for i in range(5) if i != k])
        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_val, y_val = x[val_idx], y[val_idx]
        y_tr_01 = (y_tr == 1).astype(float)

        # 1. Feature Engineering (233 dimensions)
        tx_tr, tx_val, out_names = build_advanced_features(
            x_tr, x_val, feature_names=feat_names, y_tr=y_tr, alpha=20.0
        )
        assert tx_tr.shape[1] == 233, f"Expected 233 dims, got {tx_tr.shape[1]}"

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
        w_soft, _ = train_soft_f1_optimizer(
            tx_tr, y_tr_01, max_iters=250, lr=0.03, lambda_reg=1e-5
        )
        s3_tr = sigmoid(tx_tr @ w_soft)
        s3_val = sigmoid(tx_val @ w_soft)

        # Model 4: FastHistBoostedTrees (Depth=2)
        tree = FastHistBoostedTrees(
            n_estimators=35,
            learning_rate=0.15,
            l2_reg=1.0,
            n_bins=32,
            max_features=50,
            w_pos=2.5,
        )
        tree.fit(tx_tr, y_tr_01)
        s4_tr = tree.predict_raw(tx_tr)
        s4_val = tree.predict_raw(tx_val)

        # =====================================================================
        # 1. Baseline: Percentile Rank Normalization Blend (M6.6)
        # =====================================================================
        r1_tr, r1_val = rank_normalize(s1_tr), rank_normalize(s1_val)
        r2_tr, r2_val = rank_normalize(s2_tr), rank_normalize(s2_val)
        r3_tr, r3_val = rank_normalize(s3_tr), rank_normalize(s3_val)

        blend_rank_tr = 0.35 * r1_tr + 0.40 * r2_tr + 0.25 * r3_tr
        blend_rank_val = 0.35 * r1_val + 0.40 * r2_val + 0.25 * r3_val

        tau_r, _ = best_threshold(y_tr, blend_rank_tr, np.linspace(0.65, 0.95, 61))
        pred_r = predict_labels(blend_rank_val, tau_r)
        p_r, rec_r, f1_r = precision_recall_f1(y_val, pred_r)
        bacc_r = confusion_matrix(y_val, pred_r)["balanced_accuracy"]
        acc_r = accuracy(y_val, pred_r)

        metrics_rank["f1"].append(f1_r)
        metrics_rank["prec"].append(p_r)
        metrics_rank["rec"].append(rec_r)
        metrics_rank["bacc"].append(bacc_r)
        metrics_rank["acc"].append(acc_r)
        metrics_rank["tau"].append(tau_r)

        # =====================================================================
        # 2. Innovation 1: Pure-NumPy Isotonic PAVA Probability Calibration
        # =====================================================================
        cal1 = IsotonicCalibrator().fit(s1_tr, y_tr_01)
        cal2 = IsotonicCalibrator().fit(s2_tr, y_tr_01)
        cal3 = IsotonicCalibrator().fit(s3_tr, y_tr_01)
        cal4 = IsotonicCalibrator().fit(s4_tr, y_tr_01)

        p1_tr, p1_val = cal1.predict_proba(s1_tr), cal1.predict_proba(s1_val)
        p2_tr, p2_val = cal2.predict_proba(s2_tr), cal2.predict_proba(s2_val)
        p3_tr, p3_val = cal3.predict_proba(s3_tr), cal3.predict_proba(s3_val)
        p4_tr, p4_val = cal4.predict_proba(s4_tr), cal4.predict_proba(s4_val)

        blend_pava_tr = 0.35 * p1_tr + 0.40 * p2_tr + 0.25 * p3_tr
        blend_pava_val = 0.35 * p1_val + 0.40 * p2_val + 0.25 * p3_val

        tau_pava, _ = best_threshold(y_tr, blend_pava_tr, np.linspace(0.10, 0.50, 81))
        pred_pava = predict_labels(blend_pava_val, tau_pava)
        p_p, rec_p, f1_p = precision_recall_f1(y_val, pred_pava)
        bacc_p = confusion_matrix(y_val, pred_pava)["balanced_accuracy"]
        acc_p = accuracy(y_val, pred_pava)

        metrics_pava["f1"].append(f1_p)
        metrics_pava["prec"].append(p_p)
        metrics_pava["rec"].append(rec_p)
        metrics_pava["bacc"].append(bacc_p)
        metrics_pava["acc"].append(acc_p)
        metrics_pava["tau"].append(tau_pava)

        # =====================================================================
        # 3. Innovation 2: 4-Model Calibrated Quad-Blend (Adding Boosted Stumps)
        # =====================================================================
        blend_quad_tr = 0.30 * p1_tr + 0.35 * p2_tr + 0.20 * p3_tr + 0.15 * p4_tr
        blend_quad_val = 0.30 * p1_val + 0.35 * p2_val + 0.20 * p3_val + 0.15 * p4_val

        tau_quad, _ = best_threshold(y_tr, blend_quad_tr, np.linspace(0.10, 0.50, 81))
        pred_quad = predict_labels(blend_quad_val, tau_quad)
        p_q, rec_q, f1_q = precision_recall_f1(y_val, pred_quad)
        bacc_q = confusion_matrix(y_val, pred_quad)["balanced_accuracy"]
        acc_q = accuracy(y_val, pred_quad)

        metrics_quad["f1"].append(f1_q)
        metrics_quad["prec"].append(p_q)
        metrics_quad["rec"].append(rec_q)
        metrics_quad["bacc"].append(bacc_q)
        metrics_quad["acc"].append(acc_q)
        metrics_quad["tau"].append(tau_quad)

        print(
            f"Fold {k+1}/5 ({time.time() - t_fold:.1f}s) | "
            f"Rank-Blend F1={f1_r:.4f} | "
            f"Isotonic-PAVA F1={f1_p:.4f} (Prec={p_p:.4f}, Rec={rec_p:.4f}) | "
            f"Quad-Blend F1={f1_q:.4f} (Prec={p_q:.4f}, Rec={rec_q:.4f})"
        )

    total_time = time.time() - t_start

    results = {
        "Rank_Blend_Tri_Blend": {
            "f1_mean": float(np.mean(metrics_rank["f1"])),
            "f1_std": float(np.std(metrics_rank["f1"])),
            "precision_mean": float(np.mean(metrics_rank["prec"])),
            "precision_std": float(np.std(metrics_rank["prec"])),
            "recall_mean": float(np.mean(metrics_rank["rec"])),
            "recall_std": float(np.std(metrics_rank["rec"])),
            "balanced_accuracy_mean": float(np.mean(metrics_rank["bacc"])),
            "balanced_accuracy_std": float(np.std(metrics_rank["bacc"])),
            "accuracy_mean": float(np.mean(metrics_rank["acc"])),
            "accuracy_std": float(np.std(metrics_rank["acc"])),
            "threshold_mean": float(np.mean(metrics_rank["tau"])),
        },
        "Isotonic_PAVA_Tri_Blend": {
            "f1_mean": float(np.mean(metrics_pava["f1"])),
            "f1_std": float(np.std(metrics_pava["f1"])),
            "precision_mean": float(np.mean(metrics_pava["prec"])),
            "precision_std": float(np.std(metrics_pava["prec"])),
            "recall_mean": float(np.mean(metrics_pava["rec"])),
            "recall_std": float(np.std(metrics_pava["rec"])),
            "balanced_accuracy_mean": float(np.mean(metrics_pava["bacc"])),
            "balanced_accuracy_std": float(np.std(metrics_pava["bacc"])),
            "accuracy_mean": float(np.mean(metrics_pava["acc"])),
            "accuracy_std": float(np.std(metrics_pava["acc"])),
            "threshold_mean": float(np.mean(metrics_pava["tau"])),
        },
        "Calibrated_Quad_Blend": {
            "f1_mean": float(np.mean(metrics_quad["f1"])),
            "f1_std": float(np.std(metrics_quad["f1"])),
            "precision_mean": float(np.mean(metrics_quad["prec"])),
            "precision_std": float(np.std(metrics_quad["prec"])),
            "recall_mean": float(np.mean(metrics_quad["rec"])),
            "recall_std": float(np.std(metrics_quad["rec"])),
            "balanced_accuracy_mean": float(np.mean(metrics_quad["bacc"])),
            "balanced_accuracy_std": float(np.std(metrics_quad["bacc"])),
            "accuracy_mean": float(np.mean(metrics_quad["acc"])),
            "accuracy_std": float(np.std(metrics_quad["acc"])),
            "threshold_mean": float(np.mean(metrics_quad["tau"])),
        },
        "total_time_seconds": float(total_time),
    }

    print("\n" + "=" * 115)
    print("5-Fold Cross-Validation Comparison Summary:")
    print("-" * 115)
    res_r = results["Rank_Blend_Tri_Blend"]
    res_p = results["Isotonic_PAVA_Tri_Blend"]
    res_q = results["Calibrated_Quad_Blend"]
    print(
        f"1. Rank Blend Tri-Blend:     Val F1={res_r['f1_mean']:.4f} +/- {res_r['f1_std']:.4f} | "
        f"Prec={res_r['precision_mean']:.4f} | Rec={res_r['recall_mean']:.4f} | BAcc={res_r['balanced_accuracy_mean']:.4f} | Acc={res_r['accuracy_mean']:.4f}"
    )
    print(
        f"2. Isotonic PAVA Tri-Blend:  Val F1={res_p['f1_mean']:.4f} +/- {res_p['f1_std']:.4f} | "
        f"Prec={res_p['precision_mean']:.4f} | Rec={res_p['recall_mean']:.4f} | BAcc={res_p['balanced_accuracy_mean']:.4f} | Acc={res_p['accuracy_mean']:.4f}"
    )
    print(
        f"3. Calibrated Quad-Blend:    Val F1={res_q['f1_mean']:.4f} +/- {res_q['f1_std']:.4f} | "
        f"Prec={res_q['precision_mean']:.4f} | Rec={res_q['recall_mean']:.4f} | BAcc={res_q['balanced_accuracy_mean']:.4f} | Acc={res_q['accuracy_mean']:.4f}"
    )
    print("=" * 115)

    os.makedirs("report", exist_ok=True)
    with open("report/m6_7_calibration_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved report/m6_7_calibration_benchmark.json successfully.\n")

    return results


if __name__ == "__main__":
    run_benchmark_m6_7()
