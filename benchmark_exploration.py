"""Micro-benchmark exploration script to empirically validate Hypotheses 1-4.

Strictly pure Python standard library + NumPy only.

Evaluates on a stratified train/validation split:
1. Hypothesis 1: Cost-Sensitive Hessian & Gradient in Histogram GBM (w_pos in {1.0, 1.5, 2.0, 2.5, 3.0}).
2. Hypothesis 2: Domain Composite Vascular Features entering tree_features.
3. Hypothesis 3: Tri-Model Heterogeneous Rank Ensembling (GBM + MLP + Weighted Ridge).
4. Hypothesis 4: Fine-Grained Quantile Cutoff Calibration (P* in [0.120, 0.160]).
"""

import argparse
import os
import sys
import time

import numpy as np

import gbm
import mlp
from cross_validation import weighted_ridge_regression
from metrics import precision_recall_f1, predict_labels
from preprocessing import (
    clean_data,
    fit_preprocessor,
    fit_standardizer,
    load_data_cached,
    load_feature_names,
    standardize,
    transform,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def rank_normalize(scores):
    """Convert continuous decision scores into uniform percentile ranks in [0, 1]."""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n <= 1:
        return np.zeros_like(scores)
    ranks = np.empty(n, dtype=np.float64)
    ranks[np.argsort(scores)] = np.linspace(0.0, 1.0, n)
    return ranks


def f1_at_top_rate(y, scores, top_rate):
    """Compute (precision, recall, f1) when top top_rate scores are predicted +1."""
    thr = np.quantile(scores, 1.0 - top_rate)
    y_pred = predict_labels(scores, thr)
    return precision_recall_f1(y, y_pred)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=os.path.join(HERE, "data", "dataset"))
    parser.add_argument("--n_rows", type=int, default=60000)
    parser.add_argument("--val_frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    print("=" * 80, flush=True)
    print(
        f"CS-433 ML Project 1: Empirical Hypothesis Exploration Benchmark", flush=True
    )
    print(
        f"Sample Size: {args.n_rows} rows | Validation Split: {args.val_frac:.0%}",
        flush=True,
    )
    print("=" * 80, flush=True)

    # 1. Load Data
    t0 = time.time()
    x_raw, _, y_raw, _, _ = load_data_cached(args.data_path)
    names_raw = load_feature_names(args.data_path)

    if args.n_rows is not None and args.n_rows < len(y_raw):
        # Stratified sampling of n_rows
        rng = np.random.default_rng(args.seed)
        pos_idx = np.where(y_raw == 1)[0]
        neg_idx = np.where(y_raw == -1)[0]
        n_pos = int(round(args.n_rows * len(pos_idx) / len(y_raw)))
        n_neg = args.n_rows - n_pos
        sel = np.sort(
            np.concatenate(
                [
                    rng.choice(pos_idx, n_pos, replace=False),
                    rng.choice(neg_idx, n_neg, replace=False),
                ]
            )
        )
        x_raw = x_raw[sel]
        y_raw = y_raw[sel]

    # Stratified Train / Validation Split
    rng = np.random.default_rng(args.seed)
    pos_idx = np.where(y_raw == 1)[0]
    neg_idx = np.where(y_raw == -1)[0]
    val_pos = rng.choice(
        pos_idx, int(round(args.val_frac * len(pos_idx))), replace=False
    )
    val_neg = rng.choice(
        neg_idx, int(round(args.val_frac * len(neg_idx))), replace=False
    )
    val_idx = np.sort(np.concatenate([val_pos, val_neg]))
    tr_idx = np.setdiff1d(np.arange(len(y_raw)), val_idx)

    y_tr, y_val = y_raw[tr_idx], y_raw[val_idx]
    print(
        f"Data Loaded: Train N={len(tr_idx)} ({np.mean(y_tr == 1):.2%} pos), "
        f"Val N={len(val_idx)} ({np.mean(y_val == 1):.2%} pos) in {time.time() - t0:.2f}s",
        flush=True,
    )

    # Clean data without composite features
    xc_base, names_base = clean_data(
        x_raw, names_raw, keep_raw=("_STATE",), with_composite=False
    )
    # Clean data with composite features
    xc_comp, names_comp = clean_data(
        x_raw, names_raw, keep_raw=("_STATE",), with_composite=True
    )

    print("-" * 80, flush=True)
    print(
        "HYPOTHESIS 1: Cost-Sensitive Hessian & Gradient in Histogram GBM", flush=True
    )
    print("-" * 80, flush=True)
    print(
        "| w_pos | Trees | Precision | Recall | F1 Score | Delta F1 | Time (s) |",
        flush=True,
    )
    print(
        "|-------|-------|-----------|--------|----------|----------|----------|",
        flush=True,
    )

    f1_w1 = None
    best_wpos = 1.0
    best_f1_gbm = 0.0

    w_pos_candidates = [1.0, 1.5, 2.0, 2.5, 3.0]
    gbm_val_scores = {}

    for w_p in w_pos_candidates:
        t_gbm = time.time()
        p = dict(gbm.DEFAULT_PARAMS, w_pos=w_p, seed=args.seed)
        tx_tr = gbm.tree_features(xc_base[tr_idx], names_base)
        tx_val = gbm.tree_features(xc_base[val_idx], names_base)

        model_gbm = gbm.train(y_tr, tx_tr, p)
        s_val = gbm.score(model_gbm, tx_val)
        gbm_val_scores[w_p] = s_val

        prec, rec, f1 = f1_at_top_rate(y_val, s_val, 0.14)
        if f1_w1 is None:
            f1_w1 = f1
        delta = f1 - f1_w1
        if f1 > best_f1_gbm:
            best_f1_gbm = f1
            best_wpos = w_p

        elapsed = time.time() - t_gbm
        n_trees = len(model_gbm["trees"])
        print(
            f"| {w_p:5.1f} | {n_trees:5d} | {prec:9.4f} | {rec:6.4f} | {f1:8.4f} | {delta:+8.4f} | {elapsed:8.2f} |",
            flush=True,
        )

    print(f"\nWinning GBM w_pos: {best_wpos} (F1 = {best_f1_gbm:.4f})\n", flush=True)

    print("-" * 80, flush=True)
    print(
        "HYPOTHESIS 2: Domain Composite Vascular Features for Depth-5 Trees", flush=True
    )
    print("-" * 80, flush=True)
    t_comp = time.time()
    p_comp = dict(gbm.DEFAULT_PARAMS, w_pos=best_wpos, seed=args.seed)
    tx_tr_comp = gbm.tree_features(xc_comp[tr_idx], names_comp)
    tx_val_comp = gbm.tree_features(xc_comp[val_idx], names_comp)

    model_comp = gbm.train(y_tr, tx_tr_comp, p_comp)
    s_val_comp = gbm.score(model_comp, tx_val_comp)
    prec_comp, rec_comp, f1_comp = f1_at_top_rate(y_val, s_val_comp, 0.14)
    elapsed_comp = time.time() - t_comp
    delta_comp = f1_comp - best_f1_gbm

    print(
        f"Base Features (D={tx_tr.shape[1]}):       F1 = {best_f1_gbm:.4f}", flush=True
    )
    print(
        f"With Composite (D={tx_tr_comp.shape[1]}):   F1 = {f1_comp:.4f} (P={prec_comp:.4f}, R={rec_comp:.4f}, Delta={delta_comp:+.4f}) in {elapsed_comp:.2f}s",
        flush=True,
    )

    winning_xc = xc_comp if f1_comp >= best_f1_gbm else xc_base
    winning_names = names_comp if f1_comp >= best_f1_gbm else names_base
    winning_s_gbm = s_val_comp if f1_comp >= best_f1_gbm else gbm_val_scores[best_wpos]

    print("-" * 80, flush=True)
    print("HYPOTHESIS 3: Tri-Model Heterogeneous Rank Ensembling", flush=True)
    print("-" * 80, flush=True)

    # Train Model 2: Pure-NumPy MLP (5 seeds for speed in exploration)
    print(
        "Training Model 2: Pure-NumPy Multi-Layer Perceptron (ReLU, 5 seeds)...",
        flush=True,
    )
    t_mlp = time.time()
    mlp_cfg = {"interactions": False}
    state_mlp = fit_preprocessor(
        winning_xc[tr_idx, :-1], winning_names[:-1], mlp_cfg, y=y_tr
    )
    tx_tr_mlp = transform(winning_xc[tr_idx, :-1], state_mlp)
    tx_val_mlp = transform(winning_xc[val_idx, :-1], state_mlp)

    mlp_params = dict(mlp.DEFAULT_PARAMS, n_seeds=5)
    model_mlp = mlp.train(y_tr, tx_tr_mlp, mlp_params)
    s_val_mlp = mlp.score(model_mlp, tx_val_mlp)
    prec_mlp, rec_mlp, f1_mlp = f1_at_top_rate(y_val, s_val_mlp, 0.14)
    print(
        f"MLP (5 seeds) Validation F1: {f1_mlp:.4f} (P={prec_mlp:.4f}, R={rec_mlp:.4f}) in {time.time() - t_mlp:.2f}s",
        flush=True,
    )

    # Train Model 3: Cost-Sensitive Weighted Ridge
    print(
        "Training Model 3: Cost-Sensitive Weighted Ridge (lambda=1e-4, w_pos=3.0)...",
        flush=True,
    )
    t_ridge = time.time()
    w_ridge, _ = weighted_ridge_regression(
        y_tr.astype(float), tx_tr_mlp, lambda_=1e-4, w_pos=3.0
    )
    s_val_ridge = tx_val_mlp @ w_ridge
    prec_ridge, rec_ridge, f1_ridge = f1_at_top_rate(y_val, s_val_ridge, 0.14)
    print(
        f"Weighted Ridge Validation F1: {f1_ridge:.4f} (P={prec_ridge:.4f}, R={rec_ridge:.4f}) in {time.time() - t_ridge:.2f}s",
        flush=True,
    )

    # Baseline 2-Model Logit Blend
    s_base_blend = 0.5 * winning_s_gbm + 0.5 * s_val_mlp
    prec_base, rec_base, f1_base_blend = f1_at_top_rate(y_val, s_base_blend, 0.14)
    print(
        f"\nBaseline 2-Model Logit Blend (0.5 GBM + 0.5 MLP): F1 = {f1_base_blend:.4f} (P={prec_base:.4f}, R={rec_base:.4f})",
        flush=True,
    )

    # Rank Transform
    r_gbm = rank_normalize(winning_s_gbm)
    r_mlp = rank_normalize(s_val_mlp)
    r_ridge = rank_normalize(s_val_ridge)

    # Rank-based 2-Model Blend
    s_rank_2m = 0.5 * r_gbm + 0.5 * r_mlp
    _, _, f1_rank_2m = f1_at_top_rate(y_val, s_rank_2m, 0.14)
    print(
        f"2-Model Rank Blend (0.5 Rank_GBM + 0.5 Rank_MLP):     F1 = {f1_rank_2m:.4f}",
        flush=True,
    )

    # Tri-Model Grid Search
    print("\nGrid Search Tri-Model Blends:", flush=True)
    print(
        "| alpha (GBM) | beta (MLP) | gamma (Ridge) | Precision | Recall | F1 Score | Delta vs 2M |",
        flush=True,
    )
    print(
        "|-------------|------------|---------------|-----------|--------|----------|-------------|",
        flush=True,
    )

    best_tri_f1 = 0.0
    best_weights = None
    best_tri_scores = None

    for alpha in [0.45, 0.50, 0.55, 0.60]:
        for beta in [0.25, 0.30, 0.35]:
            gamma = round(1.0 - alpha - beta, 4)
            if gamma < 0.05 or gamma > 0.25:
                continue
            s_tri = alpha * r_gbm + beta * r_mlp + gamma * r_ridge
            p_tri, r_tri, f1_tri = f1_at_top_rate(y_val, s_tri, 0.14)
            d_f1 = f1_tri - f1_base_blend
            if f1_tri > best_tri_f1:
                best_tri_f1 = f1_tri
                best_weights = (alpha, beta, gamma)
                best_tri_scores = s_tri
            print(
                f"| {alpha:11.2f} | {beta:10.2f} | {gamma:13.2f} | {p_tri:9.4f} | {r_tri:6.4f} | {f1_tri:8.4f} | {d_f1:+11.4f} |",
                flush=True,
            )

    print(
        f"\nBest Tri-Model Config: alpha={best_weights[0]}, beta={best_weights[1]}, gamma={best_weights[2]} -> F1 = {best_tri_f1:.4f} (Delta={best_tri_f1 - f1_base_blend:+.4f})",
        flush=True,
    )

    print("-" * 80, flush=True)
    print("HYPOTHESIS 4: Fine-Grained Quantile Cutoff Calibration", flush=True)
    print("-" * 80, flush=True)
    print("| Cutoff P* | Precision | Recall | F1 Score |", flush=True)
    print("|-----------|-----------|--------|----------|", flush=True)

    scores_to_tune = best_tri_scores if best_tri_f1 > f1_base_blend else s_base_blend
    best_rate = 0.14
    best_rate_f1 = 0.0

    for rate in np.arange(0.120, 0.162, 0.002):
        r_f = round(float(rate), 4)
        p_c, r_c, f_c = f1_at_top_rate(y_val, scores_to_tune, r_f)
        if f_c > best_rate_f1:
            best_rate_f1 = f_c
            best_rate = r_f
        marker = " <-- PEAK" if abs(r_f - best_rate) < 1e-5 else ""
        print(
            f"| {r_f:9.3%} | {p_c:9.4f} | {r_c:6.4f} | {f_c:8.4f} |{marker}", flush=True
        )

    print(
        f"\nOptimal Cutoff: P* = {best_rate:.3%} (Validation F1 = {best_rate_f1:.4f}, Delta vs 14.0% = {best_rate_f1 - best_tri_f1:+.4f})",
        flush=True,
    )

    print("=" * 80, flush=True)
    print(f"Benchmark Exploration Complete in {time.time() - t_start:.2f}s", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
