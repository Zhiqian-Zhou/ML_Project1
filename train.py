"""Train the final ensemble models from scratch and export portable weights.

This script executes the complete training pipeline:
    1. Loads raw training data and codebook features.
    2. Performs codebook-driven cleaning, missingness imputation and encoding.
    3. Fits empirical Bayes target risk encodings on key categorical variables.
    4. Trains a 3-seed Histogram Gradient Boosted Tree ensemble (NumPy).
    5. Fits the neural network preprocessor and trains a 10-seed MLP ensemble.
    6. Saves the complete fitted models and preprocessing state to weights/.
    7. Optionally generates the exact competition submission CSV via inference.

Usage:
    # 1. Train Clean SOTA (Track A, zero leakage)
    python train.py --output weights/clean_model.npz

    # 2. Train Leaky Model (Track B, leaderboard exploration)
    python train.py --with_leakage --output weights/leaky_model.npz
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import time

import numpy as np

import gbm
from helpers import create_csv_submission
from metrics import predict_labels
from model_io import save_model
import mlp
from preprocessing import (
    TARGET_ENC_COLUMNS,
    clean_data,
    fit_preprocessor,
    fit_standardizer,
    leaky_features,
    load_data_cached,
    load_feature_names,
    standardize,
    transform,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN_RATE = 0.136  # optimal operating point for Clean SOTA (F1 = 0.447)
LEAKY_RATE = 0.128  # optimal operating point for Leaky SOTA (F1 = 0.457)
MLP_CONFIG = {"interactions": False}


def parse_args(argv=None):
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=os.path.join(HERE, "data", "dataset"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--with_leakage", action="store_true")
    parser.add_argument("--top_rate", type=float, default=None)
    parser.add_argument("--gbm_weight", type=float, default=0.5)
    parser.add_argument("--n_jobs", type=int, default=3)
    parser.add_argument("--submission_output", default=None)
    args = parser.parse_args(argv)

    if args.top_rate is None:
        args.top_rate = LEAKY_RATE if args.with_leakage else CLEAN_RATE
    if args.output is None:
        weights_dir = os.path.join(HERE, "weights")
        name = "leaky_model.npz" if args.with_leakage else "clean_model.npz"
        args.output = os.path.join(weights_dir, name)
    if args.submission_output is None:
        sub_dir = os.path.join(HERE, "submissions")
        name = (
            "submission_with_leakage.csv"
            if args.with_leakage
            else "submission_clean_sota.csv"
        )
        args.submission_output = os.path.join(sub_dir, name)
    return args


def log(t0, message):
    """Print timestamped progress message."""
    print(f"[{time.time() - t0:6.0f}s] {message}", flush=True)


def model_inputs(x_raw, names, with_leakage=False):
    """Clean raw features and isolate post-event leaky block if requested."""
    xc, xc_names = clean_data(x_raw, names, keep_raw=("_STATE",), with_composite=True)
    if not with_leakage:
        return xc, xc_names, None
    x_tree, _, x_flags, _ = leaky_features(x_raw, names)
    return xc, xc_names, (x_tree, x_flags)


def fit_target_risk_encodings(
    y_tr, xc_tr, names, target_cols=TARGET_ENC_COLUMNS, m_smooth=50.0
):
    """Compute empirical Bayes smoothed risk encoding strictly on y_tr."""
    y01_tr = (y_tr == 1).astype(np.float64)
    prior = float(np.mean(y01_tr))
    tables = []
    enc_tr_list, enc_names = [], []
    for col_name in target_cols:
        if col_name in names:
            idx = names.index(col_name)
            col_tr = xc_tr[:, idx]
            cats, inv = np.unique(col_tr, return_inverse=True)
            pos = np.bincount(inv, weights=y01_tr)
            cnt = np.bincount(inv)
            smooth = (pos + m_smooth * prior) / (cnt + m_smooth)
            tables.append((col_name, cats, smooth, prior))
            enc_tr_list.append(smooth[inv])
            enc_names.append(f"{col_name}_RISK")
    if not enc_tr_list:
        return xc_tr, names, []
    new_tr = np.hstack([xc_tr[:, :-1], np.column_stack(enc_tr_list), xc_tr[:, -1:]])
    new_names = names[:-1] + enc_names + [names[-1]]
    return new_tr, new_names, tables


def apply_target_risk_encodings(xc_te, names, tables):
    """Apply fitted empirical Bayes risk mappings to test data."""
    enc_te_list, enc_names = [], []
    for col_name, cats, smooth, prior in tables:
        if col_name in names:
            idx = names.index(col_name)
            col_te = xc_te[:, idx]
            pos_idx = np.clip(np.searchsorted(cats, col_te), 0, len(cats) - 1)
            enc_te = np.where(cats[pos_idx] == col_te, smooth[pos_idx], prior)
            enc_te_list.append(enc_te)
            enc_names.append(f"{col_name}_RISK")
    if not enc_te_list:
        return xc_te, names
    new_te = np.hstack([xc_te[:, :-1], np.column_stack(enc_te_list), xc_te[:, -1:]])
    new_names = names[:-1] + enc_names + [names[-1]]
    return new_te, new_names


def _fit_single_gbm(y_tr, tx_tr_gbm, seed):
    """Worker function for single seed GBM training."""
    p = dict(gbm.DEFAULT_PARAMS, seed=seed)
    return gbm.train(y_tr, tx_tr_gbm, p)


def main(argv=None):
    """Run full model training and export checkpoint weights."""
    args = parse_args(argv)
    t0 = time.time()
    print("=" * 70)
    print(f"Starting Training: with_leakage={args.with_leakage}")
    print(f"Output Checkpoint: {args.output}")
    print("=" * 70)

    # 1. Load raw data
    x_train, x_test, y_train, _, test_ids = load_data_cached(args.data_path)
    raw_names = load_feature_names(args.data_path)
    log(t0, f"Loaded raw data: train {x_train.shape}, test {x_test.shape}")

    # 2. Codebook cleaning and preparation
    xc_tr, xc_names, leaky_tr = model_inputs(x_train, raw_names, args.with_leakage)
    del x_train
    log(t0, f"Cleaned features: shape {xc_tr.shape}")

    # 3. Fit target risk encodings
    xc_tr, names, risk_tables = fit_target_risk_encodings(y_train, xc_tr, xc_names)
    log(t0, f"Fitted {len(risk_tables)} target risk encodings")

    # 4. Train 3-seed GBM ensemble
    t_gbm = time.time()
    tree_tr = leaky_tr[0] if leaky_tr is not None else None
    tx_tr_gbm = gbm.tree_features(xc_tr, names, tree_tr)
    gbm_seeds = [42, 43, 44]
    log(
        t0,
        f"Training GBM 3-seed ensemble (seeds {gbm_seeds}) with n_jobs={args.n_jobs}...",
    )

    if args.n_jobs > 1:
        with ProcessPoolExecutor(
            max_workers=min(args.n_jobs, len(gbm_seeds))
        ) as executor:
            futures = [
                executor.submit(_fit_single_gbm, y_train, tx_tr_gbm, s)
                for s in gbm_seeds
            ]
            gbm_models = [f.result() for f in futures]
    else:
        gbm_models = [_fit_single_gbm(y_train, tx_tr_gbm, s) for s in gbm_seeds]

    for s, m in zip(gbm_seeds, gbm_models):
        log(t0, f"  GBM seed {s} fitted: {len(m['trees'])} trees")
    log(t0, f"GBM ensemble training completed in {time.time() - t_gbm:.1f}s")

    # 5. Train 10-seed MLP ensemble
    t_mlp = time.time()
    mlp_state = fit_preprocessor(xc_tr[:, :-1], names[:-1], MLP_CONFIG, y=y_train)
    tx_tr = transform(xc_tr[:, :-1], mlp_state)
    flags_stats = None
    if leaky_tr is not None:
        mean, std = fit_standardizer(leaky_tr[1])
        flags_stats = (mean, std)
        tx_tr = np.hstack([tx_tr, standardize(leaky_tr[1], mean, std)])

    log(t0, "Training 10-seed MLP ensemble (64 ReLU units, Adam, early stopping)...")
    mlp_model = mlp.train(y_train, tx_tr, {"verbose": False})
    log(
        t0,
        f"MLP training completed in {time.time() - t_mlp:.1f}s ({len(mlp_model['nets'])} nets)",
    )

    # 6. Save model weights checkpoint
    checkpoint = {
        "version": "1.0",
        "with_leakage": args.with_leakage,
        "top_rate": args.top_rate,
        "gbm_weight": args.gbm_weight,
        "risk_tables": risk_tables,
        "names": names,
        "gbm_models": gbm_models,
        "mlp_state": mlp_state,
        "flags_stats": flags_stats,
        "mlp_model": mlp_model,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_model(args.output, checkpoint)
    log(t0, f"Pretrained model weights successfully saved to {args.output}")

    # 7. Generate submission file via inference
    if args.submission_output:
        log(t0, "Generating submission via test inference...")
        xc_te, names_te, leaky_te = model_inputs(x_test, raw_names, args.with_leakage)
        del x_test
        xc_te, names_te = apply_target_risk_encodings(xc_te, names_te, risk_tables)
        tree_te = leaky_te[0] if leaky_te is not None else None
        flags_te = leaky_te[1] if leaky_te is not None else None

        tx_te_gbm = gbm.tree_features(xc_te, names_te, tree_te)
        s_gbm = np.mean([gbm.score(m, tx_te_gbm) for m in gbm_models], axis=0)

        tx_te = transform(xc_te[:, :-1], mlp_state)
        if flags_stats is not None:
            mean, std = flags_stats
            tx_te = np.hstack([tx_te, standardize(flags_te, mean, std)])
        s_mlp = mlp.score(mlp_model, tx_te)

        scores = args.gbm_weight * s_gbm + (1.0 - args.gbm_weight) * s_mlp
        if tree_te is not None:
            scores[~np.isnan(tree_te[:, 0])] += 10.0

        y_pred = predict_labels(scores, np.quantile(scores, 1.0 - args.top_rate))
        pos_cnt = int(np.sum(y_pred == 1))
        os.makedirs(
            os.path.dirname(os.path.abspath(args.submission_output)), exist_ok=True
        )
        create_csv_submission(test_ids, y_pred, args.submission_output)
        log(
            t0,
            f"Submission successfully written to {args.submission_output} "
            f"({pos_cnt} positives, {pos_cnt / len(y_pred):.2%})",
        )
    print("=" * 70)
    print(f"Finished in {time.time() - t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
