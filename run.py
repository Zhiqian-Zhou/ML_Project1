"""Generate competition predictions via pretrained model weights (reproducible inference).

Course Rule Compliance:
    "If training takes a long time, this run.py can include only inference
    (loading pretrained weights and generating predictions), but then please
    also explain where to find the code for training these weights."

By default, this script executes deterministic inference in about 20 seconds:
    1. Loads the BRFSS test dataset without loading or processing training data.
    2. Loads pure-NumPy model weights (GBM 3 seeds + MLP 10 seeds + fitted preprocessors)
       from weights/ using model_io (allow_pickle=False).
    3. Runs feature transformations, tree inference, and neural network forward passes.
    4. Blends logits and selects top-percentile candidates (calibrated in cross-validation).
    5. Outputs a competition-format CSV. The default uses the leakage-enabled
       checkpoint, whose current output differs from the archived submission.

To retrain weights from scratch (about 10 minutes for the measured full leaky fit):
    Run `python train.py` (or pass `python run.py --train`).

Usage:
    # 1. Run the leakage-enabled checkpoint (~20s)
    python run.py

    # 2. Reproduce the saved clean screening CSV (~20s)
    python run.py --clean

    # 3. Retrain from scratch
    python run.py --train
"""

import argparse
import os
import sys
import time

import numpy as np

import gbm
from helpers import create_csv_submission
from metrics import predict_labels
from model_io import load_model
import mlp
from preprocessing import (
    clean_data,
    leaky_features,
    load_data_cached,
    load_feature_names,
    standardize,
    transform,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS_CLEAN = os.path.join(HERE, "weights", "clean_model.npz")
DEFAULT_WEIGHTS_LEAKY = os.path.join(HERE, "weights", "leaky_model.npz")


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=os.path.join(HERE, "data", "dataset"))
    parser.add_argument("--output", default=None)
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument(
        "--with_leakage", action="store_true",
        help="Use the leakage-enabled model (the default)",
    )
    variant.add_argument(
        "--clean", action="store_true", help="Use the clean screening model",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to pretrained model checkpoint (.npz). Defaults to the archived checkpoint for the selected variant",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Force retraining models from scratch via train.py before inference",
    )
    parser.add_argument(
        "--top_rate",
        type=float,
        default=None,
        help="Custom percentile threshold override",
    )
    parser.add_argument(
        "--gbm_weight",
        type=float,
        default=0.5,
        help="Weight of GBM in ensemble score blending",
    )
    parser.add_argument(
        "--gbm_refit_full",
        action="store_true",
        help="Experimental: refit GBMs on all rows after selecting rounds",
    )
    parser.add_argument(
        "--mlp_split_per_seed",
        action="store_true",
        help="Experimental: vary the inner validation split across networks",
    )
    args = parser.parse_args(argv)
    args.with_leakage = not args.clean

    args.variant_tag = ""
    if args.gbm_refit_full:
        args.variant_tag += "_gbm_refit"
    if args.mlp_split_per_seed:
        args.variant_tag += "_mlp_splits"
    if args.gbm_weight != 0.5:
        args.variant_tag += f"_gbmw{args.gbm_weight:g}"

    if args.weights is None:
        archived = DEFAULT_WEIGHTS_LEAKY if args.with_leakage else DEFAULT_WEIGHTS_CLEAN
        args.weights = archived.replace(".npz", "_corrected.npz") if args.train else archived

    if args.output is None:
        args.output = os.path.join(HERE, "submission" + args.variant_tag + ".csv")

    return args


def log(t0, message):
    """Print timestamped progress message."""
    print(f"[{time.time() - t0:6.0f}s] {message}", flush=True)


def model_inputs(x_raw, names, with_leakage=False, composite_stats=None):
    """Clean raw test features and extract leaky block if requested."""
    xc, xc_names = clean_data(
        x_raw, names, keep_raw=("_STATE",), with_composite=True,
        composite_stats=composite_stats,
    )
    if not with_leakage:
        return xc, xc_names, None
    x_tree, _, x_flags, _ = leaky_features(x_raw, names)
    return xc, xc_names, (x_tree, x_flags)


def apply_target_risk_encodings(xc_te, names, tables):
    """Apply empirical Bayes risk mappings from the pretrained checkpoint."""
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


def infer(checkpoint, x_test_raw, raw_names, test_ids, top_rate_override=None):
    """Run model inference on raw test inputs using loaded model weights."""
    t0 = time.time()
    with_leakage = checkpoint["with_leakage"]
    top_rate = top_rate_override or checkpoint["top_rate"]
    gbm_weight = checkpoint["gbm_weight"]
    risk_tables = checkpoint["risk_tables"]
    gbm_models = checkpoint["gbm_models"]
    mlp_state = checkpoint["mlp_state"]
    flags_stats = checkpoint["flags_stats"]
    mlp_model = checkpoint["mlp_model"]

    log(t0, f"Running inference: with_leakage={with_leakage}, top_rate={top_rate:.2%}")

    # 1. Clean test features
    xc_te, names_te, leaky_te = model_inputs(
        x_test_raw, raw_names, with_leakage, checkpoint.get("composite_stats")
    )
    xc_te, names_te = apply_target_risk_encodings(xc_te, names_te, risk_tables)
    tree_te = leaky_te[0] if leaky_te is not None else None
    flags_te = leaky_te[1] if leaky_te is not None else None

    # 2. Score with 3-seed GBM ensemble
    t_gbm = time.time()
    tx_te_gbm = gbm.tree_features(xc_te, names_te, tree_te)
    s_gbm = np.mean([gbm.score(m, tx_te_gbm) for m in gbm_models], axis=0)
    log(
        t0,
        f"GBM scored {len(tx_te_gbm)} test rows ({len(gbm_models)} seeds) in {time.time() - t_gbm:.1f}s",
    )

    # 3. Score with 10-seed MLP ensemble
    t_mlp = time.time()
    tx_te = transform(xc_te[:, :-1], mlp_state)
    if flags_stats is not None:
        mean, std = flags_stats
        tx_te = np.hstack([tx_te, standardize(flags_te, mean, std)])
    s_mlp = mlp.score(mlp_model, tx_te)
    log(
        t0,
        f"MLP scored {len(tx_te)} test rows ({len(mlp_model['nets'])} seeds) in {time.time() - t_mlp:.1f}s",
    )

    # 4. Ensemble blending
    scores = gbm_weight * s_gbm + (1.0 - gbm_weight) * s_mlp
    if tree_te is not None:
        # Boost respondents who attended cardiac rehab (informative post-event signal)
        harehab_answered = ~np.isnan(tree_te[:, 0])
        scores[harehab_answered] += 10.0

    # 5. Threshold selection
    threshold = np.quantile(scores, 1.0 - top_rate)
    y_pred = predict_labels(scores, threshold)
    pos_cnt = int(np.sum(y_pred == 1))
    log(
        t0,
        f"Inference complete: {pos_cnt} positive predictions ({pos_cnt / len(y_pred):.2%})",
    )

    return y_pred, scores


def main(argv=None):
    """Main entrypoint for fast inference and submission generation."""
    args = parse_args(argv)
    t0 = time.time()

    # If --train is passed or weights do not exist, delegate to train.py
    if args.train or not os.path.exists(args.weights):
        if not args.train:
            print(
                f"Pretrained weights not found at {args.weights}. Training from scratch..."
            )
        import train

        train_argv = ["--data_path", args.data_path, "--output", args.weights]
        if args.with_leakage:
            train_argv.append("--with_leakage")
        if args.output:
            train_argv.extend(["--submission_output", args.output])
        train.main(train_argv)
        return

    print("=" * 70)
    print(f"Loading Pretrained Weights: {args.weights}")
    print(f"Generating Submission   : {args.output}")
    print("=" * 70)

    # 1. Load weights
    checkpoint = load_model(args.weights)
    log(t0, f"Loaded weights checkpoint (v{checkpoint.get('version', '1.0')})")

    # 2. Load test data
    _, x_test, _, _, test_ids = load_data_cached(args.data_path)
    raw_names = load_feature_names(args.data_path)
    log(t0, f"Loaded test data: {x_test.shape}")

    # 3. Perform inference
    y_pred, _ = infer(
        checkpoint, x_test, raw_names, test_ids, top_rate_override=args.top_rate
    )

    # 4. Save submission CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    create_csv_submission(test_ids, y_pred, args.output)
    pos_cnt = int(np.sum(y_pred == 1))
    log(
        t0,
        f"Saved competition submission to {args.output} "
        f"({pos_cnt} positives, {pos_cnt / len(y_pred):.2%})",
    )
    print("=" * 70)
    print(f"Reproduction completed successfully in {time.time() - t0:.1f}s!")
    print("=" * 70)


if __name__ == "__main__":
    main()
