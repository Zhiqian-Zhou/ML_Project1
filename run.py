"""Generate competition predictions via pretrained model weights (reproducible inference).

Course Rule Compliance:
    "If training takes a long time, this run.py can include only inference
    (loading pretrained weights and generating predictions), but then please
    also explain where to find the code for training these weights."

By default, executes fast, deterministic inference in ~12 seconds:
    1. Loads the BRFSS test dataset without loading training data.
    2. Loads pure-NumPy weights (CardiacEnsemble) from weights/ (allow_pickle=False).
    3. Runs codebook cleaning, risk encoding, GBDT + MLP inference, and clinical prior adjustment.
    4. Applies calibrated percentile cutoff and outputs the competition submission CSV.

To retrain from scratch (~4-5 mins with multi-core parallelization):
    Run `python train.py` (or pass `python run.py --train`).

Usage:
    # 1. Reproduce Track B Leaky Leaderboard (Champion Model, 0.457 F1, ~12s)
    python run.py

    # 2. Reproduce Track A Clean Screening Model (0.447 F1, ~12s)
    python run.py --clean

    # 3. Retrain from scratch
    python run.py --train
"""

import argparse
import os
import sys
import time

import numpy as np

from helpers import create_csv_submission
from models import CardiacEnsemble
from preprocessing import (
    CLEAN_RATE,
    LEAKY_RATE,
    load_data_cached,
    load_feature_names,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS_CLEAN = os.path.join(HERE, "weights", "clean_model.npz")
DEFAULT_WEIGHTS_LEAKY = os.path.join(HERE, "weights", "leaky_model.npz")


def parse_args(argv=None):
    """Parse CLI arguments for inference and reproducibility."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=os.path.join(HERE, "data", "dataset"))
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Generate Track A clean screening model predictions (0.447 F1)",
    )
    parser.add_argument(
        "--with_leakage",
        action="store_true",
        default=None,
        help="Generate Track B leaky model predictions (0.457 F1, default)",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to pretrained model checkpoint (.npz)",
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

    # By default, infer the champion model (leaky, 0.457 F1)
    if args.clean:
        args.with_leakage = False
    else:
        args.with_leakage = True

    args.variant_tag = ""
    if args.gbm_refit_full:
        args.variant_tag += "_gbm_refit"
    if args.mlp_split_per_seed:
        args.variant_tag += "_mlp_splits"
    if args.gbm_weight != 0.5:
        args.variant_tag += f"_gbmw{args.gbm_weight:g}"

    if args.weights is None:
        args.weights = (
            DEFAULT_WEIGHTS_LEAKY if args.with_leakage else DEFAULT_WEIGHTS_CLEAN
        )

    if args.output is None:
        sub_dir = os.path.join(HERE, "submissions")
        name = (
            "submission_with_leakage.csv"
            if args.with_leakage
            else "submission_clean_sota.csv"
        )
        args.output = os.path.join(
            sub_dir, name.replace(".csv", args.variant_tag + ".csv")
        )

    return args


def infer(model, x_test_raw, raw_names, top_rate_override=None):
    """Run model inference using loaded CardiacEnsemble instance."""
    if isinstance(model, dict):
        # Support legacy dictionary checkpoint by wrapping it in CardiacEnsemble
        inst = CardiacEnsemble(
            with_leakage=model["with_leakage"],
            top_rate=model["top_rate"],
            gbm_weight=model["gbm_weight"],
        )
        from models import GradientBoostedTrees, MLPEnsemble

        inst.risk_tables = model["risk_tables"]
        inst.names = model["names"]
        inst.gbm_models = [
            GradientBoostedTrees.from_dict(m) for m in model["gbm_models"]
        ]
        inst.mlp_state = model["mlp_state"]
        inst.flags_stats = model["flags_stats"]
        inst.mlp_model = MLPEnsemble.from_dict(model["mlp_model"])
        model = inst
    scores = model.predict_scores(x_test_raw, raw_names)
    rate = top_rate_override or model.top_rate
    from metrics import predict_labels

    threshold = np.quantile(scores, 1.0 - rate)
    y_pred = predict_labels(scores, threshold)
    return y_pred, scores


def main(argv=None):
    """Main entrypoint for fast inference and submission generation."""
    args = parse_args(argv)
    t0 = time.time()

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
    print(f"Target Submission File   : {args.output}")
    print("=" * 70)

    # 1. Load OOP model ensemble
    ensemble = CardiacEnsemble.load(args.weights)
    print(
        f"[{time.time() - t0:5.1f}s] Loaded CardiacEnsemble (with_leakage={ensemble.with_leakage}, top_rate={ensemble.top_rate:.2%})"
    )

    # 2. Load test features
    _, x_test, _, _, test_ids = load_data_cached(args.data_path)
    raw_names = load_feature_names(args.data_path)
    print(f"[{time.time() - t0:5.1f}s] Loaded test matrix: {x_test.shape}")

    # 3. Perform inference
    y_pred = ensemble.predict(x_test, raw_names, top_rate_override=args.top_rate)
    pos_cnt = int(np.sum(y_pred == 1))
    print(
        f"[{time.time() - t0:5.1f}s] Inference complete: {pos_cnt} positive predictions ({pos_cnt / len(y_pred):.2%})"
    )

    # 4. Save submission CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    create_csv_submission(test_ids, y_pred, args.output)
    print(f"[{time.time() - t0:5.1f}s] Saved submission to {args.output}")
    print("=" * 70)
    print(f"Reproduction completed successfully in {time.time() - t0:.1f}s!")
    print("=" * 70)


if __name__ == "__main__":
    main()
