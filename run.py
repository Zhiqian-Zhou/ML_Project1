"""End-to-end inference and training runner for EPFL CS-433 Project 1.

By default, executes fast, deterministic inference in ~12 seconds:
    1. Loads the BRFSS test dataset without loading training data.
    2. Loads pure-NumPy weights (CardiacEnsemble) from weights/ (allow_pickle=False).
    3. Runs codebook cleaning, risk encoding, GBDT + MLP inference, and clinical prior adjustment.
    4. Applies calibrated percentile cutoff and outputs the competition submission CSV.

To retrain from scratch (~4-5 mins with multi-core parallelization):
    Run `python run.py --train`.

Usage:
    # 1. Reproduce Track B Leaky Leaderboard (Champion Model, 0.457 F1, ~12s)
    python run.py

    # 2. Reproduce Track A Clean Screening Model (0.447 F1, ~12s)
    python run.py --clean

    # 3. Retrain Champion Model from scratch (~4m)
    python run.py --train

    # 4. Retrain Clean Model from scratch (~4m)
    python run.py --train --clean
"""

import argparse
import os
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
    """Parse clean, intuitive CLI arguments for inference and training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=os.path.join(HERE, "data", "dataset"))
    parser.add_argument(
        "--output", default=None, help="Custom submission CSV output path"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Use Track A clean screening model (default: Track B champion model)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Retrain model from scratch instead of loading pretrained weights",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to model checkpoint (.npz)",
    )
    parser.add_argument(
        "--top_rate",
        type=float,
        default=None,
        help="Custom percentile threshold override",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=3,
        help="CPU worker processes for parallel GBDT training (default: 3)",
    )
    args = parser.parse_args(argv)

    args.with_leakage = not args.clean
    if args.top_rate is None:
        args.top_rate = LEAKY_RATE if args.with_leakage else CLEAN_RATE
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
        args.output = os.path.join(sub_dir, name)

    return args


def run_training(args):
    """Retrain model from scratch and export checkpoint weights."""
    t0 = time.time()
    print("=" * 70)
    print(
        f"Starting Training: with_leakage={args.with_leakage}, top_rate={args.top_rate:.2%}"
    )
    print(f"Output Checkpoint: {args.weights}")
    print("=" * 70)

    y_train, x_train, x_test, _, test_ids = load_data_cached(args.data_path)
    raw_names = load_feature_names(args.data_path)
    print(
        f"[{time.time() - t0:5.1f}s] Loaded training data: {x_train.shape}, {len(raw_names)} features"
    )

    ensemble = CardiacEnsemble(
        with_leakage=args.with_leakage,
        top_rate=args.top_rate,
    )
    ensemble.fit(x_train, y_train, raw_names, n_jobs=args.n_jobs)
    ensemble.save(args.weights)
    print(f"[{time.time() - t0:5.1f}s] Saved model checkpoint to {args.weights}")

    if args.output and x_test is not None:
        y_pred = ensemble.predict(x_test, raw_names)
        pos_cnt = int(np.sum(y_pred == 1))
        create_csv_submission(test_ids, y_pred, args.output)
        print(
            f"[{time.time() - t0:5.1f}s] Saved submission to {args.output} ({pos_cnt} positives, {pos_cnt / len(y_pred):.2%})"
        )

    print("=" * 70)
    print(f"Training completed successfully in {time.time() - t0:.1f}s!")
    print("=" * 70)


def run_inference(args):
    """Run fast inference using pretrained model weights."""
    t0 = time.time()
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


def main(argv=None):
    """Main CLI entrypoint for reproduction and training."""
    args = parse_args(argv)
    if args.train or not os.path.exists(args.weights):
        run_training(args)
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
