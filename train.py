"""Train the final ensemble models from scratch and export portable weights.

Executes the complete end-to-end training pipeline via CardiacEnsemble:
    1. Loads raw training matrix and codebook feature names.
    2. Performs codebook-driven cleaning and empirical Bayes risk table fitting.
    3. Trains a 3-seed Histogram GBDT ensemble in parallel (pure NumPy).
    4. Fits the neural network preprocessor and trains a 10-seed MLP ensemble.
    5. Exports the complete fitted pipeline and weights to weights/ (.npz, allow_pickle=False).
    6. Optionally generates the exact competition submission CSV via inference.

Usage:
    # 1. Train Clean SOTA (Track A, zero leakage)
    python train.py --output weights/clean_model.npz

    # 2. Train Leaky Model (Track B, leaderboard exploration)
    python train.py --with_leakage --output weights/leaky_model.npz
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


def main(argv=None):
    """Run full model training and export checkpoint weights."""
    args = parse_args(argv)
    t0 = time.time()
    print("=" * 70)
    print(
        f"Starting Training: with_leakage={args.with_leakage}, top_rate={args.top_rate:.2%}"
    )
    print(f"Output Checkpoint: {args.output}")
    print("=" * 70)

    # 1. Load raw dataset
    y_train, x_train, x_test, _, test_ids = load_data_cached(args.data_path)
    raw_names = load_feature_names(args.data_path)
    print(
        f"[{time.time() - t0:5.1f}s] Loaded training data: {x_train.shape}, {len(raw_names)} features"
    )

    # 2. Initialize and fit OOP CardiacEnsemble
    ensemble = CardiacEnsemble(
        with_leakage=args.with_leakage,
        top_rate=args.top_rate,
        gbm_weight=args.gbm_weight,
    )
    ensemble.fit(x_train, y_train, raw_names, n_jobs=args.n_jobs)
    print(f"[{time.time() - t0:5.1f}s] Training completed successfully!")

    # 3. Export weights
    ensemble.save(args.output)
    print(f"[{time.time() - t0:5.1f}s] Saved model checkpoint to {args.output}")

    # 4. Optional submission generation
    if args.submission_output and x_test is not None:
        y_pred = ensemble.predict(x_test, raw_names)
        pos_cnt = int(np.sum(y_pred == 1))
        create_csv_submission(test_ids, y_pred, args.submission_output)
        print(
            f"[{time.time() - t0:5.1f}s] Saved submission to {args.submission_output} ({pos_cnt} positives, {pos_cnt / len(y_pred):.2%})"
        )

    print("=" * 70)
    print(f"Pipeline finished in {time.time() - t0:.1f}s!")
    print("=" * 70)


if __name__ == "__main__":
    main()
