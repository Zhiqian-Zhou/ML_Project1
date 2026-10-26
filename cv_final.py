"""5-fold cross-validation of the final model (GBM + MLP, top 14% positive).

Same steps as run.py inside every fold: the preprocessor, the target
encoding and both models are fitted on the training folds only, then the
held-out fold is scored and its top POSITIVE_RATE scores are predicted +1.

Usage:
    python cv_final.py [--data_path data/dataset] [--holdout] [--with_leakage]
                       [--n_rows N] [--k_fold 5]
--holdout: cross-validate on the 80% dev rows of holdout_split, then train on
    them and report the F1 on the 20% held-out rows (model selection used
    the CV only; the held-out rows were scored once per final candidate);
--with_leakage: the variant with the leaky questions (see run.py);
--n_rows: only the first N rows (quick check).
"""

import argparse
import os
import time

import numpy as np

from cross_validation import build_k_indices, fold_split
from metrics import precision_recall_f1, predict_labels
from preprocessing import load_data_cached, load_feature_names
from run import HERE, LEAKY_POSITIVE_RATE, POSITIVE_RATE, ensemble_scores, model_inputs


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=os.path.join(HERE, "data", "dataset"))
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--with_leakage", action="store_true")
    parser.add_argument("--n_rows", type=int, default=None)
    parser.add_argument("--k_fold", type=int, default=5)
    parser.add_argument("--top_rate", type=float, default=None)
    args = parser.parse_args()
    if args.top_rate is None:
        args.top_rate = LEAKY_POSITIVE_RATE if args.with_leakage else POSITIVE_RATE
    return args


def holdout_split(y, frac=0.2, seed=2026):
    """Stratified split of the rows: (dev_idx, holdout_idx), frac of each class held out."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == -1)[0]
    hold = np.concatenate(
        [
            rng.choice(pos, int(frac * len(pos)), replace=False),
            rng.choice(neg, int(frac * len(neg)), replace=False),
        ]
    )
    hold = np.sort(hold)
    return np.setdiff1d(np.arange(len(y)), hold), hold


def f1_top_rate(y, scores, top_rate=POSITIVE_RATE):
    """F1 when the top top_rate scores are predicted +1."""
    y_pred = predict_labels(scores, np.quantile(scores, 1.0 - top_rate))
    return precision_recall_f1(y, y_pred)


def fold_scores(y, xc, names, leaky, tr, te, k):
    """Train on the rows tr (run.ensemble_scores), return the scores of the rows te."""
    leaky_tr = leaky_te = None
    if leaky is not None:
        leaky_tr = tuple(block[tr] for block in leaky)
        leaky_te = tuple(block[te] for block in leaky)
    return ensemble_scores(y[tr], xc[tr], xc[te], names, leaky_tr, leaky_te, seed=k)


def main():
    """Run the k-fold cross-validation and print the F1 of every fold."""
    args = parse_args()
    t0 = time.time()
    x, _, y, _, _ = load_data_cached(args.data_path)
    names = load_feature_names(args.data_path)
    xc, xc_names, leaky = model_inputs(x[: args.n_rows], names, args.with_leakage)
    y = y[: args.n_rows]
    del x
    rows = np.arange(len(y))
    if args.holdout:
        rows, hold = holdout_split(y)
    if args.k_fold > 0:
        k_indices = build_k_indices(y[rows], args.k_fold, 42)
        f1s = []
        for k in range(args.k_fold):
            tr, te = (rows[i] for i in fold_split(k_indices, k))
            scores = fold_scores(y, xc, xc_names, leaky, tr, te, k)
            p, r, f1 = f1_top_rate(y[te], scores, args.top_rate)
            f1s.append(f1)
            print(
                f"fold {k}: F1 {f1:.4f} (P {p:.3f}, R {r:.3f}) [{time.time() - t0:.0f}s]",
                flush=True,
            )
        print(
            f"CV F1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} on {len(rows)} rows (top_rate={args.top_rate:.1%})",
            flush=True,
        )
    if args.holdout:
        scores = fold_scores(y, xc, xc_names, leaky, rows, hold, args.k_fold)
        p, r, f1 = f1_top_rate(y[hold], scores, args.top_rate)
        print(
            f"held-out F1 {f1:.4f} (P {p:.3f}, R {r:.3f}) on {len(hold)} rows",
            flush=True,
        )


if __name__ == "__main__":
    main()
