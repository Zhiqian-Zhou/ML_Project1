"""5-fold cross-validation of the final model (GBM + MLP, top 14% positive).

Same steps as run.py inside every fold: the preprocessor, the target
encoding and both models are fitted on the training folds only, then the
held-out fold is scored and its top POSITIVE_RATE scores are predicted +1.

Usage:
    python cv_final.py [--data_path data/dataset] [--holdout]
                       [--n_rows N] [--k_fold 5]
--holdout: cross-validate on the 80% dev rows of holdout_split, then train on
    them and report the F1 on the 20% held-out rows (model selection used
    the CV only; the held-out rows were scored once per final candidate);
--n_rows: only the first N rows (quick check).
"""

import argparse
import os
import time

import numpy as np

import gbm
import mlp
from cross_validation import build_k_indices, fold_split
from metrics import precision_recall_f1, predict_labels
from preprocessing import clean_data, fit_preprocessor, load_data_cached
from preprocessing import load_feature_names, transform
from run import HERE, MLP_CONFIG, POSITIVE_RATE


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=os.path.join(HERE, "data", "dataset"))
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--n_rows", type=int, default=None)
    parser.add_argument("--k_fold", type=int, default=5)
    return parser.parse_args()


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


def f1_top_rate(y, scores):
    """F1 when the top POSITIVE_RATE scores are predicted +1."""
    y_pred = predict_labels(scores, np.quantile(scores, 1 - POSITIVE_RATE))
    return precision_recall_f1(y, y_pred)


def final_scores(y_tr, xc_tr, xc_te, names, k):
    """Train both models on the training fold, return the held-out scores."""
    params = {"seed": gbm.DEFAULT_PARAMS["seed"] + k}  # other split per fold
    model = gbm.train(y_tr, gbm.tree_features(xc_tr, names), params)
    s_gbm = gbm.score(model, gbm.tree_features(xc_te, names))
    # the MLP input has no _STATE column (last column of xc)
    state = fit_preprocessor(xc_tr[:, :-1], names[:-1], MLP_CONFIG, y=y_tr)
    model = mlp.train(y_tr, transform(xc_tr[:, :-1], state))
    s_mlp = mlp.score(model, transform(xc_te[:, :-1], state))
    return 0.5 * s_gbm + 0.5 * s_mlp


def main():
    """Run the k-fold cross-validation and print the F1 of every fold."""
    args = parse_args()
    t0 = time.time()
    x, _, y, _, _ = load_data_cached(args.data_path)
    xc, names = clean_data(
        x[: args.n_rows], load_feature_names(args.data_path), keep_raw=("_STATE",)
    )
    y = y[: args.n_rows]
    del x
    if args.holdout:
        dev, hold = holdout_split(y)
        xc_hold, y_hold = xc[hold], y[hold]
        xc, y = xc[dev], y[dev]
    k_indices = build_k_indices(y, args.k_fold, 42)
    f1s = []
    for k in range(args.k_fold):
        tr, te = fold_split(k_indices, k)
        scores = final_scores(y[tr], xc[tr], xc[te], names, k)
        p, r, f1 = f1_top_rate(y[te], scores)
        f1s.append(f1)
        print(f"fold {k}: F1 {f1:.4f} (P {p:.3f}, R {r:.3f}) [{time.time() - t0:.0f}s]")
    print(f"CV F1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} on {len(y)} rows")
    if args.holdout:
        scores = final_scores(y, xc, xc_hold, names, args.k_fold)
        p, r, f1 = f1_top_rate(y_hold, scores)
        print(f"held-out F1 {f1:.4f} (P {p:.3f}, R {r:.3f}) on {len(y_hold)} rows")


if __name__ == "__main__":
    main()
