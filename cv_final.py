"""5-fold cross-validation of the final model (GBM + MLP, top 14% positive).

Same steps as run.py inside every fold: the preprocessor, the target
encoding and both models are fitted on the training folds only, then the
held-out fold is scored and its top POSITIVE_RATE scores are predicted +1.

Usage:
    python cv_final.py [--data_path data/dataset] [--split split.npz]
                       [--n_rows N] [--k_fold 5]
--split: npz file with the dev_idx rows to use (default: all training rows);
--n_rows: only the first N of these rows (quick check).
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
    parser.add_argument("--split", default=None)
    parser.add_argument("--n_rows", type=int, default=None)
    parser.add_argument("--k_fold", type=int, default=5)
    return parser.parse_args()


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
    rows = np.arange(len(y))
    if args.split is not None:
        rows = np.load(args.split)["dev_idx"]
    rows = rows[: args.n_rows]
    xc, names = clean_data(
        x[rows], load_feature_names(args.data_path), keep_raw=("_STATE",)
    )
    y = y[rows]
    del x
    k_indices = build_k_indices(y, args.k_fold, 42)
    f1s = []
    for k in range(args.k_fold):
        tr, te = fold_split(k_indices, k)
        scores = final_scores(y[tr], xc[tr], xc[te], names, k)
        y_pred = predict_labels(scores, np.quantile(scores, 1 - POSITIVE_RATE))
        p, r, f1 = precision_recall_f1(y[te], y_pred)
        f1s.append(f1)
        print(f"fold {k}: F1 {f1:.4f} (P {p:.3f}, R {r:.3f}) [{time.time() - t0:.0f}s]")
    print(f"CV F1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} on {len(y)} rows")


if __name__ == "__main__":
    main()
