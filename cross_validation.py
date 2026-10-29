"""Stratified K-fold cross-validation and evaluation harness.

Provides:
    1. StratifiedKFold: Stratified K-fold index generator preserving class ratio.
    2. HoldoutSplitter: Stratified 80/20 train/held-out test split.
    3. LinearBaselineValidator: Cross-validation harness for regularized linear models.
    4. EnsembleCrossValidator: Evaluation harness for GBDT + MLP ensemble models.
"""

import argparse
import os
import time

import numpy as np

from implementations import reg_logistic_regression, ridge_regression, sigmoid
from metrics import (
    accuracy,
    best_threshold,
    precision_recall_f1,
    predict_labels,
)
from models import (
    DEFAULT_GBM_PARAMS,
    DEFAULT_MLP_PARAMS,
    GradientBoostedTrees,
    MLPEnsemble,
    tree_features,
)
from preprocessing import (
    CLEAN_RATE,
    LEAKY_RATE,
    add_target_risk_encodings,
    fit_preprocessor,
    fit_standardizer,
    load_data_cached,
    load_feature_names,
    merge_config,
    model_inputs,
    standardize,
    transform,
)

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ("f1", "accuracy", "precision", "recall", "threshold", "f1_train")


# ==============================================================================
# Partitioning Classes
# ==============================================================================


class StratifiedKFold:
    """Stratified K-fold cross-validation splitter."""

    def __init__(self, n_splits=5, seed=42):
        self.n_splits = n_splits
        self.seed = seed

    def split(self, y):
        """Generate (train_indices, val_indices) preserving class ratio."""
        rng = np.random.default_rng(self.seed)
        y_arr = np.asarray(y)
        pos = rng.permutation(np.flatnonzero(y_arr == 1))
        neg = rng.permutation(np.flatnonzero(y_arr != 1))
        pos_parts = np.array_split(pos, self.n_splits)
        neg_parts = np.array_split(neg, self.n_splits)
        k_indices = [
            np.sort(np.concatenate([pos_parts[k], neg_parts[k]])).astype(np.int64)
            for k in range(self.n_splits)
        ]
        all_rows = np.arange(len(y_arr))
        for k in range(self.n_splits):
            val_idx = k_indices[k]
            train_idx = np.setdiff1d(all_rows, val_idx)
            yield train_idx, val_idx


class HoldoutSplitter:
    """Stratified 80/20 train/held-out test splitter."""

    def __init__(self, frac=0.2, seed=2026):
        self.frac = frac
        self.seed = seed

    def split(self, y):
        """Return (dev_indices, holdout_indices)."""
        rng = np.random.default_rng(self.seed)
        y_arr = np.asarray(y)
        pos, neg = np.flatnonzero(y_arr == 1), np.flatnonzero(y_arr != 1)
        hold = np.concatenate(
            [
                rng.choice(pos, int(self.frac * len(pos)), replace=False),
                rng.choice(neg, int(self.frac * len(neg)), replace=False),
            ]
        )
        hold = np.sort(hold)
        dev = np.setdiff1d(np.arange(len(y_arr)), hold)
        return dev, hold


def build_k_indices(y, k_fold=5, seed=42):
    """Backward-compatible function returning fold indices list."""
    rng = np.random.default_rng(seed)
    pos = rng.permutation(np.flatnonzero(np.asarray(y) == 1))
    neg = rng.permutation(np.flatnonzero(np.asarray(y) != 1))
    pos_parts = np.array_split(pos, k_fold)
    neg_parts = np.array_split(neg, k_fold)
    return [
        np.sort(np.concatenate([pos_parts[k], neg_parts[k]])).astype(np.int64)
        for k in range(k_fold)
    ]


def fold_split(k_indices, k):
    """Backward-compatible function returning (train_idx, val_idx) for fold k."""
    te_idx = np.asarray(k_indices[k])
    tr_idx = np.sort(
        np.concatenate([k_indices[i] for i in range(len(k_indices)) if i != k])
    )
    return tr_idx, te_idx


def holdout_split(y, frac=0.2, seed=2026):
    """Backward-compatible function returning (dev_idx, holdout_idx)."""
    return HoldoutSplitter(frac=frac, seed=seed).split(y)


def f1_top_rate(y, scores, top_rate=CLEAN_RATE):
    """Evaluate positive-class F1 when the top top_rate scores are predicted +1."""
    y_pred = predict_labels(scores, np.quantile(scores, 1.0 - top_rate))
    return precision_recall_f1(y, y_pred)


# ==============================================================================
# Linear Model Baseline Validator
# ==============================================================================


class LinearBaselineValidator:
    """Validation harness for linear models (Logistic & Ridge regression)."""

    @staticmethod
    def train_reg_logistic(y, tx, config):
        """Fit L2-regularized logistic regression via gradient descent."""
        cfg = merge_config(config)
        y01 = (np.asarray(y) == 1).astype(np.float64)
        w, _ = reg_logistic_regression(
            y01,
            tx,
            cfg["lambda_"],
            np.zeros(tx.shape[1]),
            cfg["max_iters"],
            cfg["gamma"],
        )
        return w

    @staticmethod
    def train_ridge(y, tx, config):
        """Fit Ridge regression on {-1, 1} labels."""
        w, _ = ridge_regression(np.asarray(y, dtype=np.float64), tx, config["lambda_"])
        return w

    @staticmethod
    def prepare_folds(y, xc, names, k_indices, config):
        """Fit preprocessor strictly on training folds and transform validation fold."""
        folds = []
        for k in range(len(k_indices)):
            tr_idx, te_idx = fold_split(k_indices, k)
            state, tx_tr = fit_preprocessor(
                xc[tr_idx], names, config, y=y[tr_idx], return_train=True
            )
            tx_te = transform(xc[te_idx], state)
            folds.append((tx_tr, y[tr_idx], tx_te, y[te_idx], state))
        return folds

    @staticmethod
    def cross_validate(
        y,
        xc,
        names=None,
        k_indices=None,
        config=None,
        is_logistic=True,
        verbose=False,
    ):
        """K-fold cross-validation of one linear model configuration."""
        if k_indices is None:
            k_indices = build_k_indices(y, k_fold=5, seed=42)
        cfg = merge_config(config)
        folds = LinearBaselineValidator.prepare_folds(y, xc, names, k_indices, cfg)
        train_fn = (
            LinearBaselineValidator.train_reg_logistic
            if is_logistic
            else LinearBaselineValidator.train_ridge
        )
        score_fn = (
            (lambda w, tx: sigmoid(tx @ w)) if is_logistic else (lambda w, tx: tx @ w)
        )

        res = {m: [] for m in METRICS}
        for k, (tx_tr, y_tr, tx_te, y_te, _) in enumerate(folds):
            w = train_fn(y_tr, tx_tr, cfg)
            thr, f1_tr = best_threshold(y_tr, score_fn(w, tx_tr))
            pred = predict_labels(score_fn(w, tx_te), thr)
            prec, rec, f1 = precision_recall_f1(y_te, pred)
            for m, v in zip(METRICS, (f1, accuracy(y_te, pred), prec, rec, thr, f1_tr)):
                res[m].append(v)
            if verbose:
                print(f"fold {k}: F1 {f1:.4f} (train {f1_tr:.4f}) thr {thr:.4f}")
        out = {}
        for m in METRICS:
            out[m] = np.array(res[m], dtype=np.float64)
            out[m + "_mean"] = float(np.mean(out[m]))
            out[m + "_std"] = float(np.std(out[m]))
        out["n_features"] = int(folds[0][0].shape[1])
        return out


# Backward-compatible function aliases
cross_validate = LinearBaselineValidator.cross_validate


# ==============================================================================
# Ensemble Cross-Validator (GBDT + MLP)
# ==============================================================================


class EnsembleCrossValidator:
    """Cross-validation and holdout evaluation harness for GBDT + MLP ensembles."""

    def __init__(
        self,
        with_leakage=False,
        top_rate=None,
        k_fold=5,
        holdout=False,
        gbm_weight=0.5,
    ):
        self.with_leakage = with_leakage
        self.top_rate = top_rate or (LEAKY_RATE if with_leakage else CLEAN_RATE)
        self.k_fold = k_fold
        self.holdout = holdout
        self.gbm_weight = gbm_weight

    def evaluate_split(
        self, y_tr, y_te, xc_tr, xc_te, names, leaky_tr, leaky_te, seed=0
    ):
        """Train ensemble on training fold and return blended scores for test fold."""
        tree_tr, flags_tr = leaky_tr if leaky_tr is not None else (None, None)
        tree_te, flags_te = leaky_te if leaky_te is not None else (None, None)

        # 1. Target risk encoding strictly fitted on training fold
        xc_tr, xc_te, enc_names = add_target_risk_encodings(y_tr, xc_tr, xc_te, names)

        # 2. 3-Seed Gradient Boosted Trees
        tx_tr_gbm = tree_features(xc_tr, enc_names, tree_tr)
        tx_te_gbm = tree_features(xc_te, enc_names, tree_te)
        gbm_seeds = [seed + 42, seed + 43, seed + 44]
        s_gbm_list = []
        for s in gbm_seeds:
            m = GradientBoostedTrees(seed=s).fit(tx_tr_gbm, y_tr)
            s_gbm_list.append(m.predict_logits(tx_te_gbm))
        s_gbm = np.mean(s_gbm_list, axis=0)

        # 3. 10-Seed Multi-Layer Perceptron
        mlp_cfg = {"interactions": False}
        state = fit_preprocessor(xc_tr[:, :-1], enc_names[:-1], mlp_cfg, y=y_tr)
        tx_tr = transform(xc_tr[:, :-1], state)
        tx_te = transform(xc_te[:, :-1], state)
        if flags_tr is not None:
            mean, std = fit_standardizer(flags_tr)
            tx_tr = np.hstack([tx_tr, standardize(flags_tr, mean, std)])
            tx_te = np.hstack([tx_te, standardize(flags_te, mean, std)])
        mlp_model = MLPEnsemble(seed=seed, n_seeds=10).fit(tx_tr, y_tr)
        s_mlp = mlp_model.predict_logits(tx_te)

        # 4. Ensemble score blending & cardiac rehab boost
        scores = self.gbm_weight * s_gbm + (1.0 - self.gbm_weight) * s_mlp
        if tree_te is not None:
            scores[~np.isnan(tree_te[:, 0])] += 10.0
        return scores

    def run(self, x, y, raw_names, n_rows=None):
        """Run full evaluation protocol (K-fold CV and optional 80/20 holdout)."""
        t0 = time.time()
        if n_rows:
            x, y = x[:n_rows], y[:n_rows]
        xc, xc_names, leaky = model_inputs(x, raw_names, self.with_leakage)
        del x

        rows = np.arange(len(y))
        hold = None
        if self.holdout:
            rows, hold = HoldoutSplitter(frac=0.2, seed=2026).split(y)

        print("=" * 70)
        print(
            f"Running Cross-Validation: with_leakage={self.with_leakage}, top_rate={self.top_rate:.2%}"
        )
        print(
            f"Dataset Size: {len(y)} rows (Development: {len(rows)} rows, Folds: {self.k_fold})"
        )
        print("=" * 70)

        if self.k_fold > 0:
            splitter = StratifiedKFold(n_splits=self.k_fold, seed=42)
            f1s = []
            for k, (tr_rel, te_rel) in enumerate(splitter.split(y[rows])):
                tr, te = rows[tr_rel], rows[te_rel]
                l_tr = (leaky[0][tr], leaky[1][tr]) if leaky else (None, None)
                l_te = (leaky[0][te], leaky[1][te]) if leaky else (None, None)
                scores = self.evaluate_split(
                    y[tr], y[te], xc[tr], xc[te], xc_names, l_tr, l_te
                )
                p, r, f1 = f1_top_rate(y[te], scores, self.top_rate)
                f1s.append(f1)
                print(
                    f"Fold {k + 1}/{self.k_fold}: F1 {f1:.4f} (Precision {p:.3f}, Recall {r:.3f}) [{time.time() - t0:.0f}s]",
                    flush=True,
                )
            print(
                f"--> CV F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} on {len(rows)} rows",
                flush=True,
            )

        if self.holdout and hold is not None:
            l_dev = (leaky[0][rows], leaky[1][rows]) if leaky else (None, None)
            l_hld = (leaky[0][hold], leaky[1][hold]) if leaky else (None, None)
            scores = self.evaluate_split(
                y[rows], y[hold], xc[rows], xc[hold], xc_names, l_dev, l_hld
            )
            p, r, f1 = f1_top_rate(y[hold], scores, self.top_rate)
            print("=" * 70)
            print(
                f"--> Held-out 20% Test F1: {f1:.4f} (Precision {p:.3f}, Recall {r:.3f}) on {len(hold)} rows",
                flush=True,
            )
            print("=" * 70)


def ensemble_scores(
    y_tr,
    xc_tr,
    xc_te,
    names,
    leaky_tr=None,
    leaky_te=None,
    seed=0,
    gbm_params=None,
    mlp_params=None,
    gbm_weight=0.5,
    components=None,
):
    """Backward-compatible function evaluating ensemble scores for arbitrary splits."""
    validator = EnsembleCrossValidator(
        with_leakage=(leaky_tr is not None), gbm_weight=gbm_weight
    )
    return validator.evaluate_split(
        y_tr, np.zeros(len(xc_te)), xc_tr, xc_te, names, leaky_tr, leaky_te, seed=seed
    )


def parse_args(argv=None):
    """Parse clean, concise CLI arguments for cross-validation runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=os.path.join(HERE, "data", "dataset"))
    parser.add_argument(
        "--clean", action="store_true", help="Evaluate Track A clean screening model"
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Evaluate 80/20 dev CV and held-out test",
    )
    parser.add_argument(
        "--k_fold", type=int, default=5, help="Number of folds (default: 5)"
    )
    parser.add_argument(
        "--n_rows", type=int, default=None, help="Subsample rows for quick check"
    )
    parser.add_argument(
        "--top_rate", type=float, default=None, help="Percentile threshold override"
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Main entrypoint for cross-validation runner."""
    args = parse_args(argv)
    x, _, y, _, _ = load_data_cached(args.data_path)
    raw_names = load_feature_names(args.data_path)

    validator = EnsembleCrossValidator(
        with_leakage=not args.clean,
        top_rate=args.top_rate,
        k_fold=args.k_fold,
        holdout=args.holdout,
    )
    validator.run(x, y, raw_names, n_rows=args.n_rows)


if __name__ == "__main__":
    main()
