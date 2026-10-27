"""Paired screening of training-row coverage; no production defaults are changed.

Use identical outer rows for the baseline and candidates. Select boosting rounds
inside training only. Record fixed-rate F1 and paired bootstrap intervals. This
is a reduced-budget screening experiment, not full five-fold model selection.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gbm
import mlp
from metrics import precision_recall_f1, predict_labels
from preprocessing import (
    load_data_cached,
    load_feature_names,
    fit_preprocessor,
    transform,
    fit_standardizer,
    standardize,
)
from run import model_inputs, add_target_risk_encodings, MLP_CONFIG


def evaluate(y, scores, rate):
    """Evaluate a fixed top-rate policy; never optimize on these labels."""
    prediction = predict_labels(scores, np.quantile(scores, 1 - rate))
    precision, recall, f1 = precision_recall_f1(y, prediction)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(np.mean(prediction == y)),
        "positives": int(np.sum(prediction == 1)),
    }, prediction


def paired_interval(y, baseline, candidate, seed, repetitions=500):
    """Percentile interval for fixed-model F1 difference on paired rows."""
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        rows = rng.integers(len(y), size=len(y))
        a = precision_recall_f1(y[rows], baseline[rows])[2]
        b = precision_recall_f1(y[rows], candidate[rows])[2]
        differences.append(b - a)
    return np.quantile(differences, [0.025, 0.975]).tolist()


def main():
    """Run a reproducible pair of outer holdout experiments and persist results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="data/dataset")
    parser.add_argument(
        "--output", default="experiments/results/training_coverage.json"
    )
    parser.add_argument("--n_rows", type=int, default=60000)
    parser.add_argument("--n_trees", type=int, default=300)
    parser.add_argument("--mlp_seeds", type=int, default=3)
    parser.add_argument(
        "--split_seeds", type=int, nargs="+", default=[20261027, 20261028]
    )
    parser.add_argument("--with_leakage", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    x, _, y, ids, _ = load_data_cached(args.data_path)
    selected = np.sort(
        np.random.default_rng(99).choice(
            len(y), min(args.n_rows, len(y)), replace=False
        )
    )
    x, y, ids = x[selected], y[selected], ids[selected]
    names = load_feature_names(args.data_path)
    xc, names, leakage = model_inputs(x, names, args.with_leakage)
    del x
    rate = 0.128 if args.with_leakage else 0.136
    results = {
        "sample_seed": 99,
        "arguments": vars(args),
        "positive_rate": rate,
        "runs": [],
    }
    for split_seed in args.split_seeds:
        start = time.time()
        tr, va = gbm.stratified_split(y, 0.25, split_seed)
        train, valid, input_names = add_target_risk_encodings(
            y[tr], xc[tr], xc[va], names
        )
        tree_tr = leakage[0][tr] if leakage is not None else None
        tree_va = leakage[0][va] if leakage is not None else None
        xt, xv = gbm.tree_features(train, input_names, tree_tr), gbm.tree_features(
            valid, input_names, tree_va
        )
        scores, models = {}, {}
        for refit in [False, True]:
            name = "gbm_refit" if refit else "gbm_baseline"
            params = dict(n_trees=args.n_trees, refit_full=refit, seed=42)
            t = time.time()
            model = gbm.train(y[tr], xt, params)
            scores[name] = gbm.score(model, xv)
            models[name] = {
                "seconds": time.time() - t,
                "trees": len(model["trees"]),
                "training_rows": model["training_rows"],
            }
            print(split_seed, name, models[name], flush=True)
        state = fit_preprocessor(train[:, :-1], input_names[:-1], MLP_CONFIG, y=y[tr])
        mt, mv = transform(train[:, :-1], state), transform(valid[:, :-1], state)
        if leakage is not None:
            mean, std = fit_standardizer(leakage[1][tr])
            mt = np.column_stack([mt, standardize(leakage[1][tr], mean, std)])
            mv = np.column_stack([mv, standardize(leakage[1][va], mean, std)])
        for split in [False, True]:
            name = "mlp_varied" if split else "mlp_baseline"
            t = time.time()
            model = mlp.train(
                y[tr], mt, dict(n_seeds=args.mlp_seeds, split_per_seed=split)
            )
            scores[name] = mlp.score(model, mv)
            models[name] = {"seconds": time.time() - t}
            print(split_seed, name, models[name], flush=True)
        combinations = {
            "baseline": ("gbm_baseline", "mlp_baseline"),
            "gbm_refit_only": ("gbm_refit", "mlp_baseline"),
            "mlp_varied_only": ("gbm_baseline", "mlp_varied"),
            "both": ("gbm_refit", "mlp_varied"),
        }
        metrics, predictions = {}, {}
        for name, (a, b) in combinations.items():
            combined = 0.5 * scores[a] + 0.5 * scores[b]
            if tree_va is not None:
                combined = combined + 10.0 * (~np.isnan(tree_va[:, 0]))
            scores[name] = combined
            metrics[name], predictions[name] = evaluate(y[va], combined, rate)
        for name in combinations:
            metrics[name]["delta_vs_baseline"] = (
                metrics[name]["f1"] - metrics["baseline"]["f1"]
            )
            if name != "baseline":
                metrics[name]["paired_bootstrap_95"] = paired_interval(
                    y[va], predictions["baseline"], predictions[name], split_seed
                )
        results["runs"].append(
            {
                "split_seed": split_seed,
                "train_rows": len(tr),
                "valid_rows": len(va),
                "models": models,
                "metrics": metrics,
                "seconds": time.time() - start,
            }
        )
        output.write_text(json.dumps(results, indent=2))
        np.savez_compressed(
            output.with_name(output.stem + f"_{split_seed}_scores.npz"),
            ids=ids[va],
            y=y[va],
            **scores,
        )
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
