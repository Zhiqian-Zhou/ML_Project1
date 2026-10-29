"""Training coverage and backward compatibility without the competition dataset."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gbm
import mlp
import run


def fixture():
    """Small classification problem with a raw state column."""
    rng = np.random.default_rng(16)
    x = np.column_stack([rng.normal(size=(100, 3)), rng.integers(1, 4, 100)])
    y = np.where(x[:, 0] + 0.2 * x[:, 1] > 0.3, 1, -1)
    return x, y


def test_refit_uses_all_rows_and_fixed_selected_rounds():
    """The refit consumes all training rows, with no second early-stop set."""
    x, y = fixture()
    calls = []
    original = gbm.boost

    def observe(xb, y01, xv, yv, p):
        calls.append((len(y01), xv is None, p["n_trees"]))
        return original(xb, y01, xv, yv, p)

    params = dict(n_trees=8, max_depth=2, min_leaf=3, n_bins=8, refit_full=True)
    with patch.object(gbm, "boost", side_effect=observe):
        model = gbm.train(y, x, params)
    assert calls[0][0] < len(y)
    assert calls[1] == (len(y), True, model["selected_rounds"])
    assert len(model["trees"]) == model["selected_rounds"]
    assert model["training_rows"] == len(y)
    repeated = gbm.train(y, x, params)
    np.testing.assert_array_equal(gbm.score(model, x), gbm.score(repeated, x))


def test_refit_pilot_bins_only_see_inner_training():
    """Bin fitting in the pilot excludes the inner early-stop rows."""
    x, y = fixture()
    tr, va = gbm.stratified_split(y, 0.2, 42)
    x[va, 0] = 1e6
    calls = []
    original = gbm.fit_bins

    def observe(data, bins):
        calls.append(data[:, 0].max())
        return original(data, bins)

    with patch.object(gbm, "fit_bins", side_effect=observe):
        gbm.train(y, x, dict(n_trees=3, n_bins=8, min_leaf=3, refit_full=True))
    assert calls[0] < 1e6
    assert calls[1] == 1e6


def test_default_remains_non_refitted():
    """The legacy training-row count and prediction behavior remain the default."""
    x, y = fixture()
    p = dict(n_trees=4, n_bins=8, min_leaf=3)
    a, b = gbm.train(y, x, p), gbm.train(y, x, dict(p, refit_full=False))
    assert a["training_rows"] < len(y)
    np.testing.assert_array_equal(gbm.score(a, x), gbm.score(b, x))


def test_gbm_state_encoding_excludes_each_training_label():
    codes = np.ones(10)
    y = np.array([0, 1] * 5, dtype=float)
    before = gbm.fit_oof_target_encoding(codes, y, 50)
    flipped = y.copy()
    flipped[0] = 1
    after = gbm.fit_oof_target_encoding(codes, flipped, 50)
    assert before[0] == after[0]
    assert np.any(before[1:] != after[1:])


def test_network_split_diversity_and_reproducibility():
    """Every split is disjoint; opt-in seed variation changes excluded rows."""
    x = np.arange(100, dtype=float)[:, None]
    y = np.where(np.arange(100) % 3 == 0, 1, -1)

    def gather(varied):
        calls = []

        def observe(y01, train, valid, yv, p, seed):
            assert not set(train[:, 0]) & set(valid[:, 0])
            calls.append(valid[:, 0].copy())
            return []

        with patch.object(mlp, "train_one", side_effect=observe):
            mlp.train(y, x, dict(n_seeds=3, split_per_seed=varied))
        return calls

    shared, varied = gather(False), gather(True)
    np.testing.assert_array_equal(shared[0], shared[1])
    assert not np.array_equal(varied[0], varied[1])
    for a, b in zip(varied, gather(True)):
        np.testing.assert_array_equal(a, b)


def test_leaky_default_and_experimental_output_paths():
    """The leaky default and experimental outputs avoid archived submissions."""
    with patch.object(sys, "argv", ["run.py"]):
        baseline = run.parse_args()
    with patch.object(sys, "argv", ["run.py", "--gbm_refit_full"]):
        experimental = run.parse_args()
    with patch.object(sys, "argv", ["run.py", "--clean"]):
        clean = run.parse_args()
    with patch.object(sys, "argv", ["run.py", "--train"]):
        retrained = run.parse_args()
    assert baseline.with_leakage and baseline.weights == run.DEFAULT_WEIGHTS_LEAKY
    assert not clean.with_leakage and clean.weights == run.DEFAULT_WEIGHTS_CLEAN
    assert retrained.with_leakage
    assert retrained.weights.endswith("leaky_model_corrected.npz")
    assert "submissions" not in baseline.output
    assert "submissions" not in retrained.output
    assert baseline.output != experimental.output
    assert experimental.variant_tag == "_gbm_refit"
