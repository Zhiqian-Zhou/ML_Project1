"""Regression checks for final training and validation leakage boundaries."""

import numpy as np

import cv_final
import preprocessing as prep
from preprocessing import add_derived, fit_composite_stats
from train import apply_target_risk_encodings, fit_target_risk_encodings


def test_composite_statistics_are_reused_for_held_out_rows(monkeypatch):
    names = ["_AGE80", "_BMI5", "GENHLTH", "BPHIGH4"]
    train = np.array([[30., 2000., 2., 1.], [50., 3000., 3., 2.]])
    stats = fit_composite_stats(train, names, codes={})
    held = np.array([[60., 3500., 4., 1.]])
    def reject_refit(_):
        raise AssertionError("held-out statistics were fitted")

    with monkeypatch.context() as patcher:
        patcher.setattr(prep.np, "nanmean", reject_refit)
        one, out_names = add_derived(held, held, names, composite_stats=stats)
    batched, _ = add_derived(
        np.vstack([held, [[1000., 9000., 1., 2.]]]),
        np.vstack([held, [[1000., 9000., 1., 2.]]]),
        names, composite_stats=stats,
    )
    np.testing.assert_allclose(one[0], batched[0], equal_nan=True)
    assert np.isfinite(one[0, out_names.index("AGE_VASCULAR_SYNERGY")])


def test_training_risk_encoding_excludes_own_label():
    names = ["_STATE", "dummy"]
    x = np.column_stack([np.ones(10), np.zeros(10)])
    y = np.array([1, -1] * 5)
    first, out_names, tables = fit_target_risk_encodings(
        y, x, names, target_cols=("_STATE",)
    )
    flipped = y.copy()
    flipped[0] *= -1
    second, _, _ = fit_target_risk_encodings(
        flipped, x, names, target_cols=("_STATE",)
    )
    risk_col = out_names.index("_STATE_RISK")
    assert first[0, risk_col] == second[0, risk_col]
    assert first[1, risk_col] != second[1, risk_col]
    applied, _ = apply_target_risk_encodings(x, names, tables)
    assert applied[0, risk_col] != first[0, risk_col]


def test_final_cv_never_passes_validation_labels_to_fitting(monkeypatch, tmp_path):
    observed = []

    def fake_fit(x_train, y_train, names, with_leakage, **kwargs):
        observed.append((x_train.copy(), y_train.copy(), with_leakage))
        return {"top_rate": 0.5}

    def fake_infer(checkpoint, x_test, names, ids):
        return np.ones(len(ids)), np.arange(len(ids), dtype=float)

    monkeypatch.setattr(cv_final, "fit_ensemble", fake_fit)
    monkeypatch.setattr(cv_final, "infer", fake_infer)
    x = np.arange(12).reshape(4, 3)
    y = np.array([1, -1, 1, -1])
    result = cv_final.fold_scores(y, x, ["a", "b", "c"], True,
                                  np.array([0, 2]), np.array([1, 3]), 0,
                                  scores_dir=tmp_path)
    np.testing.assert_array_equal(observed[0][0], x[[0, 2]])
    np.testing.assert_array_equal(observed[0][1], y[[0, 2]])
    assert observed[0][2] is True
    np.testing.assert_array_equal(result, [0., 1.])
    with np.load(tmp_path / "fold_0.npz") as saved:
        np.testing.assert_array_equal(saved["row_indices"], [1, 3])
        np.testing.assert_array_equal(saved["y"], [-1, -1])
        np.testing.assert_array_equal(saved["scores"], result)
