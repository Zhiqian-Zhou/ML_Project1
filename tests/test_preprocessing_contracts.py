"""Contracts that matter when simplifying preprocessing lookups and transforms."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preprocessing as prep  # noqa: E402


def test_duplicate_names_keep_first_column_during_fitting():
    names = ["_AGE80", "_AGE80", "SEX"]
    x = np.array([[20.0, 80.0, 1.0], [40.0, 90.0, 2.0], [60.0, 100.0, 1.0]])
    state = prep.fit_preprocessor(x, names, {"age_hinges": True})
    np.testing.assert_array_equal(state["kept_idx"], [0, 0, 2])
    assert state["pairs"] == [(0, 2)]


def test_structured_skips_override_observations_before_sentinel_cleaning():
    names = ["DIABETE3", "DIABEYE", "DOCTDIAB", "CIMEMLOS", "CDHELP"]
    raw = np.array([[3, 1, 99, 2, 7], [1, 7, 88, 1, 9], [1, 2, 77, 1, 3.0]])
    before = raw.copy()
    clean, names = prep.extract_structured_skip_features(raw, names)
    assert names == ["DIABEYE_CLEAN", "DOCTDIAB_CLEAN", "CDHELP_CLEAN"]
    np.testing.assert_array_equal(
        clean, [[2, 0, 5], [np.nan, 0, np.nan], [2, np.nan, 3]]
    )
    np.testing.assert_array_equal(raw, before)
    assert clean.dtype == np.float64


def test_missing_optional_and_required_columns():
    raw = np.empty((3, 0), dtype=np.float32)
    high, names = prep.extract_high_auc_features(raw, [])
    assert names == prep.HIGH_AUC_CDC_COLUMNS
    assert high.shape == (3, 9) and high.dtype == np.float64
    assert np.isnan(high).all()
    structured, names = prep.extract_structured_skip_features(raw, [])
    assert structured.shape == (3, 0) and structured.dtype == np.float64
    assert names == []
    with pytest.raises(ValueError):
        prep.leaky_features(raw, [])


def test_transform_uses_training_statistics_without_mutating_state():
    names = ["_AGE80", "SEX", "CHECKUP1"]
    train = np.array([[20.0, 1, 1], [40.0, 2, 2], [60.0, 1, 1], [np.nan, 2, 2]])
    state, fitted = prep.fit_preprocessor(train, names, return_train=True)
    before = copy.deepcopy(state)
    held_out = np.array([[1000.0, 2, 99], [np.nan, 1, 99]])
    transformed = prep.transform(held_out, state)
    np.testing.assert_array_equal(state["fill_values"], [40, 0.5, 1])
    np.testing.assert_array_equal(state["encoders"][2], [1])
    np.testing.assert_array_equal(transformed[:, 0], 1)
    for key in ("fill_values", "mean", "std", "mean2", "std2", "kept_idx", "keep"):
        np.testing.assert_array_equal(state[key], before[key])
    np.testing.assert_array_equal(prep.transform(train, state), fitted)
    # A held-out row is independent of the other rows transformed with it.
    np.testing.assert_array_equal(prep.transform(held_out[:1], state), transformed[:1])
