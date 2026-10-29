"""Checks of the cleaning step that need no data file (run: python -m pytest tests)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import clean_data, load_feature_spec  # noqa: E402

LEAKY = ["HAREHAB1", "STREHAB1", "CVDASPRN", "ASPUNSAF", "RLIVPAIN", "RDUCHART"]
LEAKY += ["RDUCSTRK"]


def test_clean_data_drops_leaky_columns():
    """The leaky questions (asked after a heart attack) never leave clean_data."""
    spec = load_feature_spec()
    raw_names = [row["name"] for row in spec if not row["source"]]
    x_raw = np.ones((4, len(raw_names)))
    xc, names = clean_data(x_raw, raw_names, spec=spec)
    assert xc.shape == (4, len(names))
    for name in LEAKY:
        assert name not in names
        assert name + "_dk" not in names


def test_leaky_features_only_in_their_own_block():
    """leaky_features returns exactly the leaky questions, with their flags."""
    from preprocessing import LEAKY_COLUMNS, leaky_features

    spec = load_feature_spec()
    raw_names = [row["name"] for row in spec if not row["source"]]
    x_raw = np.full((3, len(raw_names)), np.nan)
    x_raw[0, raw_names.index("HAREHAB1")] = 1
    x_tree, tree_names, x_flags, flag_names = leaky_features(x_raw, raw_names)
    assert sorted(LEAKY) == sorted(LEAKY_COLUMNS)
    assert tree_names[: len(LEAKY)] == LEAKY_COLUMNS
    assert x_tree.shape == (3, 2 * len(LEAKY))
    assert x_tree[0, 0] == 1 and np.isnan(x_tree[1, 0])
    assert x_flags.shape == (3, len(flag_names))
    assert x_flags[0, flag_names.index("HAREHAB1=1")] == 1
