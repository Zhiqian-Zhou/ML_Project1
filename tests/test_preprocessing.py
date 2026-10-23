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
