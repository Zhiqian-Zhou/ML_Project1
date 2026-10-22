"""Data cleaning and feature engineering for the BRFSS 2015 dataset.

STATUS: template. Fill in each function; keep everything numpy-only.

Golden rule: every statistic (medians, means, stds, kept columns, categories)
is computed on the TRAINING data only and then applied to the test data
(and, inside cross-validation, on the training folds only).

Dataset facts (from our exploration, see notebooks/01_eda.ipynb):
    * 328,135 train rows, 109,379 test rows, 321 features, labels in {-1, 1}.
    * Only ~8.8% positives (class imbalance ~10:1).
    * 147 columns are more than 50% NaN; missingness often follows skip logic.
    * Codebook special values: 7/9, 77/99, 777/999 = "don't know / refused",
      88 / 888 = "none" (means 0), 555 = "never". Handle them PER COLUMN,
      because in some columns 7 or 99 are real values (e.g. _AGEG5YR, _STATE).
    * Codebook: https://www.cdc.gov/brfss/annual_data/annual_2015.html
"""

import numpy as np


def load_feature_names(data_path):
    """Read the column names of x_train.csv (helpers.load_csv_data drops them).

    Args:
        data_path: str, folder containing x_train.csv.

    Returns:
        List of 321 feature names (without the Id column).
    """
    raise NotImplementedError("Pending: load_feature_names")


def load_data_cached(data_path, sub_sample=False):
    """Load the data with helpers.load_csv_data and cache it as .npy files.

    np.genfromtxt on ~320 MB of CSV is slow; save the arrays the first time
    and reload them with np.load afterwards.

    Args:
        data_path: str, folder containing the 3 csv files.
        sub_sample: bool, forwarded to load_csv_data.

    Returns:
        x_train, x_test, y_train, train_ids, test_ids (as in helpers.py).
    """
    raise NotImplementedError("Pending: load_data_cached")


def replace_special_codes(x, feature_names):
    """Replace codebook "don't know / refused" codes by NaN, column by column.

    Also map "none" codes (88, 888) and "never" (555) to 0 where it applies.

    Args:
        x: numpy array of shape (N, D).
        feature_names: list of D column names.

    Returns:
        Cleaned copy of x, shape (N, D).
    """
    raise NotImplementedError("Pending: replace_special_codes")


def select_columns(x_train, feature_names, max_nan_ratio=0.8):
    """Decide which columns to keep, using the training data only.

    Drop constant columns, ids/dates/survey weights, and columns with more
    than max_nan_ratio missing values.

    Args:
        x_train: numpy array of shape (N, D).
        feature_names: list of D column names.
        max_nan_ratio: float in [0, 1].

    Returns:
        Boolean mask of shape (D,), True for the columns to keep.
    """
    raise NotImplementedError("Pending: select_columns")


def fit_imputer(x_train):
    """Compute per-column fill values on the training data (e.g. the median).

    Args:
        x_train: numpy array of shape (N, D) with NaNs.

    Returns:
        numpy array of shape (D,), the fill value of each column.
    """
    raise NotImplementedError("Pending: fit_imputer")


def apply_imputer(x, fill_values, add_indicators=True):
    """Fill NaNs with fill_values, optionally appending is-missing indicators.

    Args:
        x: numpy array of shape (N, D) with NaNs.
        fill_values: numpy array of shape (D,).
        add_indicators: bool, append one 0/1 column per column that had NaNs.

    Returns:
        numpy array of shape (N, D') without NaNs.
    """
    raise NotImplementedError("Pending: apply_imputer")


def one_hot_encode(x, categorical_mask, categories=None):
    """One-hot encode the categorical columns.

    Args:
        x: numpy array of shape (N, D).
        categorical_mask: boolean array of shape (D,).
        categories: list of arrays with the levels of each categorical column,
            computed on the training data; None to compute them from x.

    Returns:
        encoded: numpy array of shape (N, D').
        categories: the levels used, to reuse on the test data.
    """
    raise NotImplementedError("Pending: one_hot_encode")


def fit_standardizer(x_train):
    """Compute per-column mean and std on the training data.

    Args:
        x_train: numpy array of shape (N, D).

    Returns:
        mean: numpy array of shape (D,).
        std: numpy array of shape (D,), with zeros replaced by 1.
    """
    raise NotImplementedError("Pending: fit_standardizer")


def standardize(x, mean, std):
    """Standardize x with precomputed mean and std.

    Args:
        x: numpy array of shape (N, D).
        mean: numpy array of shape (D,).
        std: numpy array of shape (D,).

    Returns:
        numpy array of shape (N, D).
    """
    raise NotImplementedError("Pending: standardize")


def build_poly(x, degree):
    """Polynomial expansion of each feature (no cross terms): x, x^2, ..., x^d.

    See labs/ex03/solution/build_polynomial.py for the 1-D version.

    Args:
        x: numpy array of shape (N, D).
        degree: int >= 1.

    Returns:
        numpy array of shape (N, D * degree).
    """
    raise NotImplementedError("Pending: build_poly")


def add_bias(x):
    """Prepend a column of ones (the offset w_0).

    Args:
        x: numpy array of shape (N, D).

    Returns:
        numpy array of shape (N, D + 1).
    """
    raise NotImplementedError("Pending: add_bias")


def preprocess(x_train, x_test, feature_names, config):
    """Full pipeline: fit every step on x_train, apply it to both sets.

    Args:
        x_train: numpy array of shape (N, D), raw training features.
        x_test: numpy array of shape (M, D), raw test features.
        feature_names: list of D column names.
        config: dict with the preprocessing options (see run.py CONFIG).

    Returns:
        tx_train: numpy array of shape (N, D').
        tx_test: numpy array of shape (M, D').
    """
    raise NotImplementedError("Pending: preprocess")
