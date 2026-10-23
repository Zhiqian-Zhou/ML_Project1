"""Data cleaning and feature engineering for the BRFSS 2015 dataset.

Golden rule: every statistic (medians, means, stds, kept columns, categories)
is computed on the TRAINING rows only and then applied to the other rows
(inside cross-validation: on the training folds only).

Dataset facts (notebooks/01_eda.ipynb):
    * 328,135 train rows, 109,379 test rows, 321 features, labels in {-1, 1}.
    * Only ~8.8% positives (class imbalance ~10:1).
    * 147 columns are more than 50% NaN; missingness often follows skip logic.
    * Codebook special values: 7/9, 77/99, 777/999 = "don't know / refused",
      88 / 888 = "none" (means 0), 555 = "never". They are handled PER COLUMN,
      because in some columns 7 or 99 are real values (e.g. _AGEG5YR).
    * Codebook: https://www.cdc.gov/brfss/annual_data/annual_2015.html

Pipeline (driven by eda/feature_spec.csv and eda/special_codes.json):
    * clean_data (stateless, each row uses only its own values): special
      codes and unit conversions, skip-logic fills, derived features and
      "don't know" flags.
    * fit_preprocessor (training rows) then transform (any rows): column
      selection, binary recode, missing indicators, median/mode imputation,
      one-hot encoding, log1p of counts, optional extra features,
      standardization, bias column.
    * preprocess(x_train, x_test, feature_names, config) chains both.
"""

import csv
import json
import os
import re

import numpy as np

from helpers import load_csv_data

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC_PATH = os.path.join(MODULE_DIR, "eda", "feature_spec.csv")
CODES_PATH = os.path.join(MODULE_DIR, "eda", "special_codes.json")
RAW_KEYS = ("x_train", "x_test", "y_train", "train_ids", "test_ids")
NUMERIC_KINDS = ("ordinal", "continuous", "continuous_log")
MODE_KINDS = ("binary", "onehot")

# (target, [(gate, raw gate values), ...] all required, fill value in RAW coding).
# A NaN of the target is filled when every RAW gate value is in its list: the
# question was skipped by design, so NaN means "no" / 0 (notebook section 3.3).
STRUCTURAL_FILLS = [
    ("AVEDRNK2", [("ALCDAY5", [888])], 0),
    ("DRNK3GE5", [("ALCDAY5", [888])], 0),
    ("MAXDRNKS", [("ALCDAY5", [888])], 0),
    ("POORHLTH", [("PHYSHLTH", [88]), ("MENTHLTH", [88])], 0),
    ("BPMEDS", [("BPHIGH4", [2, 3, 4])], 2),
    ("CHOLCHK", [("BLOODCHO", [2])], 5),
    ("SMOKDAY2", [("SMOKE100", [2])], 3),
    ("PA1MIN_", [("EXERANY2", [2])], 0),
    ("PA1VIGM_", [("EXERANY2", [2])], 0),
    ("JOINPAIN", [("HAVARTH3", [2])], 0),
    ("INSULIN", [("DIABETE3", [2, 3, 4])], 2),
]

# Raw columns read by add_derived.
DERIVED_INPUTS = ["NUMADULT", "HHADULT", "_AGE80", "DIABAGE2", "DIABETE3"]

# Optional products of standardized columns (config "interactions"), found by
# looking at the residuals of the logistic model on the dev rows.
INTERACTIONS = [
    ("_AGE80", "SEX"),
    ("_AGE80", "INCOME2"),
    ("GENHLTH", "DIABETE3=1"),
    ("_AGE80", "BPHIGH4=1"),
    ("_AGE80", "CVDSTRK3"),
    ("_AGE80", "CHCCOPD1"),
    ("SEX", "GENHLTH"),
    ("_AGE80", "HAVARTH3"),
    ("_AGE80", "GENHLTH"),
    ("SEX", "INCOME2"),
]
AGE_KNOTS = (45, 55, 65)  # optional hinges max(_AGE80 - knot, 0)

DEFAULT_CONFIG = {
    # preprocessing
    "max_nan_ratio": 0.95,
    "nan_indicators": "auto",
    "nan_indicator_min_share": 0.005,
    "nan_indicator_min_effect": 0.02,
    "indicator_max_corr": 0.99,
    "rare_level_min_share": 0.005,
    "log_counts": True,
    "poly_degree": 1,
    "dont_know_indicators": False,
    "dont_know_min_share": 0.002,
    "n_dont_know": False,
    "age_hinges": False,
    "interactions": True,
    "dedup_corr": None,
    # model (regularized logistic regression by gradient descent)
    "lambda_": 1e-4,
    "gamma": 0.5,
    "max_iters": 1000,
}


def merge_config(config=None):
    """Return a copy of DEFAULT_CONFIG updated with config."""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config or {})
    return cfg


# ----------------------------------------------------------------------
# Loading


def load_feature_names(data_path):
    """Read the 321 column names of x_train.csv (without the Id column)."""
    with open(os.path.join(data_path, "x_train.csv")) as f:
        return f.readline().strip().split(",")[1:]


def load_data_cached(data_path, sub_sample=False, cache_dir=None):
    """Load the data with helpers.load_csv_data and cache it as a .npz file.

    np.genfromtxt on ~320 MB of csv takes about a minute; the arrays are saved
    the first time and reloaded with np.load afterwards.

    Args:
        data_path: str, folder containing the 3 csv files.
        sub_sample: bool, forwarded to load_csv_data (cache file raw_sub.npz).
        cache_dir: str or None, cache folder. None uses <data_path>/../cache.

    Returns:
        x_train, x_test, y_train, train_ids, test_ids (as in helpers.py).
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(data_path)), "cache")
    cache = os.path.join(cache_dir, "raw_sub.npz" if sub_sample else "raw.npz")
    if not os.path.exists(cache):
        arrays = load_csv_data(data_path, sub_sample)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, **dict(zip(RAW_KEYS, arrays)))
    with np.load(cache) as data:
        return tuple(data[key] for key in RAW_KEYS)


def load_special_codes():
    """Read eda/special_codes.json: name -> {nan_codes, zero_codes, notes, ...}."""
    with open(CODES_PATH) as f:
        return json.load(f)


def load_feature_spec(path=None):
    """Read eda/feature_spec.csv as a list of dicts (one per row, file order).

    positive_values becomes a list of floats (empty unless kind is binary).
    """
    with open(SPEC_PATH if path is None else path, newline="") as f:
        spec = [dict(row) for row in csv.DictReader(f)]
    for row in spec:
        pos = row["positive_values"].strip()
        row["positive_values"] = [float(v) for v in pos.split("|")] if pos else []
    return spec


def spec_by_name(spec):
    """Index the spec rows by feature name."""
    return {row["name"]: row for row in spec}


# ----------------------------------------------------------------------
# Stateless cleaning: codes, units, skip-logic fills, derived features


def convert_units(raw, kind):
    """Convert BRFSS unit-encoded values to one unit (rules in "_transforms").

    kind is one of weight2 (-> kg), height3 (-> cm), perweek_wm (-> per week),
    food (-> per day), peryear_dwmy (-> per year), hhmm (-> minutes). Values
    outside the coded ranges (special codes, NaN) are returned as is.
    """
    out = raw.copy()

    def between(low, high):
        """Return the mask of raw values in [low, high]."""
        return (raw >= low) & (raw <= high)

    if kind == "weight2":  # 50-999 pounds, 9000-9998 kilograms -> kg
        out[between(50, 999)] = raw[between(50, 999)] * 0.45359
        out[between(9000, 9998)] = raw[between(9000, 9998)] - 9000
    elif kind == "height3":  # 200-711 feet+inches (510 = 5 ft 10 in), 9xxx cm
        ft_in = between(200, 711)
        out[ft_in] = (raw[ft_in] // 100 * 12 + raw[ft_in] % 100) * 2.54
        out[between(9000, 9998)] = raw[between(9000, 9998)] - 9000
    elif kind == "perweek_wm":  # 1xx = times per week, 2xx = per month
        out[between(101, 199)] = raw[between(101, 199)] - 100
        out[between(201, 299)] = (raw[between(201, 299)] - 200) * 7 / 30
    elif kind == "food":  # 1xx per day, 2xx per week, 3xx per month -> per day
        out[between(101, 199)] = raw[between(101, 199)] - 100
        out[between(201, 299)] = (raw[between(201, 299)] - 200) / 7
        out[between(301, 399)] = (raw[between(301, 399)] - 300) / 30
    elif kind == "peryear_dwmy":  # 1xx day, 2xx week, 3xx month, 4xx year
        for low, factor in ((100, 365), (200, 52), (300, 12), (400, 1)):
            mask = between(low + 1, low + 99)
            out[mask] = (raw[mask] - low) * factor
    elif kind == "hhmm":  # hours and minutes (130 = 1 h 30) -> minutes
        known = ~np.isnan(raw)
        out[known] = raw[known] // 100 * 60 + raw[known] % 100
    return out


def replace_special_codes(x, feature_names, codes):
    """Map special codes, convert units and apply scale factors, per column.

    On each column listed in codes, in this order: unit conversion of the raw
    values ("transform=<kind>" in notes), nan_codes -> NaN, zero_codes -> 0,
    "recode" {raw: new}, multiplication by "scale". Codes are matched on the
    raw values, so a converted value never collides with a code.

    Args:
        x: numpy array of shape (N, D), raw values.
        feature_names: list of D column names.
        codes: dict from load_special_codes.

    Returns:
        Cleaned copy of x, shape (N, D), float64.
    """
    x = np.array(x, dtype=np.float64)
    for j, name in enumerate(feature_names):
        rule = codes.get(name)
        if not rule:
            continue
        raw = x[:, j].copy()
        found = re.search(r"transform=(\w+)", rule.get("notes", ""))
        if found:
            x[:, j] = convert_units(raw, found.group(1))
        x[np.isin(raw, rule["nan_codes"]), j] = np.nan
        x[np.isin(raw, rule["zero_codes"]), j] = 0
        for code_value, new_value in rule.get("recode", {}).items():
            x[raw == float(code_value), j] = new_value
        x[:, j] *= rule.get("scale", 1)
    return x


def fill_structural(xc, x_raw, feature_names):
    """Fill skip-logic NaN with their logical value (rules in STRUCTURAL_FILLS).

    A NaN of the target is filled only when all the raw gate values match; an
    observed value is never overwritten. Rules with a missing column are skipped.
    """
    xc = xc.copy()
    ix = {name: j for j, name in enumerate(feature_names)}
    for target, gates, value in STRUCTURAL_FILLS:
        if target not in ix or any(gate not in ix for gate, _ in gates):
            continue
        mask = np.isnan(xc[:, ix[target]])
        for gate, raw_values in gates:
            mask &= np.isin(x_raw[:, ix[gate]], raw_values)
        xc[mask, ix[target]] = value
    return xc


def add_derived(xc, x_raw, feature_names):
    """Append the derived features hh_adults and diabetes_years.

    hh_adults: adults in the household, landline NUMADULT else cell HHADULT.
    diabetes_years: max(_AGE80 - DIABAGE2, 0) if the raw DIABETE3 is 1
    (diabetic), 0 if it is 2, 3 or 4 (not diabetic), NaN otherwise.

    Returns:
        (x_out, names): xc with 2 more columns, and the D + 2 names.
    """
    ix = {name: j for j, name in enumerate(feature_names)}
    hh_adults = xc[:, ix["NUMADULT"]].copy()
    missing = np.isnan(hh_adults)
    hh_adults[missing] = xc[missing, ix["HHADULT"]]
    status = x_raw[:, ix["DIABETE3"]]
    diabetic = status == 1
    years = np.full(xc.shape[0], np.nan)
    # NaN - x stays NaN, and np.maximum propagates NaN.
    age_gap = xc[diabetic, ix["_AGE80"]] - xc[diabetic, ix["DIABAGE2"]]
    years[diabetic] = np.maximum(age_gap, 0)
    years[np.isin(status, [2, 3, 4])] = 0
    x_out = np.column_stack([xc, hh_adults, years])
    return x_out, list(feature_names) + ["hh_adults", "diabetes_years"]


def clean_data(x_raw, feature_names, spec=None, codes=None, keep_raw=()):
    """Stateless cleaning of raw data (train and test can be cleaned apart).

    Args:
        x_raw: numpy array of shape (N, 321), raw values.
        feature_names: list of the 321 raw column names.
        spec: list of spec rows (load_feature_spec), or None to read SPEC_PATH.
        codes: dict from load_special_codes, or None to read CODES_PATH.
        keep_raw: names of dropped raw columns to append unchanged at the end
            (e.g. ("_STATE",) for tree models). fit_preprocessor ignores them.

    Returns:
        (xc, names): the K spec rows whose kind is not drop (raw and derived,
        in spec order), then one 0/1 flag "<name>_dk" per raw kept column
        that has "don't know / refused" codes (its nan_codes): 1 if the raw
        value is such a code, then the keep_raw columns.
    """
    spec = load_feature_spec() if spec is None else spec
    codes = load_special_codes() if codes is None else codes
    out_names = [r["name"] for r in spec if r["kind"] != "drop"]
    raw_out = [n for n in out_names if n in feature_names]
    # Raw columns needed by the outputs, the skip-logic fills and add_derived.
    needed = set(raw_out) | set(DERIVED_INPUTS)
    for target, gates, _ in STRUCTURAL_FILLS:
        if target in needed:
            needed |= {gate for gate, _ in gates}
    sub_idx = [j for j, name in enumerate(feature_names) if name in needed]
    sub_names = [feature_names[j] for j in sub_idx]
    x_sub = np.asarray(x_raw, dtype=np.float64)[:, sub_idx]
    xc = replace_special_codes(x_sub, sub_names, codes)
    xc = fill_structural(xc, x_sub, sub_names)
    xc, all_names = add_derived(xc, x_sub, sub_names)
    ix = {name: j for j, name in enumerate(all_names)}
    dk_names = [n for n in raw_out if codes.get(n, {}).get("nan_codes")]
    dk = [np.isin(x_sub[:, ix[n]], codes[n]["nan_codes"]) for n in dk_names]
    extra = [x_raw[:, feature_names.index(n)] for n in keep_raw]
    xc = np.column_stack([xc[:, [ix[n] for n in out_names]]] + dk + extra)
    return xc, out_names + [n + "_dk" for n in dk_names] + list(keep_raw)


# ----------------------------------------------------------------------
# Stateful steps (fitted on the training rows)


def select_columns(xc_train, names, rows, max_nan_ratio):
    """Kept names: spec kind not drop, train NaN share <= max_nan_ratio, not constant.

    Columns that are not in the spec (e.g. the "_dk" flags) are not kept here.
    """
    nan = np.isnan(xc_train)
    nan_share = nan.mean(axis=0)
    col_min = np.where(nan, np.inf, xc_train).min(axis=0)
    col_max = np.where(nan, -np.inf, xc_train).max(axis=0)
    kept = []
    for j, name in enumerate(names):
        if name not in rows or rows[name]["kind"] == "drop":
            continue
        if nan_share[j] <= max_nan_ratio and col_max[j] > col_min[j]:
            kept.append(name)
    return kept


def binary_recode(x, kinds, positive_values):
    """Map binary columns to 1 (one of their positive raw values) or 0, keep NaN.

    The positive values are explicit per column (e.g. _RFDRHV5 is coded
    1 = no, 2 = yes, so its positive value is 2).
    """
    out = np.array(x, dtype=np.float64)
    for j, kind in enumerate(kinds):
        if kind != "binary":
            continue
        if not positive_values[j]:
            raise ValueError("binary column %d has no positive value" % j)
        col = out[:, j]
        hit = np.isin(col, positive_values[j]).astype(np.float64)
        out[:, j] = np.where(np.isnan(col), np.nan, hit)
    return out


def nan_indicator_candidates(x_train, config, y=None):
    """Columns that may get a missing indicator (residual NaN, training rows).

    config["nan_indicators"] is "none" (no indicator) or "auto": every column
    whose NaN share s satisfies min_share <= s <= 1 - min_share and, if y is
    given, |P(y=1 | NaN) - P(y=1 | observed)| >= nan_indicator_min_effect.

    Returns:
        int array of column indices into x_train.
    """
    if config["nan_indicators"] == "none":
        return np.zeros(0, dtype=int)
    min_share = config["nan_indicator_min_share"]
    is_nan = np.isnan(x_train)
    n_nan = is_nan.sum(axis=0)
    share = n_nan / x_train.shape[0]
    chosen = (share > 0) & (share >= min_share) & (share <= 1.0 - min_share)
    if y is not None:
        pos = np.asarray(y).reshape(-1) == 1
        pos_nan = (is_nan & pos[:, None]).sum(axis=0)
        n_obs = x_train.shape[0] - n_nan
        rate_nan = pos_nan / np.maximum(n_nan, 1)
        rate_obs = (pos.sum() - pos_nan) / np.maximum(n_obs, 1)
        defined = (n_nan > 0) & (n_obs > 0)
        effect = np.where(defined, np.abs(rate_nan - rate_obs), 0.0)
        chosen &= effect >= config["nan_indicator_min_effect"]
    return np.flatnonzero(chosen).astype(int)


def dedup_indicators(ind_train, ref_train, max_abs_corr):
    """Greedy removal of redundant missing indicators, on the training rows.

    In candidate order, an indicator is dropped if it is constant, or if its
    |Pearson correlation| exceeds max_abs_corr with an indicator already kept
    or with a reference column (binary columns and one-hot dummies).

    Returns:
        bool array of shape (K,), True for the indicators to keep.
    """
    n_rows, n_ind = ind_train.shape
    centered = ind_train - ind_train.mean(axis=0)
    ind_std = np.sqrt((centered**2).mean(axis=0))
    valid = ind_std > 1e-12
    safe_std = np.where(valid, ind_std, 1.0)
    corr_ind = (centered.T @ centered) / n_rows / np.outer(safe_std, safe_std)
    # The centered indicators sum to zero, so centered.T @ ref is N times the
    # covariance without centering ref.
    ref_std = np.std(ref_train, axis=0)
    ref_ok = ref_std > 1e-12
    cov = (centered.T @ ref_train[:, ref_ok]) / n_rows
    max_ref = np.abs(cov / np.outer(safe_std, ref_std[ref_ok])).max(axis=1)
    keep = np.zeros(n_ind, dtype=bool)
    for k in range(n_ind):
        if not valid[k] or max_ref[k] > max_abs_corr:
            continue
        if np.any(np.abs(corr_ind[k, keep]) > max_abs_corr):
            continue
        keep[k] = True
    return keep


def fit_imputer(x_train, kinds):
    """Fill value of each column: median (numeric kinds) or mode (binary, onehot).

    The mode takes the smallest value on ties.
    """
    unknown = sorted({k for k in kinds if k not in NUMERIC_KINDS + MODE_KINDS})
    if unknown:
        raise ValueError("cannot impute kinds: %s" % unknown)
    fill = np.zeros(x_train.shape[1])
    med_cols = [j for j, k in enumerate(kinds) if k in NUMERIC_KINDS]
    fill[med_cols] = np.nanmedian(x_train[:, med_cols], axis=0)
    for j in [j for j, k in enumerate(kinds) if k in MODE_KINDS]:
        col = x_train[:, j]
        levels, counts = np.unique(col[~np.isnan(col)], return_counts=True)
        fill[j] = levels[np.argmax(counts)]
    return fill


def apply_imputer(x, fill_values):
    """Replace every NaN of x by the fill value of its column."""
    out = np.array(x, dtype=np.float64)
    rows, cols = np.nonzero(np.isnan(out))
    out[rows, cols] = fill_values[cols]
    return out


def fit_one_hot(x_train, onehot_idx, names, min_share):
    """Fit one encoder per one-hot column on the imputed training rows.

    The most frequent level is the reference and gets no dummy (avoids
    collinearity with the bias). Levels with a train share below min_share
    share one "other" dummy if their total share is at least min_share,
    otherwise they join the reference.
    """
    encoders = []
    for j in onehot_idx:
        levels, counts = np.unique(x_train[:, j], return_counts=True)
        ref_pos = int(np.argmax(counts))
        share = counts / x_train.shape[0]
        is_rare = share < min_share
        is_rare[ref_pos] = False
        own = ~is_rare
        own[ref_pos] = False
        encoders.append(
            {
                "name": names[j],
                "levels": levels[own],
                "reference": levels[ref_pos],
                "has_other": bool(share[is_rare].sum() >= min_share),
            }
        )
    return encoders


def one_hot_encode(x, onehot_idx, encoders):
    """Dummies "NAME=level" (and "NAME=other") of the one-hot columns.

    Rare and unseen values go to "other" if it exists, otherwise to all zeros.

    Returns:
        (dummies, dummy_names): array of shape (N, K) and K names.
    """
    blocks, dummy_names = [], []
    for j, enc in zip(onehot_idx, encoders):
        col = x[:, j]
        blocks.append(col[:, None] == enc["levels"][None, :])
        dummy_names.extend("%s=%g" % (enc["name"], v) for v in enc["levels"])
        if enc["has_other"]:
            known = np.isin(col, enc["levels"]) | (col == enc["reference"])
            blocks.append(~known[:, None])
            dummy_names.append("%s=other" % enc["name"])
    return np.hstack(blocks).astype(np.float64), dummy_names


def fit_standardizer(x_train):
    """Per-column mean and std (np.std, as the lab helper); std 1 if constant."""
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    constant = np.max(x_train, axis=0) == np.min(x_train, axis=0)
    return mean, np.where(constant | (std == 0), 1.0, std)


def standardize(x, mean, std):
    """Return (x - mean) / std."""
    return (x - mean) / std


def build_poly(x, degree):
    """Powers x, x^2, ..., x^degree of each column (lab ex03, without the bias)."""
    return np.hstack([x] + [np.power(x, d) for d in range(2, degree + 1)])


def add_bias(x):
    """Prepend a column of ones (the offset w_0)."""
    return np.hstack([np.ones((x.shape[0], 1)), x])


def corr_dedup(z, names, y, max_corr):
    """Drop an indicator or dummy whose |r| > max_corr with another column.

    Pairs are visited by decreasing |r|. Only indicator / dummy columns (names
    with "_nan", "_dk" or "=") are dropped; if both are, the one less
    correlated with the label goes. z is standardized, so z.T @ z / N is the
    correlation matrix.

    Returns:
        bool array of shape (D,), True for the columns to keep.
    """
    r = np.abs(z.T @ z) / z.shape[0]
    np.fill_diagonal(r, 0)
    y01 = np.zeros(z.shape[0]) if y is None else (np.asarray(y) == 1) * 1.0
    label_corr = np.abs(z.T @ (y01 - y01.mean()))
    is_aux = [("_nan" in n) or ("_dk" in n) or ("=" in n) for n in names]
    keep = np.ones(z.shape[1], dtype=bool)
    rows, cols = np.nonzero(np.triu(r) > max_corr)
    for k in np.argsort(-r[rows, cols], kind="stable"):
        pair = [c for c in (rows[k], cols[k]) if is_aux[c]]
        if keep[rows[k]] and keep[cols[k]] and pair:
            keep[min(pair, key=lambda c: label_corr[c])] = False
    return keep


# ----------------------------------------------------------------------
# Fit / transform


def _fit_columns(xc_train, names, rows, cfg, y):
    """Fit column selection, imputation, one-hot and missing indicators."""
    kept = select_columns(xc_train, names, rows, cfg["max_nan_ratio"])
    kinds = [rows[n]["kind"] for n in kept]
    state = {
        "kept_idx": np.array([names.index(n) for n in kept], dtype=int),
        "kept_names": kept,
        "kinds": kinds,
        "positive_values": [rows[n]["positive_values"] for n in kept],
        "onehot_idx": [i for i, k in enumerate(kinds) if k == "onehot"],
        "numeric_idx": [i for i, k in enumerate(kinds) if k != "onehot"],
    }
    x_kept = xc_train[:, state["kept_idx"]]
    x = binary_recode(x_kept, kinds, state["positive_values"])
    cand = nan_indicator_candidates(x, cfg, y)
    state["fill_values"] = fit_imputer(x, kinds)
    x = apply_imputer(x, state["fill_values"])
    state["encoders"] = fit_one_hot(
        x, state["onehot_idx"], kept, cfg["rare_level_min_share"]
    )
    dummies, _ = one_hot_encode(x, state["onehot_idx"], state["encoders"])
    binary_idx = [i for i, k in enumerate(kinds) if k == "binary"]
    ref = np.hstack([x[:, binary_idx], dummies])
    ind = np.isnan(x_kept[:, cand]).astype(np.float64)
    state["ind_cols"] = cand[dedup_indicators(ind, ref, cfg["indicator_max_corr"])]
    return state


def _fit_dont_know(xc_train, names, state, cfg):
    """Pick the "<col>_dk" flags of the kept columns (and the n_dk count).

    A flag is kept if its train share is at least dont_know_min_share and it
    is not equal to the column's kept missing indicator. The leaky questions
    are dropped in the spec, so they never enter the flags or the count.
    """
    dk_all, dk_idx = [], []
    ind_names = {state["kept_names"][c] for c in state["ind_cols"]}
    for i, name in enumerate(state["kept_names"]):
        if name + "_dk" not in names:
            continue
        j = names.index(name + "_dk")
        dk_all.append(j)
        n_dk = xc_train[:, j].sum()
        n_nan = np.isnan(xc_train[:, state["kept_idx"][i]]).sum()
        same_as_ind = name in ind_names and n_dk == n_nan
        if n_dk >= cfg["dont_know_min_share"] * len(xc_train) and not same_as_ind:
            dk_idx.append(j)
    state["dk_idx"] = dk_idx if cfg["dont_know_indicators"] else []
    state["n_dk_idx"] = dk_all if cfg["n_dont_know"] else []


def _base_block(xc, state):
    """Recode, impute, encode and stack [numeric | dummies | indicators | extras].

    Returns:
        (base, base_names): array of shape (N, B) and its B column names.
    """
    cfg, kept = state["config"], state["kept_names"]
    x_kept = xc[:, state["kept_idx"]]
    x = binary_recode(x_kept, state["kinds"], state["positive_values"])
    x = apply_imputer(x, state["fill_values"])
    num_idx = state["numeric_idx"]
    x_num, num_names = x[:, num_idx], [kept[i] for i in num_idx]
    if cfg["log_counts"]:
        log = [
            j for j, i in enumerate(num_idx) if state["kinds"][i] == "continuous_log"
        ]
        x_num[:, log] = np.log1p(np.maximum(x_num[:, log], 0.0))
    poly = [j for j, i in enumerate(num_idx) if state["kinds"][i] in NUMERIC_KINDS]
    if cfg["poly_degree"] >= 2:
        powers = build_poly(x_num[:, poly], cfg["poly_degree"])[:, len(poly) :]
        x_num = np.hstack([x_num, powers])
        for d in range(2, cfg["poly_degree"] + 1):
            num_names += ["%s^%d" % (num_names[j], d) for j in poly]
    dummies, dummy_names = one_hot_encode(x, state["onehot_idx"], state["encoders"])
    blocks = [x_num, dummies, np.isnan(x_kept[:, state["ind_cols"]]) * 1.0]
    names = num_names + dummy_names + [kept[c] + "_nan" for c in state["ind_cols"]]
    if state["dk_idx"]:
        blocks.append(xc[:, state["dk_idx"]])
        names += [state["input_names"][j] for j in state["dk_idx"]]
    if state["n_dk_idx"]:
        blocks.append(xc[:, state["n_dk_idx"]].sum(axis=1, keepdims=True))
        names.append("n_dk")
    if cfg["age_hinges"] and "_AGE80" in kept:
        age = x[:, kept.index("_AGE80")]
        blocks.append(np.column_stack([np.maximum(age - k, 0) for k in AGE_KNOTS]))
        names += ["_AGE80>%d" % k for k in AGE_KNOTS]
    return np.hstack(blocks), names


def fit_preprocessor(
    xc_train, names, config=None, spec=None, y=None, return_train=False
):
    """Fit every stateful step on the cleaned training rows.

    Args:
        xc_train: numpy array of shape (N, D), clean_data output (training rows).
        names: list of D column names of xc_train.
        config: dict, overrides of DEFAULT_CONFIG.
        spec: list of spec rows (load_feature_spec); None reads SPEC_PATH.
        y: numpy array of shape (N,), training labels in {-1, 1}; used by the
            missing-indicator effect rule and by dedup_corr (None: no effect
            rule, and dedup_corr drops the first column of a pair).
        return_train: bool, also return transform(xc_train, state).

    Returns:
        state: dict read by transform; state["out_names"] names the columns.
        tx_train: only if return_train.
    """
    cfg = merge_config(config)
    rows = spec_by_name(load_feature_spec() if spec is None else spec)
    names = list(names)
    # Column selection, binary recode, imputation, one-hot, missing indicators.
    state = _fit_columns(xc_train, names, rows, cfg, y)
    state["config"] = cfg
    state["input_names"] = names
    # Optional "don't know" flags and count.
    _fit_dont_know(xc_train, names, state, cfg)
    # Standardization of the base block (log1p and powers are inside it).
    base, base_names = _base_block(xc_train, state)
    state["mean"], state["std"] = fit_standardizer(base)
    z = standardize(base, state["mean"], state["std"])
    del base
    # Optional interactions (standardized again) and correlation pruning.
    state["pairs"] = []
    if cfg["interactions"]:
        state["pairs"] = [
            (base_names.index(a), base_names.index(b))
            for a, b in INTERACTIONS
            if a in base_names and b in base_names
        ]
        products = np.column_stack([z[:, a] * z[:, b] for a, b in state["pairs"]])
        state["mean2"], state["std2"] = fit_standardizer(products)
    keep = np.ones(z.shape[1], dtype=bool)
    if cfg["dedup_corr"] is not None:
        keep = corr_dedup(z, base_names, y, cfg["dedup_corr"])
    state["keep"] = keep
    pair_names = ["%s*%s" % (base_names[a], base_names[b]) for a, b in state["pairs"]]
    kept_base = [n for n, k in zip(base_names, keep) if k]
    state["out_names"] = ["bias"] + kept_base + pair_names
    if return_train:
        return state, transform(xc_train, state)
    return state


def transform(xc, state):
    """Apply a fitted preprocessor to cleaned rows (train, held-out or test).

    Args:
        xc: numpy array of shape (N, D), clean_data output with the same
            columns as the rows given to fit_preprocessor.
        state: dict returned by fit_preprocessor.

    Returns:
        tx: numpy array of shape (N, len(state["out_names"])), first column
            all ones, no NaN.
    """
    base, _ = _base_block(xc, state)
    z = standardize(base, state["mean"], state["std"])
    del base
    blocks = [z if state["keep"].all() else z[:, state["keep"]]]
    if state["pairs"]:
        products = np.column_stack([z[:, a] * z[:, b] for a, b in state["pairs"]])
        blocks.append(standardize(products, state["mean2"], state["std2"]))
    return add_bias(np.hstack(blocks))


def preprocess(x_train, x_test, feature_names, config=None, y_train=None):
    """Full pipeline: clean both sets, fit on x_train, transform both.

    Args:
        x_train: numpy array of shape (N, 321), raw training features.
        x_test: numpy array of shape (M, 321), raw test features.
        feature_names: list of 321 column names.
        config: dict with the options (see DEFAULT_CONFIG).
        y_train: numpy array of shape (N,), training labels in {-1, 1}.

    Returns:
        tx_train: numpy array of shape (N, D').
        tx_test: numpy array of shape (M, D').
    """
    spec = load_feature_spec()
    xc_train, names = clean_data(x_train, feature_names, spec=spec)
    state, tx_train = fit_preprocessor(
        xc_train, names, config, spec=spec, y=y_train, return_train=True
    )
    del xc_train
    xc_test, _ = clean_data(x_test, feature_names, spec=spec)
    return tx_train, transform(xc_test, state)
