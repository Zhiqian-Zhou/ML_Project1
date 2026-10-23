"""Data cleaning and feature engineering for the BRFSS 2015 dataset.

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

Pipeline (driven by eda/feature_spec.csv and eda/special_codes.json):
    * Stage C, stateless (clean_data): special codes and unit conversions,
      structural (skip-logic) fills, derived features. Each row uses only its
      own values, so it runs once on train and test.
    * Stage F, stateful (fit_preprocessor on the training rows, then
      transform on any rows): column selection, binary recode, missing
      indicators, median/mode imputation, one-hot encoding with a dropped
      reference level, log1p of counts, optional clipping, standardization,
      optional polynomial terms and interactions, bias column.
    * preprocess(x_train, x_test, feature_names, config) chains both stages.
"""

import csv
import json
import os
import re

import numpy as np

KINDS = ("drop", "binary", "ordinal", "onehot", "continuous", "continuous_log")
NUMERIC_KINDS = ("ordinal", "continuous", "continuous_log")


# ----------------------------------------------------------------------
# Stage C: loading and stateless cleaning (codes, units, skip-logic fills, derived features)

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC_PATH = os.path.join(MODULE_DIR, "eda", "feature_spec.csv")
CODES_PATH = os.path.join(MODULE_DIR, "eda", "special_codes.json")
RAW_KEYS = ("x_train", "x_test", "y_train", "train_ids", "test_ids")

# (target, [(gate, raw gate values), ...] all required, fill value in RAW coding).
# A NaN of the cleaned target is filled when every RAW gate value is in its list:
# the question was skipped by design, so NaN means "no" / 0 (EDA section 3.3, A3).
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

# Derived features built by add_derived when no spec is given: (name, source).
# The recipe name before "(" is a key of DERIVED, the arguments are column names.
DERIVED_SOURCES = [
    ("hh_adults", "derive:coalesce(NUMADULT;HHADULT)"),
    ("diabetes_years", "derive:diabetes_years(_AGE80;DIABAGE2;DIABETE3)"),
]


def load_feature_names(data_path):
    """Read the column names of x_train.csv (helpers.load_csv_data drops them).

    Args:
        data_path: str, folder containing x_train.csv.

    Returns:
        List of 321 feature names (without the Id column).
    """
    with open(os.path.join(data_path, "x_train.csv")) as f:
        return f.readline().strip().split(",")[1:]


def load_data_cached(data_path, sub_sample=False, cache_dir=None):
    """Load the data with helpers.load_csv_data and cache it as a .npz file.

    np.genfromtxt on ~320 MB of CSV takes about a minute; the arrays are saved
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
        from helpers import load_csv_data

        arrays = load_csv_data(data_path, sub_sample)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, **dict(zip(RAW_KEYS, arrays)))
    with np.load(cache) as data:
        return tuple(data[key] for key in RAW_KEYS)


def load_special_codes(path=None):
    """Read the per-column special codes (eda/special_codes.json).

    Args:
        path: str or None, json path. None uses CODES_PATH.

    Returns:
        dict name -> {"nan_codes", "zero_codes", "notes", optional "recode"
        and "scale"}, plus the "_transforms" description entry.
    """
    with open(CODES_PATH if path is None else path) as f:
        return json.load(f)


def convert_units(raw, kind):
    """Convert BRFSS unit-encoded values to one unit (rules in "_transforms").

    Values outside the coded ranges (special codes, NaN) are returned as is.

    Args:
        raw: numpy array of shape (N,), raw values of one column.
        kind: str, one of weight2 (-> kg), height3 (-> cm), perweek_wm
            (-> per week), food (-> per day), peryear_dwmy (-> per year),
            hhmm (-> minutes). Any other kind returns a copy.

    Returns:
        numpy array of shape (N,), converted copy.
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


def replace_special_codes(x, feature_names, codes=None):
    """Map special codes, convert units and apply scale factors, per column.

    On each column listed in codes, in this order: unit conversion of the RAW
    values ("transform=<kind>" in notes), nan_codes -> NaN, zero_codes -> 0,
    "recode" {raw: new}, multiplication by "scale". Codes are matched on the
    raw values, so a converted value never collides with a code.

    Args:
        x: numpy array of shape (N, D), raw values.
        feature_names: list of D column names.
        codes: dict from load_special_codes, or None to read CODES_PATH.

    Returns:
        Cleaned copy of x, shape (N, D), float64.
    """
    if codes is None:
        codes = load_special_codes()
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

    A NaN of the target is filled only when all the RAW gate values match; an
    observed target value is never overwritten. Rules whose target or gate
    columns are not in feature_names are skipped.

    Args:
        xc: numpy array of shape (N, D), output of replace_special_codes.
        x_raw: numpy array of shape (N, D), raw values of the same columns.
        feature_names: list of D column names.

    Returns:
        Filled copy of xc, shape (N, D).
    """
    xc = xc.copy()
    ix = {name: j for j, name in enumerate(feature_names)}
    for target, gates, value in STRUCTURAL_FILLS:
        if target not in ix or any(gate not in ix for gate, _ in gates):
            continue
        j = ix[target]
        mask = np.isnan(xc[:, j])
        for gate, raw_values in gates:
            mask &= np.isin(x_raw[:, ix[gate]], raw_values)
        xc[mask, j] = value
    return xc


def _derive_coalesce(xc, x_raw, ix, args):
    """Return the first non-NaN cleaned value among the columns args.

    Args:
        xc: numpy array of shape (N, D), cleaned values.
        x_raw: numpy array of shape (N, D), raw values (unused).
        ix: dict column name -> index into xc.
        args: list of column names, by priority.

    Returns:
        numpy array of shape (N,).
    """
    out = xc[:, ix[args[0]]].copy()
    for name in args[1:]:
        missing = np.isnan(out)
        out[missing] = xc[missing, ix[name]]
    return out


def _derive_diabetes_years(xc, x_raw, ix, args):
    """Return the years since the diabetes diagnosis.

    max(age - diagnosis age, 0) if the RAW status is 1 (diabetic), 0 if it is
    2, 3 or 4 (not diabetic), NaN otherwise or when an age is NaN.

    Args:
        xc: numpy array of shape (N, D), cleaned values.
        x_raw: numpy array of shape (N, D), raw values.
        ix: dict column name -> index.
        args: [age column, diagnosis age column, status column], e.g.
            [_AGE80, DIABAGE2, DIABETE3].

    Returns:
        numpy array of shape (N,), values >= 0 or NaN.
    """
    age, diag_age, status = args
    status_raw = x_raw[:, ix[status]]
    out = np.full(xc.shape[0], np.nan)
    diabetic = status_raw == 1
    # NaN - x stays NaN, and np.maximum propagates NaN.
    years = np.maximum(xc[diabetic, ix[age]] - xc[diabetic, ix[diag_age]], 0)
    out[diabetic] = years
    out[np.isin(status_raw, [2, 3, 4])] = 0
    return out


DERIVED = {"coalesce": _derive_coalesce, "diabetes_years": _derive_diabetes_years}


def _parse_source(source):
    """Split a spec source "derive:<recipe>(<a>;<b>...)" into its parts.

    Args:
        source: str, e.g. "derive:coalesce(NUMADULT;HHADULT)".

    Returns:
        (recipe, args): str key of DERIVED and list of column names.
    """
    found = re.fullmatch(r"derive:(\w+)\(([^()]*)\)", source.strip())
    if found is None or found.group(1) not in DERIVED:
        raise ValueError("unknown derived source: %r" % source)
    args = [a.strip() for a in found.group(2).split(";") if a.strip()]
    return found.group(1), args


def add_derived(xc, x_raw, feature_names, derived=None):
    """Append the derived features (C4) to the cleaned matrix.

    Default features: hh_adults = NUMADULT if not NaN else HHADULT (cleaned),
    diabetes_years (see _derive_diabetes_years; uses the RAW DIABETE3 gate).

    Args:
        xc: numpy array of shape (N, D), cleaned and structurally filled.
        x_raw: numpy array of shape (N, D), raw values of the same columns.
        feature_names: list of D column names.
        derived: list of (name, source) pairs, or None for DERIVED_SOURCES.

    Returns:
        (x_out, names): array of shape (N, D + len(derived)) and the D + len(derived)
        column names.
    """
    if derived is None:
        derived = DERIVED_SOURCES
    ix = {name: j for j, name in enumerate(feature_names)}
    new_cols = []
    for _, source in derived:
        recipe, args = _parse_source(source)
        new_cols.append(DERIVED[recipe](xc, x_raw, ix, args))
    x_out = np.column_stack([xc] + new_cols) if new_cols else xc.copy()
    return x_out, list(feature_names) + [name for name, _ in derived]


def _read_spec_rows(spec):
    """Return the spec rows with the keys name, kind, source and group.

    Args:
        spec: list of dicts (load_feature_spec), a csv path, or None for
            SPEC_PATH.

    Returns:
        List of dicts with str values for name, kind, source, group.
    """
    if spec is None or isinstance(spec, str):
        with open(SPEC_PATH if spec is None else spec, newline="") as f:
            spec = list(csv.DictReader(f))
    keys = ("name", "kind", "source", "group")
    return [{k: (row.get(k) or "").strip() for k in keys} for row in spec]


def clean_data(x_raw, feature_names, spec=None, codes=None):
    """Stateless cleaning of raw data (stage C: C1 to C5).

    Each row uses only its own values, so this can run on train and test
    together without leakage.

    Args:
        x_raw: numpy array of shape (N, 321), raw values.
        feature_names: list of the 321 raw column names.
        spec: list of spec rows (load_feature_spec), csv path, or None for
            SPEC_PATH. Uses the fields name, kind, source and group.
        codes: dict from load_special_codes, or None to read CODES_PATH.

    Returns:
        (xc, names): xc of shape (N, K) with the K spec rows whose kind is not
        drop (raw and derived), in spec order, and their names.
    """
    rows = _read_spec_rows(spec)
    raw_names = set(feature_names)
    out_rows = [r for r in rows if r["kind"] != "drop"]
    derived = [(r["name"], r["source"]) for r in out_rows if r["source"]]
    # C1: subset of raw columns needed by the outputs, the fills and the recipes.
    needed = {r["name"] for r in out_rows if not r["source"]}
    needed |= {r["name"] for r in rows if r["group"] in ("state", "leaky")}
    for target, gates, _ in STRUCTURAL_FILLS:
        if target in needed:
            needed |= {gate for gate, _ in gates}
    for _, source in derived:
        needed |= set(_parse_source(source)[1])
    missing = sorted(n for n in needed if n not in raw_names)
    if missing:
        raise ValueError("columns not found in the raw data: %s" % missing)
    sub_idx = [j for j, name in enumerate(feature_names) if name in needed]
    sub_names = [feature_names[j] for j in sub_idx]
    x_sub = np.asarray(x_raw, dtype=np.float64)[:, sub_idx]
    # C2 to C4.
    xc = replace_special_codes(x_sub, sub_names, codes)
    xc = fill_structural(xc, x_sub, sub_names)
    xc, all_names = add_derived(xc, x_sub, sub_names, derived)
    # C5: non-drop rows in spec order.
    ix = {name: j for j, name in enumerate(all_names)}
    names = [r["name"] for r in out_rows]
    return xc[:, [ix[name] for name in names]], names


# ----------------------------------------------------------------------
# Feature spec, column selection and binary recode (F1-F2)


def load_feature_spec(path=None):
    """Read the normalized feature spec (eda/feature_spec.csv).

    Args:
        path: str or None, csv path. None uses the module constant SPEC_PATH.

    Returns:
        List of dict, one per spec row in file order, with keys name, source,
        kind, positive_values (list of float, empty unless kind is binary),
        missing_ind (int 0/1), clip (int 0/1), group and notes.
    """
    if path is None:
        path = SPEC_PATH
    spec = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            pos = row["positive_values"].strip()
            row["positive_values"] = [float(v) for v in pos.split("|")] if pos else []
            row["missing_ind"] = int(row["missing_ind"])
            row["clip"] = int(row["clip"])
            spec.append(row)
    return spec


def spec_by_name(spec):
    """Index the spec rows by feature name.

    Args:
        spec: list of dict, output of load_feature_spec.

    Returns:
        Dict name -> spec row.
    """
    return {row["name"]: row for row in spec}


def select_columns(xc_train, names, spec, config):
    """Choose the columns kept by the model, using training rows only (F1).

    A column is kept when its spec kind is not "drop", it is not listed in
    config["drop_features"], its group is enabled ("state" needs onehot_state,
    "leaky" needs include_leaky), its NaN share on train is at most
    max_nan_ratio (leaky rows are exempt) and it is not constant on train
    (a column without observed values counts as constant). Names missing from
    the spec are dropped.

    Args:
        xc_train: numpy array of shape (N, D), cleaned training rows.
        names: list of D column names of xc_train.
        spec: list of dict, output of load_feature_spec.
        config: dict with keys max_nan_ratio, onehot_state, include_leaky and
            drop_features (missing keys take the DEFAULT_CONFIG values).

    Returns:
        List of kept names, in the order of names.
    """
    x = np.asarray(xc_train, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError("xc_train must have shape (N, len(names))")
    rows = spec_by_name(spec)
    max_nan = config.get("max_nan_ratio", 0.95)
    groups_on = {
        "": True,
        "state": bool(config.get("onehot_state", False)),
        "leaky": bool(config.get("include_leaky", False)),
    }
    excluded = set(config.get("drop_features", []) or [])

    nan = np.isnan(x)
    nan_share = nan.mean(axis=0) if x.shape[0] > 0 else np.ones(x.shape[1])
    col_min = np.where(nan, np.inf, x).min(axis=0, initial=np.inf)
    col_max = np.where(nan, -np.inf, x).max(axis=0, initial=-np.inf)
    constant = ~(col_max > col_min)

    kept = []
    for j, name in enumerate(names):
        row = rows.get(name)
        if row is None or row["kind"] == "drop" or name in excluded:
            continue
        group = row["group"]
        if not groups_on.get(group, False):
            continue
        if group != "leaky" and nan_share[j] > max_nan:
            continue
        if constant[j]:
            continue
        kept.append(name)
    return kept


def binary_recode(x, kinds, positive_values):
    """Map binary columns to 1 (positive raw value) or 0, keeping NaN (F2).

    Only columns whose kind is "binary" change. The positive values are
    explicit per column (e.g. _RFDRHV5 is coded 1=no 2=yes, so its positive
    value is 2); there is no generic "1 -> 1" rule.

    Args:
        x: numpy array of shape (N, D).
        kinds: list of D str, the spec kind of each column.
        positive_values: list of D entries, a list of positive raw values for
            binary columns and None (or empty) for the others.

    Returns:
        Recoded copy of x, shape (N, D), float64.
    """
    out = np.array(x, dtype=np.float64)
    if (
        out.ndim != 2
        or out.shape[1] != len(kinds)
        or len(kinds) != len(positive_values)
    ):
        raise ValueError("kinds and positive_values must have one entry per column")
    for j, kind in enumerate(kinds):
        if kind != "binary":
            continue
        pos = positive_values[j]
        if not pos:
            raise ValueError("binary column %d has no positive value" % j)
        col = out[:, j]
        hit = np.isin(col, np.asarray(pos, dtype=np.float64)).astype(np.float64)
        out[:, j] = np.where(np.isnan(col), np.nan, hit)
    return out


# ----------------------------------------------------------------------
# Missing indicators, imputation and one-hot encoding (F3-F6)

IMPUTE_MEDIAN_KINDS = ("ordinal", "continuous", "continuous_log")
IMPUTE_MODE_KINDS = ("binary", "onehot")


def _spec_lookup(spec):
    """Return the feature spec as a dict name -> row.

    Args:
        spec: list of row dicts (load_feature_spec) or dict name -> row.

    Returns:
        dict mapping each feature name to its spec row.
    """
    if isinstance(spec, dict):
        return spec
    return {row["name"]: row for row in spec}


def nan_indicator_candidates(x_train, names, spec, config, y=None):
    """Choose the columns that get a missing indicator (F3), on train only.

    Works on the residual NaN (after the structural fill). Modes of
    config["nan_indicators"]:
        "auto": every column whose train NaN share s satisfies
            s > 0, s >= nan_indicator_min_share and s <= 1 - that share.
            If y is given, the column must also satisfy
            |P(y=1 | NaN) - P(y=1 | observed)| >= nan_indicator_min_effect.
        "spec": only the columns whose spec row has missing_ind == 1.
        "none": no indicators.
    Leaky columns (spec group "leaky") always get one in "auto" and "spec".

    Args:
        x_train: numpy array of shape (N, D), kept training columns with NaN.
        names: list of D column names.
        spec: list of spec row dicts, or dict name -> row.
        config: dict with "nan_indicators", "nan_indicator_min_share"
            (default 0.005) and "nan_indicator_min_effect" (default 0.02).
        y: numpy array of shape (N,), training labels in {-1, 1}, or None
            to skip the effect rule.

    Returns:
        numpy int array of column indices into x_train, in column order.
    """
    mode = config.get("nan_indicators", "auto")
    if mode == "none":
        return np.zeros(0, dtype=int)
    if mode not in ("auto", "spec"):
        raise ValueError("unknown nan_indicators mode: %r" % (mode,))
    rows = _spec_lookup(spec)
    leaky = np.array(
        [rows.get(n, {}).get("group", "") == "leaky" for n in names], dtype=bool
    )
    if mode == "spec":
        chosen = np.array(
            [int(rows.get(n, {}).get("missing_ind", 0)) == 1 for n in names],
            dtype=bool,
        )
    else:
        min_share = float(config.get("nan_indicator_min_share", 0.005))
        is_nan = np.isnan(x_train)
        n_nan = is_nan.sum(axis=0)
        n_rows = max(x_train.shape[0], 1)
        share = n_nan / n_rows
        chosen = (share > 0) & (share >= min_share) & (share <= 1.0 - min_share)
        if y is not None:
            min_effect = float(config.get("nan_indicator_min_effect", 0.02))
            pos = np.asarray(y).reshape(-1) == 1
            pos_nan = (is_nan & pos[:, None]).sum(axis=0)
            n_obs = x_train.shape[0] - n_nan
            pos_obs = pos.sum() - pos_nan
            defined = (n_nan > 0) & (n_obs > 0)
            rate_nan = pos_nan / np.maximum(n_nan, 1)
            rate_obs = pos_obs / np.maximum(n_obs, 1)
            effect = np.where(defined, np.abs(rate_nan - rate_obs), 0.0)
            chosen &= effect >= min_effect
    return np.flatnonzero(chosen | leaky).astype(int)


def make_nan_indicators(x, cols):
    """Build 0/1 missing indicators for the given columns.

    Args:
        x: numpy array of shape (N, D) with NaN (before imputation).
        cols: int array of column indices into x.

    Returns:
        numpy float array of shape (N, len(cols)), 1.0 where x is NaN.
    """
    cols = np.asarray(cols, dtype=int)
    return np.isnan(x[:, cols]).astype(np.float64)


def _column_std(x, chunk=64):
    """Population std of every column, computed in column chunks.

    Args:
        x: numpy array of shape (N, M).
        chunk: int, number of columns per chunk (limits temporary memory).

    Returns:
        numpy array of shape (M,).
    """
    out = np.empty(x.shape[1])
    for start in range(0, x.shape[1], chunk):
        out[start : start + chunk] = np.std(x[:, start : start + chunk], axis=0)
    return out


def dedup_indicators(ind_train, ref_train, max_abs_corr=0.99):
    """Greedy removal of redundant missing indicators (F6), on train only.

    In candidate order, an indicator is dropped if it is constant, or if its
    absolute Pearson correlation exceeds max_abs_corr with an indicator
    already kept or with any reference column (imputed binary and one-hot
    dummy columns). Constant reference columns are ignored.

    Args:
        ind_train: numpy array of shape (N, K), 0/1 indicators on train.
        ref_train: numpy array of shape (N, M) without NaN, or None.
        max_abs_corr: float, correlation threshold.

    Returns:
        numpy bool array of shape (K,), True for the indicators to keep.
    """
    ind = np.asarray(ind_train, dtype=np.float64)
    n_rows, n_ind = ind.shape
    keep = np.zeros(n_ind, dtype=bool)
    if n_ind == 0 or n_rows == 0:
        return keep
    centered = ind - ind.mean(axis=0)
    ind_std = np.sqrt((centered**2).mean(axis=0))
    valid = ind_std > 1e-12
    safe_std = np.where(valid, ind_std, 1.0)
    # Correlation between indicators.
    corr_ind = (centered.T @ centered) / n_rows / np.outer(safe_std, safe_std)
    # Max |corr| of each indicator with the reference columns. The centered
    # indicators sum to zero, so centered.T @ ref is the covariance times N
    # without centering (or copying) ref.
    max_ref = np.zeros(n_ind)
    if ref_train is not None and np.asarray(ref_train).shape[1] > 0:
        ref = np.asarray(ref_train, dtype=np.float64)
        ref_std = _column_std(ref)
        ref_ok = ref_std > 1e-12
        if ref_ok.any():
            cov = (centered.T @ ref[:, ref_ok]) / n_rows
            corr_ref = cov / np.outer(safe_std, ref_std[ref_ok])
            max_ref = np.abs(corr_ref).max(axis=1)
    for k in range(n_ind):
        if not valid[k] or max_ref[k] > max_abs_corr:
            continue
        if np.any(np.abs(corr_ind[k, keep]) > max_abs_corr):
            continue
        keep[k] = True
    return keep


def fit_imputer(x_train, kinds):
    """Compute the fill value of each column on the training rows (F4).

    Median of the observed values for ordinal, continuous and continuous_log
    columns; mode (smallest value on ties) for binary and onehot columns.
    A column without observed values gets 0.

    Args:
        x_train: numpy array of shape (N, D) with NaN.
        kinds: list of D spec kinds.

    Returns:
        numpy array of shape (D,), the fill values.
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    kinds = list(kinds)
    if len(kinds) != x_train.shape[1]:
        raise ValueError("kinds must have one entry per column")
    fill = np.zeros(x_train.shape[1])
    is_median = np.array([k in IMPUTE_MEDIAN_KINDS for k in kinds], dtype=bool)
    is_mode = np.array([k in IMPUTE_MODE_KINDS for k in kinds], dtype=bool)
    unknown = [k for k in kinds if k not in IMPUTE_MEDIAN_KINDS + IMPUTE_MODE_KINDS]
    if unknown:
        raise ValueError("cannot impute kinds: %s" % sorted(set(unknown)))
    observed = ~np.isnan(x_train)
    has_obs = observed.any(axis=0)
    med_cols = np.flatnonzero(is_median & has_obs)
    if med_cols.size:
        fill[med_cols] = np.nanmedian(x_train[:, med_cols], axis=0)
    for j in np.flatnonzero(is_mode & has_obs):
        levels, counts = np.unique(x_train[observed[:, j], j], return_counts=True)
        fill[j] = levels[np.argmax(counts)]
    return fill


def apply_imputer(x, fill_values):
    """Replace every NaN of x by the fill value of its column.

    Args:
        x: numpy array of shape (N, D) with NaN.
        fill_values: numpy array of shape (D,), from fit_imputer.

    Returns:
        float copy of x, shape (N, D), without NaN.
    """
    out = np.array(x, dtype=np.float64)
    fill = np.asarray(fill_values, dtype=np.float64)
    rows, cols = np.nonzero(np.isnan(out))
    out[rows, cols] = fill[cols]
    return out


def fit_one_hot(x_train, onehot_idx, names, min_share):
    """Fit the one-hot encoders on the imputed training rows (F5).

    The most frequent level (smallest on ties) is the reference and gets no
    dummy. Other levels with a train share below min_share are rare: they
    share one "other" dummy if their total share is at least min_share,
    otherwise they join the reference (all zeros).

    Args:
        x_train: numpy array of shape (N, D), imputed training data.
        onehot_idx: int array of the one-hot column indices into x_train.
        names: list of D column names.
        min_share: float, rare-level threshold (share of train rows).

    Returns:
        list with one dict per one-hot column: "name" (str), "levels" (sorted
        levels with their own dummy), "reference" (float), "rare" (sorted
        rare levels) and "has_other" (bool).
    """
    encoders = []
    n_rows = max(x_train.shape[0], 1)
    for j in np.asarray(onehot_idx, dtype=int):
        col = x_train[:, j]
        levels, counts = np.unique(col[~np.isnan(col)], return_counts=True)
        if levels.size == 0:
            encoders.append(
                {
                    "name": names[j],
                    "levels": [],
                    "reference": 0.0,
                    "rare": [],
                    "has_other": False,
                }
            )
            continue
        ref_pos = int(np.argmax(counts))
        share = counts / n_rows
        is_rare = share < min_share
        is_rare[ref_pos] = False
        own = ~is_rare
        own[ref_pos] = False
        encoders.append(
            {
                "name": names[j],
                "levels": [float(v) for v in levels[own]],
                "reference": float(levels[ref_pos]),
                "rare": [float(v) for v in levels[is_rare]],
                "has_other": bool(share[is_rare].sum() >= min_share),
            }
        )
    return encoders


def one_hot_encode(x, onehot_idx, encoders):
    """Encode the one-hot columns with encoders fitted on train.

    Each column gives one dummy per level in encoder "levels", then an
    "other" dummy if has_other. Rare, unseen and NaN values go to "other"
    if it exists, otherwise to all zeros like the reference.

    Args:
        x: numpy array of shape (N, D), imputed data.
        onehot_idx: int array of the one-hot column indices into x.
        encoders: list from fit_one_hot, same order as onehot_idx.

    Returns:
        dummies: numpy float array of shape (N, K).
        dummy_names: list of K names "NAME=%g" and "NAME=other".
    """
    onehot_idx = np.asarray(onehot_idx, dtype=int)
    if len(encoders) != onehot_idx.size:
        raise ValueError("one encoder per one-hot column is required")
    blocks = []
    dummy_names = []
    for j, enc in zip(onehot_idx, encoders):
        col = x[:, j]
        levels = np.asarray(enc["levels"], dtype=np.float64)
        if levels.size:
            blocks.append(col[:, None] == levels[None, :])
            dummy_names.extend("%s=%g" % (enc["name"], v) for v in levels)
        if enc["has_other"]:
            known = np.isin(col, levels) | (col == enc["reference"])
            blocks.append(~known[:, None])
            dummy_names.append("%s=other" % enc["name"])
    if not blocks:
        return np.zeros((x.shape[0], 0)), dummy_names
    return np.hstack(blocks).astype(np.float64), dummy_names


# ----------------------------------------------------------------------
# Tails, standardization, expansions and bias (F7-F11)


def log_transform(x, cols):
    """Apply x <- log1p(max(x, 0)) to the given columns (F7).

    Negative values become 0 before the log, NaN stays NaN.

    Args:
        x: numpy array of shape (N, D).
        cols: sequence of int, column indices to transform.

    Returns:
        Transformed copy of x, shape (N, D), float64.
    """
    out = np.array(x, dtype=np.float64, copy=True)
    cols = np.asarray(cols, dtype=int)
    if cols.size > 0:
        out[:, cols] = np.log1p(np.maximum(out[:, cols], 0.0))
    return out


def fit_clip(x_train, cols, quantile):
    """Compute the upper clipping bound of each column on the training data (F8).

    NaN values are ignored. A column without observed values gets +inf (no clip).

    Args:
        x_train: numpy array of shape (N, D), training rows only.
        cols: sequence of int, column indices to clip.
        quantile: float in (0, 1], e.g. 0.99.

    Returns:
        numpy array of shape (len(cols),), the train quantile of each column.
    """
    cols = np.asarray(cols, dtype=int)
    if cols.size == 0:
        return np.zeros(0, dtype=np.float64)
    sub = np.asarray(x_train, dtype=np.float64)[:, cols]
    upper = np.full(cols.size, np.inf)
    observed = np.any(~np.isnan(sub), axis=0)
    if np.any(observed):
        upper[observed] = np.nanquantile(sub[:, observed], quantile, axis=0)
    return upper


def apply_clip(x, cols, upper):
    """Clip the given columns from above with precomputed bounds (F8).

    Args:
        x: numpy array of shape (N, D).
        cols: sequence of int, column indices to clip.
        upper: numpy array of shape (len(cols),), from fit_clip.

    Returns:
        Clipped copy of x, shape (N, D), float64 (NaN stays NaN).
    """
    out = np.array(x, dtype=np.float64, copy=True)
    cols = np.asarray(cols, dtype=int)
    if cols.size > 0:
        out[:, cols] = np.minimum(out[:, cols], np.asarray(upper, dtype=np.float64))
    return out


def fit_standardizer(x_train):
    """Compute per-column mean and std on the training data (F9).

    Same formula as the lab helper standardize (np.std, ddof 0), per column.

    Args:
        x_train: numpy array of shape (N, D).

    Returns:
        mean: numpy array of shape (D,).
        std: numpy array of shape (D,), with 1 for constant columns.
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    if x_train.shape[0] > 0:
        constant = np.max(x_train, axis=0) == np.min(x_train, axis=0)
        std = np.where(constant | (std == 0), 1.0, std)
    else:
        std = np.ones_like(std)
    return mean, std


def standardize(x, mean, std):
    """Standardize x with precomputed mean and std (F9).

    Args:
        x: numpy array of shape (N, D).
        mean: numpy array of shape (D,).
        std: numpy array of shape (D,).

    Returns:
        numpy array of shape (N, D), (x - mean) / std.
    """
    return (np.asarray(x, dtype=np.float64) - mean) / std


def build_poly(x, degree):
    """Polynomial expansion of each feature (no cross terms): x, x^2, ..., x^d.

    Same as labs/ex03 build_poly without the bias column. The output is made of
    degree blocks of D columns: block k-1 holds x**k.

    Args:
        x: numpy array of shape (N, D) (or (N,), treated as (N, 1)).
        degree: int >= 1.

    Returns:
        numpy array of shape (N, D * degree).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if degree < 1:
        raise ValueError("degree must be >= 1")
    return np.hstack([np.power(x, deg) for deg in range(1, degree + 1)])


def build_interactions(x, pairs):
    """Products of pairs of columns (F10).

    Args:
        x: numpy array of shape (N, D).
        pairs: list of (i, j) int column indices.

    Returns:
        numpy array of shape (N, len(pairs)), column p = x[:, i_p] * x[:, j_p].
    """
    x = np.asarray(x, dtype=np.float64)
    if len(pairs) == 0:
        return np.zeros((x.shape[0], 0), dtype=np.float64)
    idx = np.asarray(pairs, dtype=int).reshape(-1, 2)
    return x[:, idx[:, 0]] * x[:, idx[:, 1]]


def add_bias(x):
    """Prepend a column of ones (the offset w_0) (F11).

    Args:
        x: numpy array of shape (N, D).

    Returns:
        numpy array of shape (N, D + 1), first column all ones.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.hstack([np.ones((x.shape[0], 1), dtype=np.float64), x])


# ----------------------------------------------------------------------
# Stage F: fit / transform pipeline

# Sections 4.5 and 5b of the design. The first 13 keys change the preprocessing
# (cross_validation.PREP_KEYS), the others are model keys.
DEFAULT_CONFIG = {
    "max_nan_ratio": 0.95,
    "onehot_state": False,
    "include_leaky": False,
    "drop_features": [],
    "nan_indicators": "auto",
    "nan_indicator_min_share": 0.005,
    "nan_indicator_min_effect": 0.02,
    "indicator_max_corr": 0.99,
    "rare_level_min_share": 0.005,
    "log_counts": True,
    "clip_quantile": None,
    "poly_degree": 1,
    "interactions": [],
    "seed": 42,
    "lambda_": 1e-4,
    "gamma": 0.5,
    "max_iters": 300,
    "threshold": None,
    "class_weight": None,
    "undersample_ratio": None,
}

POLY_KINDS = ("ordinal", "continuous", "continuous_log")


def _numeric_block(x_imp, state):
    """Non-onehot block after log1p (F7) and clipping (F8).

    Args:
        x_imp: numpy array of shape (N, D_kept), recoded and imputed.
        state: dict with numeric_idx, log_idx, clip_idx, clip_upper.

    Returns:
        numpy array of shape (N, len(numeric_idx)).
    """
    x_num = x_imp[:, state["numeric_idx"]]
    if len(state["log_idx"]):
        x_num = log_transform(x_num, state["log_idx"])
    if len(state["clip_idx"]):
        x_num = apply_clip(x_num, state["clip_idx"], state["clip_upper"])
    return x_num


def _expansions(z, state):
    """Polynomial powers and pairwise products of standardized columns (F10).

    Args:
        z: numpy array of shape (N, B), the standardized base block.
        state: dict with config["poly_degree"], poly_idx and pair_idx.

    Returns:
        numpy array of shape (N, E), not yet standardized (E may be 0).
    """
    blocks = []
    degree = int(state["config"]["poly_degree"])
    poly_idx = state["poly_idx"]
    if degree >= 2 and len(poly_idx):
        blocks.append(build_poly(z[:, poly_idx], degree)[:, len(poly_idx) :])
    if len(state["pair_idx"]):
        blocks.append(build_interactions(z, state["pair_idx"]))
    if not blocks:
        return np.zeros((z.shape[0], 0))
    return np.hstack(blocks)


def fit_preprocessor(
    xc_train, names, config=None, spec=None, y=None, return_train=False
):
    """Fit every stateful step (F1-F11) on the cleaned training rows.

    Index spaces stored in the state:
        * kept_idx: columns of xc (the cleaned input).
        * onehot_idx, numeric_idx, ind_cols: columns of the kept block
          xc[:, kept_idx].
        * log_idx, clip_idx, poly_idx: columns of the non-onehot block
          (kept columns taken in numeric_idx order).
        * pair_idx: columns of the standardized base block
          [non-onehot | dummies | indicators], without the bias.

    Args:
        xc_train: numpy array of shape (N, D), output of clean_data on the
            training rows only.
        names: list of D column names of xc_train.
        config: dict, overrides of DEFAULT_CONFIG (unknown keys ignored).
        spec: list of spec rows (load_feature_spec); None reads SPEC_PATH.
        y: numpy array of shape (N,), labels of the training rows in
            {-1, 1}, used only by the supervised indicator rule
            (nan_indicator_min_effect); None disables that rule.
        return_train: bool, also return the transformed training matrix
            (avoids a second transform of the same rows).

    Returns:
        state: dict read by transform (see the design, section 4.2).
        tx_train: numpy array of shape (N, len(state["out_names"])), only if
            return_train is True.
    """
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config or {})
    if spec is None:
        spec = load_feature_spec()
    rows = spec_by_name(spec)
    names = list(names)
    if xc_train.shape[1] != len(names):
        raise ValueError(
            "xc_train has %d columns, names %d" % (xc_train.shape[1], len(names))
        )

    # F1 column selection and F2 binary recode.
    kept_names = list(select_columns(xc_train, names, spec, cfg))
    kept_idx = np.array([names.index(n) for n in kept_names], dtype=int)
    kinds = [rows[n]["kind"] for n in kept_names]
    positive_values = [
        rows[n]["positive_values"] if k == "binary" else None
        for n, k in zip(kept_names, kinds)
    ]
    x = binary_recode(xc_train[:, kept_idx], kinds, positive_values)

    # F3 indicator candidates on the residual NaN, F4 imputation.
    cand = np.asarray(
        nan_indicator_candidates(x, kept_names, spec, cfg, y=y), dtype=int
    )
    ind = make_nan_indicators(x, cand)
    fill_values = fit_imputer(x, kinds)
    x = apply_imputer(x, fill_values)

    # F5 one-hot.
    onehot_idx = np.array([i for i, k in enumerate(kinds) if k == "onehot"], dtype=int)
    numeric_idx = np.array([i for i, k in enumerate(kinds) if k != "onehot"], dtype=int)
    encoders = fit_one_hot(x, onehot_idx, kept_names, cfg["rare_level_min_share"])
    dummies, dummy_names = one_hot_encode(x, onehot_idx, encoders)

    # F6 indicator dedup against kept indicators, binary columns and dummies.
    binary_idx = np.array([i for i, k in enumerate(kinds) if k == "binary"], dtype=int)
    ref = np.hstack([x[:, binary_idx], dummies])
    keep = dedup_indicators(ind, ref, cfg["indicator_max_corr"])
    ind_cols = cand[np.asarray(keep, dtype=bool)]
    del ref, ind

    # F7 / F8 positions inside the non-onehot block.
    num_names = [kept_names[i] for i in numeric_idx]
    num_kinds = [kinds[i] for i in numeric_idx]
    log_idx = np.array(
        [j for j, k in enumerate(num_kinds) if k == "continuous_log"]
        if cfg["log_counts"]
        else [],
        dtype=int,
    )
    clip_idx = np.array(
        [j for j, n in enumerate(num_names) if int(rows[n]["clip"]) == 1]
        if cfg["clip_quantile"] is not None
        else [],
        dtype=int,
    )
    state = {
        "config": cfg,
        "input_names": names,
        "kept_idx": kept_idx,
        "kept_names": kept_names,
        "kinds": kinds,
        "positive_values": positive_values,
        "fill_values": fill_values,
        "onehot_idx": onehot_idx,
        "encoders": encoders,
        "numeric_idx": numeric_idx,
        "ind_cols": ind_cols,
        "log_idx": log_idx,
        "clip_idx": clip_idx,
        "clip_upper": np.zeros(0),
    }
    x_num = x[:, numeric_idx]
    if len(log_idx):
        x_num = log_transform(x_num, log_idx)
    if len(clip_idx):
        state["clip_upper"] = fit_clip(x_num, clip_idx, cfg["clip_quantile"])
        x_num = apply_clip(x_num, clip_idx, state["clip_upper"])

    # F9 standardize the base block [numeric, dummies, indicators].
    ind_kept = make_nan_indicators(xc_train[:, kept_idx], ind_cols)
    base = np.hstack([x_num, dummies, ind_kept])
    del x, x_num, dummies, ind_kept
    base_names = (
        num_names + list(dummy_names) + [kept_names[c] + "_nan" for c in ind_cols]
    )
    mean, std = fit_standardizer(base)
    z = standardize(base, mean, std)
    del base
    state["mean"] = mean
    state["std"] = std

    # F10 expansions on the standardized block, standardized again.
    state["poly_idx"] = np.array(
        [j for j, k in enumerate(num_kinds) if k in POLY_KINDS], dtype=int
    )
    exp_names = []
    if int(cfg["poly_degree"]) >= 2:
        for deg in range(2, int(cfg["poly_degree"]) + 1):
            exp_names.extend("%s^%d" % (num_names[j], deg) for j in state["poly_idx"])
    pairs = []
    for a, b in cfg["interactions"]:
        if a not in base_names or b not in base_names:
            raise ValueError("interaction (%s, %s) not in the output columns" % (a, b))
        pairs.append((base_names.index(a), base_names.index(b)))
        exp_names.append("%s*%s" % (a, b))
    state["pair_idx"] = np.array(pairs, dtype=int).reshape(-1, 2)
    exp = _expansions(z, state)
    mean2, std2 = fit_standardizer(exp)
    state["mean2"] = mean2
    state["std2"] = std2
    state["out_names"] = ["bias"] + base_names + exp_names

    if not return_train:
        return state
    tx = add_bias(np.hstack([z, standardize(exp, mean2, std2)]))
    return state, tx


def transform(xc, state):
    """Apply a fitted preprocessor to cleaned rows (train, held-out or test).

    Args:
        xc: numpy array of shape (N, D), output of clean_data, same columns
            as the rows given to fit_preprocessor.
        state: dict returned by fit_preprocessor.

    Returns:
        tx: float64 numpy array of shape (N, len(state["out_names"])), first
            column all ones, no NaN or inf.
    """
    if xc.shape[1] != len(state["input_names"]):
        raise ValueError(
            "xc has %d columns, the state expects %d"
            % (xc.shape[1], len(state["input_names"]))
        )
    x_kept = xc[:, state["kept_idx"]]
    x = binary_recode(x_kept, state["kinds"], state["positive_values"])
    x = apply_imputer(x, state["fill_values"])
    dummies, _ = one_hot_encode(x, state["onehot_idx"], state["encoders"])
    x_num = _numeric_block(x, state)
    ind = make_nan_indicators(x_kept, state["ind_cols"])
    del x, x_kept
    base = np.hstack([x_num, dummies, ind])
    del x_num, dummies, ind
    z = standardize(base, state["mean"], state["std"])
    del base
    exp = _expansions(z, state)
    tx = add_bias(np.hstack([z, standardize(exp, state["mean2"], state["std2"])]))
    return np.asarray(tx, dtype=np.float64)


def preprocess(x_train, x_test, feature_names, config=None, y_train=None):
    """Full pipeline: clean both sets, fit every step on x_train, apply to both.

    Args:
        x_train: numpy array of shape (N, 321), raw training features.
        x_test: numpy array of shape (M, 321), raw test features.
        feature_names: list of 321 column names.
        config: dict with the preprocessing options (see DEFAULT_CONFIG).
        y_train: numpy array of shape (N,), training labels in {-1, 1}, for
            the supervised indicator rule (None disables it).

    Returns:
        tx_train: numpy array of shape (N, D').
        tx_test: numpy array of shape (M, D').
    """
    spec = load_feature_spec()
    xc_train, names = clean_data(x_train, feature_names, spec=spec)
    xc_test, _ = clean_data(x_test, feature_names, spec=spec)
    state, tx_train = fit_preprocessor(
        xc_train, names, config, spec=spec, y=y_train, return_train=True
    )
    del xc_train
    return tx_train, transform(xc_test, state)
