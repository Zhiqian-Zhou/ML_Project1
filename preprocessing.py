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
      codes and unit conversions, skip-logic fills, derived features, composite
      vascular multi-morbidity terms, and "don't know" flags.
    * fit_preprocessor (training rows) then transform (any rows): column
      selection, binary recode, missing indicators, median/mode imputation,
      one-hot encoding, log1p of counts, optional extra features,
      standardization, bias column.
    * leaky_features (only for the "with leakage" variant): the questions
      answered only after a heart attack, kept apart from the official model.
"""

import csv
import json
import os

import numpy as np

from helpers import load_csv_data

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC_PATH = os.path.join(MODULE_DIR, "eda", "feature_spec.csv")
CODES_PATH = os.path.join(MODULE_DIR, "eda", "special_codes.json")
RAW_KEYS = ("x_train", "x_test", "y_train", "train_ids", "test_ids")
NUMERIC_KINDS = ("ordinal", "continuous", "continuous_log")
MODE_KINDS = ("binary", "onehot")

COMPOSITE_COLUMNS = (
    "VASCULAR_BURDEN",
    "AGE_VASCULAR_SYNERGY",
    "METABOLIC_INDEX",
    "HEALTH_DAYS_BURDEN",
)


def _first_indices(names):
    """Index first occurrences, preserving list.index behavior for duplicates."""
    indices = {}
    for j, name in enumerate(names):
        indices.setdefault(name, j)
    return indices


def _required_indices(names, selected):
    """Resolve required columns once, retaining ValueError for absent names."""
    indices = _first_indices(names)
    result = []
    for name in selected:
        if name not in indices:
            raise ValueError(f"{name!r} is not in list")
        result.append(indices[name])
    return result


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

# Raw columns read by add_derived and composite clinical calculations.
DERIVED_INPUTS = [
    "NUMADULT",
    "HHADULT",
    "_AGE80",
    "DIABAGE2",
    "DIABETE3",
    "BPHIGH4",
    "_RFCHOL",
    "TOLDHI2",
    "_SMOKER3",
    "CVDSTRK3",
    "_BMI5",
    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",
]

# High-AUC physiological and demographic features (recovered from column_decisions)
HIGH_AUC_CDC_COLUMNS = [
    "MAXVO2_",
    "FC60_",
    "_AGEG5YR",
    "_RFHYPE5",
    "_RFCHOL",
    "_PNEUMO2",
    "_FLSHOT6",
    "_RFHLTH",
    "_AGE65YR",
]

HIGH_AUC_SENTINELS = {
    "MAXVO2_": [999, 99900],
    "FC60_": [999, 99900],
    "_AGEG5YR": [14, 77, 99],
    "_RFHYPE5": [9, 7, 77, 99],
    "_RFCHOL": [9, 7, 77, 99],
    "_PNEUMO2": [9, 7, 77, 99],
    "_FLSHOT6": [9, 7, 77, 99],
    "_RFHLTH": [9, 7, 77, 99],
    "_AGE65YR": [3, 9, 77, 99],
}

HIGH_AUC_KINDS = {
    "_RFHYPE5": "binary",
    "_RFCHOL": "binary",
    "_RFHLTH": "binary",
    "_AGE65YR": "binary",
    "_PNEUMO2": "binary",
    "_FLSHOT6": "binary",
    "_AGEG5YR": "ordinal",
    "MAXVO2_": "continuous",
    "FC60_": "continuous",
}

HIGH_AUC_POS_VALS = {
    "_RFHYPE5": (2.0,),
    "_RFCHOL": (2.0,),
    "_RFHLTH": (2.0,),
    "_AGE65YR": (2.0,),
    "_PNEUMO2": (1.0,),
    "_FLSHOT6": (1.0,),
    "_AGEG5YR": (),
    "MAXVO2_": (),
    "FC60_": (),
}


def extract_high_auc_features(x_raw, feature_names):
    """Extract and sanitize high-AUC CDC physiological and demographic features.

    Features:
        MAXVO2_ (AUC=0.7359), FC60_ (AUC=0.7359), _AGEG5YR (AUC=0.7354),
        _RFHYPE5 (AUC=0.6880), _RFCHOL (AUC=0.6762), _PNEUMO2 (AUC=0.6795),
        _FLSHOT6 (AUC=0.6737), _RFHLTH (AUC=0.6615), _AGE65YR (AUC=0.6706).
    """
    ix = {n: j for j, n in enumerate(feature_names)}
    N = x_raw.shape[0]
    cols, names = [], []
    for col in HIGH_AUC_CDC_COLUMNS:
        if col in ix:
            val = np.asarray(x_raw[:, ix[col]], dtype=np.float64).copy()
            sentinels = HIGH_AUC_SENTINELS.get(col, [7, 9, 77, 99, 999, 99900])
            val[np.isin(val, sentinels)] = np.nan
            val[val >= 99900] = np.nan
            cols.append(val)
        else:
            cols.append(np.full(N, np.nan, dtype=np.float64))
        names.append(col)
    return np.column_stack(cols), names


STRUCTURED_SKIP_KINDS = {
    "DIABEYE_CLEAN": "binary",
    "DOCTDIAB_CLEAN": "continuous",
    "CDASSIST_CLEAN": "ordinal",
    "CDHOUSE_CLEAN": "ordinal",
    "CDHELP_CLEAN": "ordinal",
    "CDSOCIAL_CLEAN": "ordinal",
}

STRUCTURED_SKIP_POS_VALS = {
    "DIABEYE_CLEAN": (1.0,),
    "DOCTDIAB_CLEAN": (),
    "CDASSIST_CLEAN": (),
    "CDHOUSE_CLEAN": (),
    "CDHELP_CLEAN": (),
    "CDSOCIAL_CLEAN": (),
}


def extract_structured_skip_features(x_raw, feature_names):
    """Extract and impute structural skip-logic features for diabetes complications & cognitive decline.

    Features:
        DIABEYE_CLEAN: Diabetic retinopathy (2 = No retinopathy if non-diabetic; 1 = Yes retinopathy)
        DOCTDIAB_CLEAN: Annual doctor visits for diabetes (0 if non-diabetic or none; 88=0)
        CDASSIST_CLEAN, CDHOUSE_CLEAN, CDHELP_CLEAN, CDSOCIAL_CLEAN: Cognitive decline limitations
            (5 = Never if no memory loss/confusion CIMEMLOS == 2)
    """
    ix = {n: j for j, n in enumerate(feature_names)}
    extra_cols = []
    extra_names = []
    no_diab = (
        np.isin(x_raw[:, ix["DIABETE3"]], [2, 3, 4])
        if "DIABETE3" in ix
        else None
    )
    no_ci = x_raw[:, ix["CIMEMLOS"]] == 2 if "CIMEMLOS" in ix else None

    # 1. DIABEYE (diabetic retinopathy) & DOCTDIAB (annual doctor visits)
    if "DIABEYE" in ix and "DIABETE3" in ix:
        diabeye = np.asarray(x_raw[:, ix["DIABEYE"]], dtype=np.float64).copy()
        diabeye[no_diab] = 2.0  # 2 = No retinopathy
        diabeye[np.isin(diabeye, [7, 9])] = np.nan
        extra_cols.append(diabeye)
        extra_names.append("DIABEYE_CLEAN")

    if "DOCTDIAB" in ix and "DIABETE3" in ix:
        docdiab = np.asarray(x_raw[:, ix["DOCTDIAB"]], dtype=np.float64).copy()
        docdiab[no_diab] = 0.0  # 0 visits
        docdiab[docdiab == 88] = 0.0  # 88 = None
        docdiab[np.isin(docdiab, [77, 99])] = np.nan
        extra_cols.append(docdiab)
        extra_names.append("DOCTDIAB_CLEAN")

    # 2. Cognitive Decline Module: CDASSIST, CDHOUSE, CDHELP, CDSOCIAL
    cd_cols = ["CDASSIST", "CDHOUSE", "CDHELP", "CDSOCIAL"]
    for cdc in cd_cols:
        if cdc in ix:
            col = np.asarray(x_raw[:, ix[cdc]], dtype=np.float64).copy()
            if no_ci is not None:
                col[no_ci] = 5.0  # 5 = Never
            col[np.isin(col, [7, 9])] = np.nan
            extra_cols.append(col)
            extra_names.append(cdc + "_CLEAN")

    if not extra_cols:
        return np.zeros((len(x_raw), 0), dtype=np.float64), []
    return np.column_stack(extra_cols), extra_names


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

    Checks if existing pre-parsed numpy cache exists for instant sub-second loading.
    Scopes the cache file to the data_path folder to avoid cross-dataset pollution.

    Args:
        data_path: str, folder containing the 3 csv files.
        sub_sample: bool, forwarded to load_csv_data (cache file raw_sub.npz).
        cache_dir: str or None, cache folder. None uses <data_path>/../cache.

    Returns:
        tuple: (x_train, x_test, y_train, train_ids, test_ids).
    """
    base_dir = os.path.dirname(os.path.abspath(data_path))
    if cache_dir is None:
        cache_dir = os.path.join(base_dir, "cache")
    folder_tag = os.path.basename(os.path.normpath(data_path))
    cache_name = "raw_sub.npz" if sub_sample else "raw.npz"
    if folder_tag and folder_tag != "dataset":
        cache_name = f"{folder_tag}_{cache_name}"
    cache = os.path.join(cache_dir, cache_name)
    if not os.path.exists(cache):
        arrays = load_csv_data(data_path, sub_sample)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, **dict(zip(RAW_KEYS, arrays)))
    with np.load(cache) as data:
        return tuple(data[key] for key in RAW_KEYS)


def load_special_codes():
    """Read eda/special_codes.json: per-column special values and transforms."""
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
    """Return a dict mapping feature name to its spec row."""
    return {r["name"]: r for r in spec}


# ----------------------------------------------------------------------
# Stateless transforms (row by row, no training statistics)


def convert_units(raw, kind):
    """Turn human-coded BRFSS questions into real physical quantities.

    * Height: 4-digit feet and inches (300-711) -> centimeters;
              9000-9998 -> already centimeters (offset 9000).
    * Weight: 50-776 -> pounds -> kilograms;
              9000-9998 -> already kilograms (offset 9000).
    * Days per month (30 days): 101-199 -> days/week -> days/month (x 30/7);
              201-230 -> days/month directly.
    * Frequency per year: 101-199 -> per day (x 365.25);
              201-299 -> per week (x 52.18);
              301-399 -> per month (x 12);
              401-499 -> per year.
    * Drinks per week: 101-199 -> days/week x AVEDRNK2;
              201-299 -> days/month x AVEDRNK2 / (30/7).

    Returns:
        numpy float64 array of shape (N,); values that did not match the unit
        pattern are left as NaN.
    """
    out = np.full(raw.shape, np.nan, dtype=np.float64)
    if kind == "height_cm":
        ft_in = (raw >= 200) & (raw <= 711)
        ft = raw[ft_in] // 100
        inch = raw[ft_in] % 100
        out[ft_in] = (ft * 12.0 + inch) * 2.54
        metric = (raw >= 9000) & (raw <= 9998)
        out[metric] = raw[metric] - 9000.0
    elif kind == "weight_kg":
        lbs = (raw >= 50) & (raw <= 776)
        out[lbs] = raw[lbs] * 0.45359237
        metric = (raw >= 9000) & (raw <= 9998)
        out[metric] = raw[metric] - 9000.0
    elif kind == "days30":
        per_week = (raw >= 101) & (raw <= 199)
        out[per_week] = np.minimum((raw[per_week] - 100.0) * (30.0 / 7.0), 30.0)
        per_month = (raw >= 201) & (raw <= 230)
        out[per_month] = raw[per_month] - 200.0
    elif kind == "freq_year":
        d = (raw >= 101) & (raw <= 199)
        out[d] = (raw[d] - 100.0) * 365.25
        w = (raw >= 201) & (raw <= 299)
        out[w] = (raw[w] - 200.0) * 52.18
        m = (raw >= 301) & (raw <= 399)
        out[m] = (raw[m] - 300.0) * 12.0
        y = (raw >= 401) & (raw <= 499)
        out[y] = raw[y] - 400.0
    return out


def replace_special_codes(x, feature_names, codes):
    """Replace codebook special values in x according to special_codes.json.

    For every column:
        1. Replace its nan_codes with NaN.
        2. If it has a unit transform (e.g. height_cm), apply convert_units.
        3. If it has zero_codes (e.g. 88 = "none"), set them to 0.
        4. If it has recode (e.g. 1-76 -> 1-76), apply it.
        5. If it has scale (e.g. _BMI5 is in units of 100), multiply by it.

    Returns:
        A copy of x with the clean values (float64).
    """
    x = np.asarray(x, dtype=np.float64).copy()
    for j, name in enumerate(feature_names):
        rule = codes.get(name)
        if not rule:
            continue
        raw = x[:, j].copy()
        x[np.isin(raw, rule.get("nan_codes", [])), j] = np.nan
        if "transform" in rule:
            x[:, j] = convert_units(raw, rule["transform"])
        x[np.isin(raw, rule.get("zero_codes", [])), j] = 0.0
        for code_value, new_value in rule.get("recode", {}).items():
            x[raw == float(code_value), j] = new_value
        x[:, j] *= rule.get("scale", 1.0)
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


def add_derived(xc, x_raw, feature_names, with_composite=True):
    """Append the derived features hh_adults, diabetes_years, and 4 composite clinical indicators.

    hh_adults: adults in the household, landline NUMADULT else cell HHADULT.
    diabetes_years: max(_AGE80 - DIABAGE2, 0) if the raw DIABETE3 is 1
    (diabetic), 0 if it is 2, 3 or 4 (not diabetic), NaN otherwise.

    Composite indicators:
    - VASCULAR_BURDEN: (BPHIGH4 == 1) + (_RFCHOL == 2) + (DIABETE3 == 1) + (_SMOKER3 in [1, 2]) + (CVDSTRK3 == 1)
    - AGE_VASCULAR_SYNERGY: standardized(_AGE80) * VASCULAR_BURDEN
    - METABOLIC_INDEX: standardized(_BMI5) * GENHLTH
    - HEALTH_DAYS_BURDEN: PHYSHLTH + MENTHLTH (with 88 mapped to 0)

    Returns:
        (x_out, names): xc with the added derived columns, and their names.
    """
    ix = {name: j for j, name in enumerate(feature_names)}
    N = xc.shape[0]

    # hh_adults
    if "NUMADULT" in ix and "HHADULT" in ix:
        hh_adults = xc[:, ix["NUMADULT"]].copy()
        missing = np.isnan(hh_adults)
        hh_adults[missing] = xc[missing, ix["HHADULT"]]
    else:
        hh_adults = np.full(N, np.nan)

    # diabetes_years
    if "DIABETE3" in ix and "_AGE80" in ix and "DIABAGE2" in ix:
        status = x_raw[:, ix["DIABETE3"]]
        diabetic = status == 1
        years = np.full(N, np.nan)
        age_gap = xc[diabetic, ix["_AGE80"]] - xc[diabetic, ix["DIABAGE2"]]
        years[diabetic] = np.maximum(age_gap, 0.0)
        years[np.isin(status, [2, 3, 4])] = 0.0
    else:
        years = np.full(N, np.nan)

    cols = [xc, hh_adults, years]
    names = list(feature_names) + ["hh_adults", "diabetes_years"]

    if with_composite:
        # 1. VASCULAR_BURDEN
        bphigh = (
            (x_raw[:, ix["BPHIGH4"]] == 1) if "BPHIGH4" in ix else np.zeros(N, bool)
        )
        rfchol = (
            (x_raw[:, ix["_RFCHOL"]] == 2)
            if "_RFCHOL" in ix
            else (
                (x_raw[:, ix["TOLDHI2"]] == 1) if "TOLDHI2" in ix else np.zeros(N, bool)
            )
        )
        diab = (
            (x_raw[:, ix["DIABETE3"]] == 1) if "DIABETE3" in ix else np.zeros(N, bool)
        )
        smoke = (
            np.isin(x_raw[:, ix["_SMOKER3"]], [1, 2])
            if "_SMOKER3" in ix
            else np.zeros(N, bool)
        )
        stroke = (
            (x_raw[:, ix["CVDSTRK3"]] == 1) if "CVDSTRK3" in ix else np.zeros(N, bool)
        )

        vascular_burden = (
            bphigh.astype(np.float64)
            + rfchol.astype(np.float64)
            + diab.astype(np.float64)
            + smoke.astype(np.float64)
            + stroke.astype(np.float64)
        )

        # 2. AGE_VASCULAR_SYNERGY
        if "_AGE80" in ix:
            age = xc[:, ix["_AGE80"]].copy()
            age_mean = float(np.nanmean(age)) if np.any(~np.isnan(age)) else 54.0
            age_std = float(np.nanstd(age)) if np.any(~np.isnan(age)) else 17.0
            if age_std == 0.0 or np.isnan(age_std):
                age_std = 1.0
            age_norm = (age - age_mean) / age_std
            age_vascular = age_norm * vascular_burden
        else:
            age_vascular = vascular_burden.copy()

        # 3. METABOLIC_INDEX
        if "_BMI5" in ix and "GENHLTH" in ix:
            bmi = xc[:, ix["_BMI5"]].copy()
            bmi_mean = float(np.nanmean(bmi)) if np.any(~np.isnan(bmi)) else 28.0
            bmi_std = float(np.nanstd(bmi)) if np.any(~np.isnan(bmi)) else 6.0
            if bmi_std == 0.0 or np.isnan(bmi_std):
                bmi_std = 1.0
            bmi_norm = (bmi - bmi_mean) / bmi_std
            genhlth = xc[:, ix["GENHLTH"]].copy()
            metabolic_index = bmi_norm * genhlth
        else:
            metabolic_index = np.zeros(N, dtype=np.float64)

        # 4. HEALTH_DAYS_BURDEN
        phys = xc[:, ix["PHYSHLTH"]].copy() if "PHYSHLTH" in ix else np.zeros(N)
        ment = xc[:, ix["MENTHLTH"]].copy() if "MENTHLTH" in ix else np.zeros(N)
        phys_clean = np.where(np.isnan(phys), 0.0, phys)
        ment_clean = np.where(np.isnan(ment), 0.0, ment)
        health_days = phys_clean + ment_clean

        cols.extend([vascular_burden, age_vascular, metabolic_index, health_days])
        names.extend(COMPOSITE_COLUMNS)

    x_out = np.column_stack(cols)
    return x_out, names


def clean_data(
    x_raw,
    feature_names,
    spec=None,
    codes=None,
    keep_raw=(),
    with_composite=True,
):
    """Stateless cleaning of raw data (train and test can be cleaned apart).

    Args:
        x_raw: numpy array of shape (N, 321), raw values.
        feature_names: list of the 321 raw column names.
        spec: list of spec rows (load_feature_spec), or None to read SPEC_PATH.
        codes: dict from load_special_codes, or None to read CODES_PATH.
        keep_raw: names of dropped raw columns to append unchanged at the end
            (e.g. ("_STATE",) for tree models). fit_preprocessor ignores them.
        with_composite: bool, whether to include the 4 composite clinical indicators.

    Returns:
        (xc, names): the K spec rows whose kind is not drop (raw and derived,
        in spec order), optional composite indicators, then one 0/1 flag "<name>_dk"
        per raw kept column that has "don't know / refused" codes, then keep_raw columns.
    """
    spec = load_feature_spec() if spec is None else spec
    codes = load_special_codes() if codes is None else codes
    out_names = [r["name"] for r in spec if r["kind"] != "drop"]
    assert not set(keep_raw) & set(out_names), "keep_raw takes dropped columns"
    raw_out = [n for n in out_names if n in feature_names]

    needed = set(raw_out) | set(DERIVED_INPUTS)
    for target, gates, _ in STRUCTURAL_FILLS:
        if target in needed:
            needed |= {gate for gate, _ in gates}
    sub_idx = [j for j, name in enumerate(feature_names) if name in needed]
    sub_names = [feature_names[j] for j in sub_idx]
    x_sub = np.asarray(x_raw, dtype=np.float64)[:, sub_idx]
    xc = replace_special_codes(x_sub, sub_names, codes)
    xc = fill_structural(xc, x_sub, sub_names)
    xc, all_names = add_derived(xc, x_sub, sub_names, with_composite=with_composite)
    ix = {name: j for j, name in enumerate(all_names)}

    selected_names = [n for n in out_names if n in ix]
    if with_composite:
        for cname in COMPOSITE_COLUMNS:
            if cname in ix and cname not in selected_names:
                selected_names.append(cname)

    high_auc_cols, high_auc_names = extract_high_auc_features(x_raw, feature_names)
    struct_cols, struct_names = extract_structured_skip_features(x_raw, feature_names)

    dk_names = [n for n in raw_out if codes.get(n, {}).get("nan_codes")]
    dk = [np.isin(x_sub[:, ix[n]], codes[n]["nan_codes"]) for n in dk_names]
    extra = [x_raw[:, j] for j in _required_indices(feature_names, keep_raw)]

    xc_selected = xc[:, [ix[n] for n in selected_names]]
    xc_blocks = [xc_selected, high_auc_cols, struct_cols]
    if dk:
        xc_blocks.append(np.column_stack(dk))
    if extra:
        xc_blocks.append(np.column_stack(extra))

    xc_out = np.column_stack(xc_blocks)
    final_names = (
        selected_names
        + high_auc_names
        + struct_names
        + [n + "_dk" for n in dk_names]
        + list(keep_raw)
    )
    return xc_out, final_names


# ----------------------------------------------------------------------
# Leaky questions: only for the "with leakage" variant (run.py --with_leakage)

LEAKY_COLUMNS = [
    "HAREHAB1",  # outpatient rehabilitation after the heart attack
    "STREHAB1",  # outpatient rehabilitation after the stroke
    "CVDASPRN",  # takes aspirin daily or every other day
    "RDUCHART",  # aspirin to reduce the chance of a heart attack
    "RDUCSTRK",  # aspirin to reduce the chance of a stroke
    "RLIVPAIN",  # aspirin to relieve pain
    "ASPUNSAF",  # aspirin unsafe: 1 yes, 2 yes (stomach), 3 no
]
LEAKY_LEVELS = {"ASPUNSAF": (1, 2, 3)}  # answer codes; the others are 1 yes, 2 no


def leaky_features(x_raw, feature_names, codes=None):
    """Stateless features of the leaky questions (the leakage variant only).

    Args:
        x_raw: numpy array of shape (N, 321), raw values.
        feature_names: list of the 321 raw column names.
        codes: dict from load_special_codes, or None to read CODES_PATH.

    Returns:
        (x_tree, tree_names, x_flags, flag_names):
        * x_tree: matrix for gbm.py (cleaned, NaN preserved, followed by a
          "was answered" 0/1 flag per column);
        * x_flags: standardized 0/1 flags for mlp.py (one per non-missing
          answer code, e.g. "HAREHAB1=1", and one "was answered" flag).
    """
    codes = load_special_codes() if codes is None else codes
    cols = _required_indices(feature_names, LEAKY_COLUMNS)
    raw = np.asarray(x_raw, dtype=np.float64)[:, cols]
    clean = replace_special_codes(raw, LEAKY_COLUMNS, codes)
    answered = (~np.isnan(clean)).astype(np.float64)
    x_tree = np.hstack([clean, answered])
    tree_names = list(LEAKY_COLUMNS) + [n + "_answered" for n in LEAKY_COLUMNS]

    flags, flag_names = [], []
    for j, name in enumerate(LEAKY_COLUMNS):
        levels = LEAKY_LEVELS.get(name, (1, 2))
        for level in levels:
            flags.append((clean[:, j] == level).astype(np.float64))
            flag_names.append(f"{name}={level}")
        flags.append(answered[:, j])
        flag_names.append(f"{name}_answered")
    x_flags = np.column_stack(flags)
    return x_tree, tree_names, x_flags, flag_names


TARGET_ENC_COLUMNS = [
    "GENHLTH",
    "CVDSTRK3",
    "DIABETE3",
    "_AGEG5YR",
    "INCOME2",
    "EMPLOY1",
]


def add_target_risk_encodings(
    y_tr, xc_tr, xc_te, names, target_cols=TARGET_ENC_COLUMNS, m_smooth=50.0
):
    """Compute empirical Bayes smoothed risk encoding on key categorical variables.

    Fitted strictly on y_tr labels to guarantee zero data leakage.
    Smooth prevalence = (pos + m * prior) / (cnt + m).
    """
    y01_tr = (y_tr == 1).astype(np.float64)
    prior = float(np.mean(y01_tr))
    enc_tr_list, enc_te_list, enc_names = [], [], []
    indices = _first_indices(names)
    for col_name in target_cols:
        if col_name in indices:
            idx = indices[col_name]
            col_tr = xc_tr[:, idx]
            col_te = xc_te[:, idx]
            cats, inv = np.unique(col_tr, return_inverse=True)
            pos = np.bincount(inv, weights=y01_tr)
            cnt = np.bincount(inv)
            smooth = (pos + m_smooth * prior) / (cnt + m_smooth)
            pos_idx = np.clip(np.searchsorted(cats, col_te), 0, len(cats) - 1)
            enc_te = np.where(cats[pos_idx] == col_te, smooth[pos_idx], prior)
            enc_tr_list.append(smooth[inv])
            enc_te_list.append(enc_te)
            enc_names.append(f"{col_name}_RISK")
    if not enc_tr_list:
        return xc_tr, xc_te, names
    # Insert new continuous risk features directly before _STATE
    new_tr = np.hstack([xc_tr[:, :-1], np.column_stack(enc_tr_list), xc_tr[:, -1:]])
    new_te = np.hstack([xc_te[:, :-1], np.column_stack(enc_te_list), xc_te[:, -1:]])
    new_names = names[:-1] + enc_names + [names[-1]]
    return new_tr, new_te, new_names


# ----------------------------------------------------------------------
# Stateful preprocessor (fitted on the training rows only)


def select_columns(xc_train, names, rows, max_nan_ratio):
    """Kept names: spec kind not drop, train NaN share <= max_nan_ratio, not constant."""
    nan = np.isnan(xc_train)
    nan_share = nan.mean(axis=0)
    col_min = np.where(nan, np.inf, xc_train).min(axis=0)
    col_max = np.where(nan, -np.inf, xc_train).max(axis=0)
    kept = []
    for j, name in enumerate(names):
        # Allow high-AUC features, structured skip features, composite features, or spec non-drop features
        if name in HIGH_AUC_CDC_COLUMNS or name.endswith("_CLEAN"):
            pass
        elif name in COMPOSITE_COLUMNS:
            pass
        elif name in rows:
            if rows[name]["kind"] == "drop":
                continue
        else:
            continue
        if nan_share[j] <= max_nan_ratio and col_max[j] > col_min[j]:
            kept.append(name)
    return kept


def binary_recode(x, kinds, positive_values):
    """Map binary columns to 1 (positive values) or 0, keep NaN."""
    x = x.copy()
    for col, (kind, pos) in enumerate(zip(kinds, positive_values)):
        if kind != "binary":
            continue
        valid = ~np.isnan(x[:, col])
        is_pos = np.isin(x[valid, col], pos)
        x[valid, col] = is_pos.astype(np.float64)
    return x


def nan_indicator_candidates(x_train, config, y=None):
    """Index of columns in x_train that qualify for a NaN indicator column."""
    cfg = config
    mode = cfg["nan_indicators"]
    if mode is None:
        return np.array([], dtype=int)
    nan = np.isnan(x_train)
    share = nan.mean(axis=0)
    eligible = (share >= cfg["nan_indicator_min_share"]) & (
        share <= 1.0 - cfg["nan_indicator_min_share"]
    )
    if mode == "all":
        return np.where(eligible)[0]
    if y is None:
        return np.where(eligible)[0]
    y_pos = y == 1
    p_pos_if_nan = np.zeros(x_train.shape[1])
    p_pos_if_obs = np.zeros(x_train.shape[1])
    for c in np.where(eligible)[0]:
        m = nan[:, c]
        p_pos_if_nan[c] = np.mean(y_pos[m])
        p_pos_if_obs[c] = np.mean(y_pos[~m])
    effect = np.abs(p_pos_if_nan - p_pos_if_obs)
    return np.where(eligible & (effect >= cfg["nan_indicator_min_effect"]))[0]


def dedup_indicators(ind_train, ref_train, max_abs_corr):
    """Select a subset of indicators whose absolute correlation is <= max_abs_corr."""
    if ind_train.shape[1] == 0:
        return np.array([], dtype=int)
    ref_std = np.std(ref_train, axis=0)
    active = ref_std > 1e-12
    ref_z = (
        (ref_train[:, active] - np.mean(ref_train[:, active], axis=0)) / ref_std[active]
        if np.any(active)
        else np.zeros((len(ind_train), 0))
    )
    kept = []
    kept_z = []
    n = len(ind_train)
    for i in range(ind_train.shape[1]):
        v = ind_train[:, i]
        s = np.std(v)
        if s < 1e-12:
            continue
        z = (v - np.mean(v)) / s
        if ref_z.shape[1] > 0 and np.max(np.abs(z @ ref_z / n)) > max_abs_corr:
            continue
        if kept_z and np.max(np.abs(z @ np.column_stack(kept_z) / n)) > max_abs_corr:
            continue
        kept.append(i)
        kept_z.append(z)
    return np.array(kept, dtype=int)


def fit_imputer(x_train, kinds):
    """Median of numeric columns and mode of categorical columns on non-NaN rows."""
    vals = np.zeros(x_train.shape[1], dtype=np.float64)
    for c, kind in enumerate(kinds):
        col = x_train[:, c]
        obs = col[~np.isnan(col)]
        if len(obs) == 0:
            vals[c] = 0.0
        elif kind in NUMERIC_KINDS or kind == "binary":
            vals[c] = float(np.median(obs))
        else:
            u, counts = np.unique(obs, return_counts=True)
            vals[c] = float(u[np.argmax(counts)])
    return vals


def apply_imputer(x, fill_values):
    """Replace NaN in x by fill_values column by column."""
    x = x.copy()
    nan = np.isnan(x)
    for c in range(x.shape[1]):
        if np.any(nan[:, c]):
            x[nan[:, c], c] = fill_values[c]
    return x


def fit_one_hot(x_train, onehot_idx, names, min_share):
    """For each one-hot column, the unique levels with share >= min_share."""
    encoders = {}
    for c in onehot_idx:
        col = x_train[:, c]
        u, counts = np.unique(col, return_counts=True)
        levels = u[counts / len(col) >= min_share]
        if len(levels) > 1:
            encoders[c] = levels[:-1]
        elif len(levels) == 1:
            encoders[c] = levels
        else:
            encoders[c] = u[:1]
    return encoders


def one_hot_encode(x, onehot_idx, encoders):
    """Replace each column in onehot_idx with 0/1 dummy columns."""
    if not onehot_idx:
        return np.zeros((len(x), 0), dtype=np.float64), []
    blocks, names = [], []
    for c in onehot_idx:
        for lvl in encoders.get(c, []):
            blocks.append((x[:, c] == lvl).astype(np.float64))
            lvl_str = str(int(lvl)) if lvl == int(lvl) else str(lvl)
            names.append(f"col_{c}={lvl_str}")
    if not blocks:
        return np.zeros((len(x), 0), dtype=np.float64), []
    return np.column_stack(blocks), names


def fit_standardizer(x_train):
    """Mean and standard deviation of each column (std=0 replaced by 1)."""
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std[std == 0.0] = 1.0
    return mean, std


def standardize(x, mean, std):
    """(x - mean) / std column by column."""
    return (x - mean) / std


def build_poly(x, degree):
    """Polynomial expansion [x, x^2, ..., x^degree]."""
    if degree <= 1:
        return x
    return np.hstack([x**d for d in range(1, degree + 1)])


def add_bias(x):
    """Prepend a column of ones."""
    return np.c_[np.ones(len(x)), x]


def corr_dedup(z, names, y, max_corr):
    """Drop the feature of each correlated pair that has smaller correlation with y."""
    n = len(z)
    corrs_y = np.abs(z.T @ y / n) if y is not None else np.zeros(z.shape[1])
    keep = np.ones(z.shape[1], dtype=bool)
    R = np.abs(z.T @ z / n)
    np.fill_diagonal(R, 0.0)
    for i in range(z.shape[1]):
        if not keep[i]:
            continue
        too_high = np.where(keep & (R[i] > max_corr))[0]
        for j in too_high:
            if j <= i:
                continue
            drop = j if corrs_y[i] >= corrs_y[j] else i
            keep[drop] = False
            if drop == i:
                break
    return keep


def _feature_metadata(name, rows):
    """Resolve kind and positive values with the existing override precedence."""
    if name in HIGH_AUC_KINDS:
        return HIGH_AUC_KINDS[name], HIGH_AUC_POS_VALS[name]
    if name in STRUCTURED_SKIP_KINDS:
        return STRUCTURED_SKIP_KINDS[name], STRUCTURED_SKIP_POS_VALS[name]
    row = rows.get(name)
    if row is not None and row["kind"] != "drop":
        return row["kind"], row["positive_values"]
    return "continuous", ()


def _fit_columns(xc_train, names, rows, cfg, y):
    """Fit column selection, imputation, one-hot and missing indicators."""
    kept = select_columns(xc_train, names, rows, cfg["max_nan_ratio"])
    metadata = [_feature_metadata(n, rows) for n in kept]
    kinds = [kind for kind, _ in metadata]
    pos_vals = [pos for _, pos in metadata]
    state = {
        "kept_idx": np.array(_required_indices(names, kept), dtype=int),
        "kept_names": kept,
        "kinds": kinds,
        "positive_values": pos_vals,
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
    """Identify which "_dk" columns pass the variance threshold."""
    dk_idx, n_dk_idx = [], []
    if cfg["dont_know_indicators"] or cfg["n_dont_know"]:
        for j, name in enumerate(names):
            if not name.endswith("_dk"):
                continue
            share = np.mean(xc_train[:, j] == 1)
            if share >= cfg["dont_know_min_share"]:
                dk_idx.append(j)
            if share > 0:
                n_dk_idx.append(j)
    state["dk_idx"] = (
        np.array(dk_idx, dtype=int)
        if cfg["dont_know_indicators"]
        else np.array([], dtype=int)
    )
    state["n_dk_idx"] = (
        np.array(n_dk_idx, dtype=int) if cfg["n_dont_know"] else np.array([], dtype=int)
    )


def _base_block(xc, state):
    """Build the base design matrix before interaction and final standardization."""
    cfg = state["config"]
    x_kept = xc[:, state["kept_idx"]]
    x = binary_recode(x_kept, state["kinds"], state["positive_values"])
    nan_mask = np.isnan(x_kept)
    x = apply_imputer(x, state["fill_values"])

    num_cols = state["numeric_idx"]
    x_num = x[:, num_cols].copy()
    num_names = [state["kept_names"][i] for i in num_cols]
    if cfg["log_counts"]:
        for j, name in enumerate(num_names):
            kind = state["kinds"][num_cols[j]]
            if kind == "continuous_log":
                x_num[:, j] = np.log1p(np.maximum(x_num[:, j], 0.0))
    if cfg["poly_degree"] > 1:
        x_num = build_poly(x_num, cfg["poly_degree"])
        deg_names = []
        for d in range(1, cfg["poly_degree"] + 1):
            deg_names += [f"{n}^{d}" if d > 1 else n for n in num_names]
        num_names = deg_names

    dummies, dummy_names = one_hot_encode(x, state["onehot_idx"], state["encoders"])
    kept = state["kept_names"]

    blocks = [x_num, dummies]
    if len(state["ind_cols"]) > 0:
        blocks.append(nan_mask[:, state["ind_cols"]].astype(np.float64))
    names = num_names + dummy_names + [kept[c] + "_nan" for c in state["ind_cols"]]

    if len(state["dk_idx"]) > 0:
        blocks.append(xc[:, state["dk_idx"]])
        names += [state["input_names"][j] for j in state["dk_idx"]]
    if len(state["n_dk_idx"]) > 0:
        blocks.append(xc[:, state["n_dk_idx"]].sum(axis=1, keepdims=True))
        names.append("n_dk")
    if cfg["age_hinges"] and "_AGE80" in kept:
        age = x[:, kept.index("_AGE80")]
        blocks.append(np.column_stack([np.maximum(age - k, 0.0) for k in AGE_KNOTS]))
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
        y: numpy array of shape (N,), training labels in {-1, 1}.
        return_train: bool, also return transform(xc_train, state).

    Returns:
        state: dict read by transform; state["out_names"] names the columns.
        tx_train: only if return_train.
    """
    cfg = merge_config(config)
    rows = spec_by_name(load_feature_spec() if spec is None else spec)
    names = list(names)
    state = _fit_columns(xc_train, names, rows, cfg, y)
    state["config"] = cfg
    state["input_names"] = names
    _fit_dont_know(xc_train, names, state, cfg)
    base, base_names = _base_block(xc_train, state)
    state["mean"], state["std"] = fit_standardizer(base)
    z = standardize(base, state["mean"], state["std"])
    del base

    state["pairs"] = []
    if cfg["interactions"]:
        base_indices = _first_indices(base_names)
        state["pairs"] = [
            (base_indices[a], base_indices[b])
            for a, b in INTERACTIONS
            if a in base_indices and b in base_indices
        ]
    if state["pairs"]:
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
    """Transform arbitrary rows using statistics of the training rows."""
    base, _ = _base_block(xc, state)
    raw_z = standardize(base, state["mean"], state["std"])
    z = raw_z[:, state["keep"]]
    if state["pairs"]:
        products = np.column_stack(
            [raw_z[:, a] * raw_z[:, b] for a, b in state["pairs"]]
        )
        z_prod = standardize(products, state["mean2"], state["std2"])
        return add_bias(np.hstack([z, z_prod]))
    return add_bias(z)
