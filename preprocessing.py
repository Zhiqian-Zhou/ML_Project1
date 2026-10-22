"""Data cleaning and feature engineering for the BRFSS 2015 dataset.

Every statistic (medians, means, stds, kept columns, categories) is computed
strictly on the training data and then applied to test/validation data.
"""

import os
import numpy as np

# Administrative and identifier columns with zero predictive value
ADMIN_COLUMNS = {
    "Id",
    "_STATE",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
}

# Survey sample weighting columns designed for population-level census adjustment
WEIGHT_COLUMNS = {
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_CLLCPWT",
    "_LLCPWT",
}

# Raw features with unit ambiguities replaced by standardized calculated metrics
RAW_CONFLICT_COLUMNS = {
    "HEIGHT3",
    "WEIGHT2",
}

# Explicit skip-pattern target leaks to preserve despite extreme sparsity
PRESERVED_SPARSE_FEATURES = {
    "HAREHAB1",
}

# CDC survey day-count columns: 88 represents 0 days (healthy), 77/99 are unknown
DAY_COUNT_COLUMNS = {
    "PHYSHLTH",
    "MENTHLTH",
    "POORHLTH",
    "DRNK3GE5",
    "CHILDREN",
}

# CDC survey frequency columns: 888 represents 0 drinks/sessions, 777/999 are unknown
FREQUENCY_COLUMNS = {
    "ALCDAY5",
    "STRENGTH",
}

# CDC exercise frequency columns: 777/999 are unknown
EXERCISE_FREQ_COLUMNS = {
    "EXEROFT1",
    "EXEROFT2",
}

# Categorical bracket columns where 77/99 represent unknown/refused
BRACKET_COLUMNS = {
    "INCOME2",
    "JOINPAIN",
}

# Categorical columns where 9 represents refused (preserving valid categories 1-8)
REFUSED_9_COLUMNS = {
    "MARITAL",
    "EDUCA",
    "EMPLOY1",
}

# Age bracket column: 14 represents unknown/refused
AGE_BRACKET_COLUMN = "_AGEG5YR"

# Standard binary or ordinal survey questions where 7=Don't know, 9=Refused
STANDARD_SURVEY_COLUMNS = {
    "GENHLTH",
    "HLTHPLN1",
    "PERSDOC2",
    "MEDCOST",
    "CHECKUP1",
    "BPHIGH4",
    "BPMEDS",
    "BLOODCHO",
    "CHOLCHK",
    "TOLDHI2",
    "CVDSTRK3",
    "ASTHMA3",
    "CHCSCNCR",
    "CHCOCNCR",
    "CHCCOPD1",
    "HAVARTH3",
    "ADDEPEV2",
    "CHCKIDNY",
    "DIABETE3",
    "NUMHHOL2",
    "CPDEMO1",
    "VETERAN3",
    "INTERNET",
    "QLACTLM2",
    "USEEQUIP",
    "BLIND",
    "DECIDE",
    "DIFFWALK",
    "DIFFDRES",
    "DIFFALON",
    "SMOKE100",
    "SMOKDAY2",
    "EXERANY2",
    "LMTJOIN3",
    "ARTHDIS2",
    "SEATBELT",
    "FLUSHOT6",
    "PNEUVAC3",
    "HIVTST6",
    "PDIABTST",
    "PREDIAB1",
    "CAREGIV1",
    "CRGVEXPT",
    "CIMEMLOS",
    "SCNTMNY1",
    "SCNTMEL1",
    "SXORIENT",
    "TRNSGNDR",
    "LANDLINE",
    "CADULT",
    "PVTRESD2",
    "CSTATE",
}

# CDC calculated indicators prefixed with '_' where 9 represents missing/unknown
CALCULATED_FLAG_COLUMNS = {
    "_CHISPNC",
    "_DUALUSE",
    "_RFHLTH",
    "_HCVU651",
    "_RFHYPE5",
    "_CHOLCHK",
    "_RFCHOL",
    "_LTASTH1",
    "_CASTHM1",
    "_ASTHMS1",
    "_PRACE1",
    "_MRACE1",
    "_HISPANC",
    "_RACE",
    "_RACEG21",
    "_RACEGR3",
    "_RACE_G1",
    "_AGE65YR",
    "_BMI5CAT",
    "_RFBMI5",
    "_CHLDCNT",
    "_EDUCAG",
    "_INCOMG",
    "_RFSMOK3",
    "_RFBING5",
    "_RFDRHV5",
    "_FRTLT1",
    "_VEGLT1",
    "_TOTINDA",
    "_PACAT1",
    "_PAINDX1",
    "_PA150R2",
    "_PA300R2",
    "_PA30021",
    "_PASTRNG",
    "_PAREC1",
    "_PASTAE1",
    "_LMTACT1",
    "_LMTWRK1",
    "_LMTSCL1",
    "_RFSEAT2",
    "_RFSEAT3",
    "_FLSHOT6",
    "_PNEUMO2",
    "_AIDTST3",
}


def load_feature_names(data_path):
    """Read the column names of x_train.csv (helpers.load_csv_data drops them).

    Args:
        data_path (str): Folder containing x_train.csv.

    Returns:
        list: List of 321 feature names (excluding the Id column).
    """
    train_path = os.path.join(data_path, "x_train.csv")
    with open(train_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    return header[1:]


def load_data_cached(data_path, sub_sample=False):
    """Fast loader for BRFSS data with binary caching.

    Args:
        data_path (str): Directory containing the CSV files.
        sub_sample (bool): If True, subsamples 1 out of 50 training rows.

    Returns:
        tuple: (x_train, x_test, y_train, train_ids, test_ids)
    """
    cache_dir = os.path.join(data_path, "..", "cache_raw")
    os.makedirs(cache_dir, exist_ok=True)

    cache_xtr = os.path.join(cache_dir, "x_train_raw.npy")
    cache_xte = os.path.join(cache_dir, "x_test_raw.npy")
    cache_ytr = os.path.join(cache_dir, "y_train_raw.npy")
    cache_tr_id = os.path.join(cache_dir, "train_ids_raw.npy")
    cache_te_id = os.path.join(cache_dir, "test_ids_raw.npy")

    if (
        os.path.exists(cache_xtr)
        and os.path.exists(cache_xte)
        and os.path.exists(cache_ytr)
    ):
        x_train = np.load(cache_xtr)
        x_test = np.load(cache_xte)
        y_train = np.load(cache_ytr)
        train_ids = np.load(cache_tr_id)
        test_ids = np.load(cache_te_id)
    else:
        # Fast streaming parse
        def parse_file(path):
            with open(path, "r", encoding="utf-8") as f:
                f.readline()  # header
                ids_list = []
                data_list = []
                for line in f:
                    toks = line.strip().split(",")
                    ids_list.append(int(toks[0]))
                    data_list.append(
                        [float(tok) if tok != "" else np.nan for tok in toks[1:]]
                    )
            return np.array(ids_list, dtype=np.int32), np.array(
                data_list, dtype=np.float32
            )

        train_ids, x_train = parse_file(os.path.join(data_path, "x_train.csv"))
        test_ids, x_test = parse_file(os.path.join(data_path, "x_test.csv"))

        # Parse y_train
        y_ids = []
        y_labels = []
        with open(os.path.join(data_path, "y_train.csv"), "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                toks = line.strip().split(",")
                y_ids.append(int(toks[0]))
                y_labels.append(int(toks[1]))
        y_train = np.array(y_labels, dtype=np.int8)

        np.save(cache_xtr, x_train)
        np.save(cache_xte, x_test)
        np.save(cache_ytr, y_train)
        np.save(cache_tr_id, train_ids)
        np.save(cache_te_id, test_ids)

    if sub_sample:
        x_train = x_train[::50]
        y_train = y_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def replace_special_codes(x, feature_names):
    """Replace codebook 'don't know / refused' codes by NaN, and 'none' (88/888) by 0.

    Args:
        x (np.ndarray): Array of shape (N, D).
        feature_names (list): List of D column names.

    Returns:
        np.ndarray: Cleaned copy of x.
    """
    cleaned = x.copy()
    col_map = {name: idx for idx, name in enumerate(feature_names)}

    for col in DAY_COUNT_COLUMNS:
        if col in col_map:
            idx = col_map[col]
            v = cleaned[:, idx]
            v[v == 88.0] = 0.0
            v[v == 77.0] = np.nan
            v[v == 99.0] = np.nan

    for col in FREQUENCY_COLUMNS:
        if col in col_map:
            idx = col_map[col]
            v = cleaned[:, idx]
            v[v == 888.0] = 0.0
            v[v == 777.0] = np.nan
            v[v == 999.0] = np.nan

    for col in EXERCISE_FREQ_COLUMNS:
        if col in col_map:
            idx = col_map[col]
            v = cleaned[:, idx]
            v[v == 777.0] = np.nan
            v[v == 999.0] = np.nan

    for col in BRACKET_COLUMNS:
        if col in col_map:
            idx = col_map[col]
            v = cleaned[:, idx]
            v[v == 77.0] = np.nan
            v[v == 99.0] = np.nan

    if AGE_BRACKET_COLUMN in col_map:
        idx = col_map[AGE_BRACKET_COLUMN]
        v = cleaned[:, idx]
        v[v == 14.0] = np.nan

    for col in REFUSED_9_COLUMNS:
        if col in col_map:
            idx = col_map[col]
            v = cleaned[:, idx]
            v[v == 9.0] = np.nan

    for col in STANDARD_SURVEY_COLUMNS:
        if col in col_map:
            idx = col_map[col]
            v = cleaned[:, idx]
            v[v == 7.0] = np.nan
            v[v == 9.0] = np.nan

    for col in CALCULATED_FLAG_COLUMNS:
        if col in col_map:
            idx = col_map[col]
            v = cleaned[:, idx]
            v[v == 9.0] = np.nan

    return cleaned


def select_columns(x_train, feature_names, max_nan_ratio=0.85):
    """Decide which columns to keep using the training data only.

    Args:
        x_train (np.ndarray): Training features of shape (N, D).
        feature_names (list): List of D feature names.
        max_nan_ratio (float): Maximum allowed missing ratio.

    Returns:
        np.ndarray: Boolean mask of shape (D,), True for retained columns.
    """
    static_drop = ADMIN_COLUMNS | WEIGHT_COLUMNS | RAW_CONFLICT_COLUMNS
    n_samples = x_train.shape[0]
    mask = np.zeros(len(feature_names), dtype=bool)

    for i, name in enumerate(feature_names):
        if name in static_drop:
            continue
        missing_ratio = np.isnan(x_train[:, i]).sum() / n_samples
        if missing_ratio > max_nan_ratio and name not in PRESERVED_SPARSE_FEATURES:
            continue
        mask[i] = True

    return mask


def fit_imputer(x_train):
    """Compute per-column median fill values on the training data.

    Args:
        x_train (np.ndarray): Feature array with NaNs.

    Returns:
        np.ndarray: Array of shape (D,) with median fill values.
    """
    med = np.nanmedian(x_train, axis=0)
    med = np.nan_to_num(med, nan=0.0)
    return med


def apply_imputer(x, fill_values, add_indicators=True):
    """Fill NaNs with fill_values, optionally appending missing indicators.

    Args:
        x (np.ndarray): Feature array of shape (N, D).
        fill_values (np.ndarray): Array of shape (D,).
        add_indicators (bool): If True, appends 0/1 missing indicators.

    Returns:
        np.ndarray: Imputed feature array of shape (N, D').
    """
    x_imp = x.copy()
    nan_mask = np.isnan(x_imp)

    for col in range(x.shape[1]):
        if np.any(nan_mask[:, col]):
            x_imp[nan_mask[:, col], col] = fill_values[col]

    if add_indicators:
        # Append indicators for columns that had any NaNs
        has_nan_cols = np.where(np.any(nan_mask, axis=0))[0]
        if len(has_nan_cols) > 0:
            indicators = nan_mask[:, has_nan_cols].astype(np.float32)
            x_imp = np.hstack([x_imp, indicators])

    return x_imp


def fit_standardizer(x_train):
    """Compute per-column mean and std on training data.

    Args:
        x_train (np.ndarray): Imputed feature array of shape (N, D).

    Returns:
        tuple: (mean, std) arrays of shape (D,).
    """
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std[std == 0.0] = 1.0
    return mean, std


def standardize(x, mean, std):
    """Standardize feature matrix using precomputed moments.

    Args:
        x (np.ndarray): Feature array of shape (N, D).
        mean (np.ndarray): Mean vector.
        std (np.ndarray): Standard deviation vector.

    Returns:
        np.ndarray: Standardized array.
    """
    return (x - mean) / std


def build_poly(x, degree):
    """Polynomial expansion of each feature: x, x^2, ..., x^d.

    Args:
        x (np.ndarray): Array of shape (N, D).
        degree (int): Expansion degree >= 1.

    Returns:
        np.ndarray: Array of shape (N, D * degree).
    """
    if degree <= 1:
        return x
    poly_terms = [x**d for d in range(1, degree + 1)]
    return np.hstack(poly_terms)


def add_bias(x):
    """Prepend a column of ones for intercept offset w_0.

    Args:
        x (np.ndarray): Array of shape (N, D).

    Returns:
        np.ndarray: Array of shape (N, D + 1).
    """
    return np.c_[np.ones(len(x)), x]


def preprocess(x_train, x_test, feature_names, config):
    """Full end-to-end preprocessing pipeline.

    Fits all operations strictly on x_train and transforms both x_train and x_test.

    Args:
        x_train (np.ndarray): Raw or selected training features.
        x_test (np.ndarray): Raw or selected test features.
        feature_names (list): List of feature names.
        config (dict): Pipeline configuration dictionary.

    Returns:
        tuple: (tx_train, tx_test)
    """
    from feature_engineering import engineer_features_fold

    max_nan_ratio = config.get("max_nan_ratio", 0.85)
    add_nan_indicators = config.get("add_nan_indicators", True)
    one_hot_nominal = config.get("one_hot_nominal", True)
    clinical_interactions = config.get("clinical_interactions", True)
    drop_first_nominal = config.get("drop_first_nominal", True)
    poly_degree = config.get("poly_degree", 1)

    # 1. Clean CDC special survey codes if raw uncleaned data
    if x_train.shape[1] == len(feature_names) and len(feature_names) > 250:
        x_tr_clean = replace_special_codes(x_train, feature_names)
        x_te_clean = replace_special_codes(x_test, feature_names)

        # 2. Add companion skip-pattern indicator for HAREHAB1 before pruning
        if "HAREHAB1" in feature_names:
            h_idx = feature_names.index("HAREHAB1")
            h_ind_tr = (~np.isnan(x_tr_clean[:, h_idx])).astype(np.float32)[
                :, np.newaxis
            ]
            h_ind_te = (~np.isnan(x_te_clean[:, h_idx])).astype(np.float32)[
                :, np.newaxis
            ]

        # 3. Select columns based on training set statistics
        keep_mask = select_columns(
            x_tr_clean, feature_names, max_nan_ratio=max_nan_ratio
        )
        x_tr_sel = x_tr_clean[:, keep_mask]
        x_te_sel = x_te_clean[:, keep_mask]
        active_names = [
            feature_names[i] for i in range(len(feature_names)) if keep_mask[i]
        ]

        if "HAREHAB1" in feature_names:
            x_tr_sel = np.hstack([x_tr_sel, h_ind_tr])
            x_te_sel = np.hstack([x_te_sel, h_ind_te])
            active_names.append("HAREHAB1_ANSWERED")
    else:
        x_tr_sel = x_train
        x_te_sel = x_test
        active_names = list(feature_names)

    # 4. Feature engineering (missing indicators, nominal one-hot, clinical interactions)
    tx_train, tx_test, out_names = engineer_features_fold(
        x_tr=x_tr_sel,
        x_te=x_te_sel,
        feature_names=active_names,
        add_missing_indicators=add_nan_indicators,
        one_hot_nominal=one_hot_nominal,
        clinical_interactions=clinical_interactions,
        min_nan_ratio=0.05,
        max_nan_ratio=max_nan_ratio,
        drop_first_nominal=drop_first_nominal,
        standardize_features=True,
        add_bias=False,
    )

    # 5. Polynomial expansion if degree > 1
    if poly_degree > 1:
        tx_train = build_poly(tx_train, poly_degree)
        tx_test = build_poly(tx_test, poly_degree)

    # 6. Add intercept bias
    tx_train = add_bias(tx_train)
    tx_test = add_bias(tx_test)

    return tx_train, tx_test
