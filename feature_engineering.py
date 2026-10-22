"""Hypothesis-driven feature engineering for the CDC BRFSS 2015 dataset.

Implements pure-NumPy feature expansions:
1. Informative missingness indicator masks (MNAR).
2. Nominal categorical one-hot encoding (removes spurious ordinal slopes).
3. Clinical synergistic non-linear interaction terms.

All statistics (missingness rates, median imputations, categories, standardizations)
are fit strictly on training fold data to guarantee zero data leakage.
"""

import numpy as np


def extract_missing_indicators(x_tr, x_te=None, min_ratio=0.05, max_ratio=0.85):
    """Extract binary missingness indicators for columns with missing ratio in [min_ratio, max_ratio].

    In CDC BRFSS, non-response is often informative (Missing Not At Random, MNAR),
    such as survey branching skip patterns (e.g. POORHLTH is only asked when
    PHYSHLTH > 0 or MENTHLTH > 0).

    Args:
        x_tr (np.ndarray): Training features of shape (N_tr, D).
        x_te (np.ndarray, optional): Test/validation features of shape (N_te, D).
        min_ratio (float): Minimum missing ratio threshold (default: 0.05).
        max_ratio (float): Maximum missing ratio threshold (default: 0.85).

    Returns:
        tuple: (ind_tr, ind_te, sel_indices) if x_te is provided, else (ind_tr, sel_indices).
    """
    nan_mask_tr = np.isnan(x_tr)
    nan_ratios_tr = np.mean(nan_mask_tr, axis=0)

    sel_indices = np.where((nan_ratios_tr >= min_ratio) & (nan_ratios_tr <= max_ratio))[
        0
    ]

    if len(sel_indices) == 0:
        empty_tr = np.empty((x_tr.shape[0], 0), dtype=np.float32)
        if x_te is not None:
            empty_te = np.empty((x_te.shape[0], 0), dtype=np.float32)
            return empty_tr, empty_te, sel_indices
        return empty_tr, sel_indices

    ind_tr = nan_mask_tr[:, sel_indices].astype(np.float32)

    if x_te is not None:
        nan_mask_te = np.isnan(x_te)
        ind_te = nan_mask_te[:, sel_indices].astype(np.float32)
        return ind_tr, ind_te, sel_indices

    return ind_tr, sel_indices


def one_hot_encode_nominal(
    x_tr,
    x_te=None,
    feature_names=None,
    nominal_cols=("MARITAL", "EMPLOY1", "_RACE"),
    drop_first=True,
):
    """One-hot encode nominal categorical features to eliminate spurious linear orderings.

    In linear and logistic models, fitting an ordinal slope beta * x on nominal
    variables (e.g. EMPLOY1: 1=employed, 7=retired, 8=disabled) forces an unnatural
    monotonic constraint. One-hot encoding assigns independent weights to each level.

    Args:
        x_tr (np.ndarray): Imputed training features of shape (N_tr, D).
        x_te (np.ndarray, optional): Imputed test features of shape (N_te, D).
        feature_names (list): List of D feature names.
        nominal_cols (tuple): Names of nominal columns to encode.
        drop_first (bool): If True, drops the reference category to prevent collinearity.

    Returns:
        tuple: (x_tr_out, x_te_out, new_feature_names) if x_te is given,
               else (x_tr_out, new_feature_names).
    """
    if feature_names is None:
        raise ValueError("feature_names must be provided for nominal one-hot encoding.")

    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    valid_nominal = [c for c in nominal_cols if c in name_to_idx]
    nominal_indices = {name_to_idx[c] for c in valid_nominal}

    # Retain non-nominal columns
    kept_indices = [i for i in range(len(feature_names)) if i not in nominal_indices]
    x_tr_kept = x_tr[:, kept_indices]
    x_te_kept = x_te[:, kept_indices] if x_te is not None else None
    new_names = [feature_names[i] for i in kept_indices]

    ohe_tr_cols = []
    ohe_te_cols = []
    ohe_names = []

    for col_name in valid_nominal:
        idx = name_to_idx[col_name]
        col_tr = x_tr[:, idx]
        col_te = x_te[:, idx] if x_te is not None else None

        # Discover unique categories strictly from training fold
        categories = np.sort(np.unique(col_tr[~np.isnan(col_tr)]))
        if drop_first and len(categories) > 1:
            categories = categories[1:]

        for cat in categories:
            cat_name = f"{col_name}_is_{int(cat) if cat == int(cat) else cat}"
            ohe_names.append(cat_name)
            ohe_tr_cols.append((col_tr == cat).astype(np.float32))
            if col_te is not None:
                ohe_te_cols.append((col_te == cat).astype(np.float32))

    if len(ohe_tr_cols) > 0:
        ohe_tr_mat = np.column_stack(ohe_tr_cols)
        x_tr_out = np.hstack([x_tr_kept, ohe_tr_mat])
        new_names.extend(ohe_names)

        if x_te is not None:
            ohe_te_mat = np.column_stack(ohe_te_cols)
            x_te_out = np.hstack([x_te_kept, ohe_te_mat])
            return x_tr_out, x_te_out, new_names
        return x_tr_out, new_names

    if x_te is not None:
        return x_tr_kept, x_te_kept, new_names
    return x_tr_kept, new_names


def build_clinical_interactions(x_tr, x_te=None, feature_names=None):
    """Construct clinically grounded non-linear synergistic interaction terms.

    Cardiovascular risk factors (Framingham, AHA/ACC) multiply synergistically:
    1. AGE_x_BMI: _AGEG5YR * _BMI5 (vascular aging exacerbated by adiposity)
    2. AGE_x_HBP: _AGEG5YR * _RFHYPE5 (cumulative hypertensive shear stress)
    3. AGE_x_SMOKE: _AGEG5YR * _SMOKER3 (cumulative pack-year endothelial damage)
    4. HBP_x_CHOL: _RFHYPE5 * _RFCHOL (classic atherogenic two-hit hypothesis)
    5. INACTIVE_x_POORHLTH: (2 - _TOTINDA) * GENHLTH (sedentary lifestyle amplification)

    Args:
        x_tr (np.ndarray): Imputed training features.
        x_te (np.ndarray, optional): Imputed test features.
        feature_names (list): List of column names matching x_tr columns.

    Returns:
        tuple: (inter_tr, inter_te, interaction_names) if x_te is given,
               else (inter_tr, interaction_names).
    """
    if feature_names is None:
        raise ValueError(
            "feature_names must be provided to build clinical interactions."
        )

    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    def get_terms(arr):
        terms = []
        names = []

        # 1. AGE_x_BMI
        if "_AGEG5YR" in name_to_idx and "_BMI5" in name_to_idx:
            t = arr[:, name_to_idx["_AGEG5YR"]] * arr[:, name_to_idx["_BMI5"]]
            terms.append(t.astype(np.float32))
            names.append("AGE_x_BMI")

        # 2. AGE_x_HBP
        if "_AGEG5YR" in name_to_idx and "_RFHYPE5" in name_to_idx:
            t = arr[:, name_to_idx["_AGEG5YR"]] * arr[:, name_to_idx["_RFHYPE5"]]
            terms.append(t.astype(np.float32))
            names.append("AGE_x_HBP")

        # 3. AGE_x_SMOKE
        if "_AGEG5YR" in name_to_idx and "_SMOKER3" in name_to_idx:
            t = arr[:, name_to_idx["_AGEG5YR"]] * arr[:, name_to_idx["_SMOKER3"]]
            terms.append(t.astype(np.float32))
            names.append("AGE_x_SMOKE")

        # 4. HBP_x_CHOL
        if "_RFHYPE5" in name_to_idx and "_RFCHOL" in name_to_idx:
            t = arr[:, name_to_idx["_RFHYPE5"]] * arr[:, name_to_idx["_RFCHOL"]]
            terms.append(t.astype(np.float32))
            names.append("HBP_x_CHOL")

        # 5. INACTIVE_x_POORHLTH: (2 - _TOTINDA) * GENHLTH
        if "_TOTINDA" in name_to_idx and "GENHLTH" in name_to_idx:
            t = (2.0 - arr[:, name_to_idx["_TOTINDA"]]) * arr[:, name_to_idx["GENHLTH"]]
            terms.append(t.astype(np.float32))
            names.append("INACTIVE_x_POORHLTH")

        if len(terms) == 0:
            return np.empty((arr.shape[0], 0), dtype=np.float32), names
        return np.column_stack(terms), names

    inter_tr, names = get_terms(x_tr)
    if x_te is not None:
        inter_te, _ = get_terms(x_te)
        return inter_tr, inter_te, names
    return inter_tr, names


def engineer_features_fold(
    x_tr,
    x_te=None,
    feature_names=None,
    add_missing_indicators=True,
    one_hot_nominal=True,
    clinical_interactions=True,
    nominal_cols=("MARITAL", "EMPLOY1", "_RACE"),
    min_nan_ratio=0.05,
    max_nan_ratio=0.85,
    drop_first_nominal=True,
    standardize_features=True,
    add_bias=True,
):
    """Complete leak-free feature engineering pipeline for a single train/validation fold.

    Imputation medians, missingness mask definitions, nominal categories, and
    standardization moments are fit exclusively on x_tr and applied to x_te.

    Args:
        x_tr (np.ndarray): Raw/cleaned training feature matrix of shape (N_tr, D).
        x_te (np.ndarray, optional): Validation/test feature matrix of shape (N_te, D).
        feature_names (list): List of D feature names.
        add_missing_indicators (bool): Whether to append missingness indicator masks.
        one_hot_nominal (bool): Whether to one-hot encode nominal variables.
        clinical_interactions (bool): Whether to append clinical interaction terms.
        nominal_cols (tuple): Nominal column names to one-hot encode.
        min_nan_ratio (float): Lower bound for missingness indicator inclusion.
        max_nan_ratio (float): Upper bound for missingness indicator inclusion.
        drop_first_nominal (bool): Whether to drop the reference category.
        standardize_features (bool): Whether to standardize columns using training moments.
        add_bias (bool): Whether to prepend a bias column of ones.

    Returns:
        tuple: (tx_tr, tx_te, out_names) if x_te is given, else (tx_tr, out_names).
    """
    if feature_names is None:
        raise ValueError("feature_names list is required for feature engineering.")

    curr_names = list(feature_names)
    x_tr_curr = x_tr.copy()
    x_te_curr = x_te.copy() if x_te is not None else None

    # 1. Missingness indicator masks (computed strictly on raw training missingness)
    m_tr, m_te, sel_nan_idx = None, None, []
    if add_missing_indicators:
        if x_te_curr is not None:
            m_tr, m_te, sel_nan_idx = extract_missing_indicators(
                x_tr_curr, x_te_curr, min_ratio=min_nan_ratio, max_ratio=max_nan_ratio
            )
        else:
            m_tr, sel_nan_idx = extract_missing_indicators(
                x_tr_curr, min_ratio=min_nan_ratio, max_ratio=max_nan_ratio
            )
        nan_indicator_names = [f"MISS_{feature_names[i]}" for i in sel_nan_idx]

    # 2. Median imputation fit exclusively on x_tr
    med = np.nanmedian(x_tr_curr, axis=0)
    med = np.nan_to_num(med, nan=0.0)

    nan_mask_tr = np.isnan(x_tr_curr)
    for j in range(x_tr_curr.shape[1]):
        if np.any(nan_mask_tr[:, j]):
            x_tr_curr[nan_mask_tr[:, j], j] = med[j]

    if x_te_curr is not None:
        nan_mask_te = np.isnan(x_te_curr)
        for j in range(x_te_curr.shape[1]):
            if np.any(nan_mask_te[:, j]):
                x_te_curr[nan_mask_te[:, j], j] = med[j]

    # 3. Clinical non-linear interactions (computed from imputed base features)
    inter_tr, inter_te, inter_names = None, None, []
    if clinical_interactions:
        if x_te_curr is not None:
            inter_tr, inter_te, inter_names = build_clinical_interactions(
                x_tr_curr, x_te_curr, curr_names
            )
        else:
            inter_tr, inter_names = build_clinical_interactions(
                x_tr_curr, feature_names=curr_names
            )

    # 4. Nominal one-hot encoding
    if one_hot_nominal:
        if x_te_curr is not None:
            x_tr_curr, x_te_curr, curr_names = one_hot_encode_nominal(
                x_tr_curr,
                x_te_curr,
                curr_names,
                nominal_cols=nominal_cols,
                drop_first=drop_first_nominal,
            )
        else:
            x_tr_curr, curr_names = one_hot_encode_nominal(
                x_tr_curr,
                feature_names=curr_names,
                nominal_cols=nominal_cols,
                drop_first=drop_first_nominal,
            )

    # 5. Assemble all feature matrices
    feature_blocks_tr = [x_tr_curr]
    feature_blocks_te = [x_te_curr] if x_te_curr is not None else None
    assembled_names = list(curr_names)

    if clinical_interactions and inter_tr is not None and inter_tr.shape[1] > 0:
        feature_blocks_tr.append(inter_tr)
        if feature_blocks_te is not None:
            feature_blocks_te.append(inter_te)
        assembled_names.extend(inter_names)

    if add_missing_indicators and m_tr is not None and m_tr.shape[1] > 0:
        feature_blocks_tr.append(m_tr)
        if feature_blocks_te is not None:
            feature_blocks_te.append(m_te)
        assembled_names.extend(nan_indicator_names)

    tx_tr = np.hstack(feature_blocks_tr)
    tx_te = np.hstack(feature_blocks_te) if feature_blocks_te is not None else None

    # 6. Standardization fit strictly on training fold
    if standardize_features:
        mean_tr = np.mean(tx_tr, axis=0)
        std_tr = np.std(tx_tr, axis=0)
        std_tr[std_tr == 0.0] = 1.0

        tx_tr = (tx_tr - mean_tr) / std_tr
        if tx_te is not None:
            tx_te = (tx_te - mean_tr) / std_tr

    # 7. Prepend intercept bias column of ones
    if add_bias:
        tx_tr = np.c_[np.ones(len(tx_tr)), tx_tr]
        if tx_te is not None:
            tx_te = np.c_[np.ones(len(tx_te)), tx_te]
        assembled_names = ["BIAS"] + assembled_names

    if x_te is not None:
        return tx_tr, tx_te, assembled_names
    return tx_tr, assembled_names
