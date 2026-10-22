"""Hypothesis-driven feature engineering for the CDC BRFSS 2015 dataset.

Implements high-signal, streamlined feature transformations:
1. 5 Clinical synergistic non-linear interaction terms (Framingham epidemiology).
2. Curated quadratic terms for non-linear risk curves (_BMI5^2, _AGEG5YR^2).
3. Legacy ablation helpers for missingness indicators (MNAR) and nominal one-hot encoding.

All statistics (median imputations, moments) are fit strictly on training fold
data to guarantee zero data leakage.
"""

import numpy as np


def build_clinical_interactions_and_polynomials(x_tr, x_te=None, feature_names=None):
    """Construct clinically grounded synergistic cross-terms and quadratic curves.

    Cardiovascular risk factors (Framingham, AHA/ACC) multiply synergistically
    and exhibit non-linear physiological risk curves:
    1. AGE_x_BMI: _AGEG5YR * _BMI5 (vascular aging exacerbated by adiposity)
    2. AGE_x_HBP: _AGEG5YR * _RFHYPE5 (cumulative hypertensive shear stress)
    3. AGE_x_SMOKE: _AGEG5YR * _SMOKER3 (cumulative pack-year endothelial damage)
    4. HBP_x_CHOL: _RFHYPE5 * _RFCHOL (classic atherogenic two-hit hypothesis)
    5. INACTIVE_x_POORHLTH: (2 - _TOTINDA) * GENHLTH (sedentary lifestyle amplification)
    6. BMI_SQ: _BMI5^2 (non-linear U-shaped mortality curve in adiposity)
    7. AGE_SQ: _AGEG5YR^2 (exponential age-related vascular stiffening)

    Args:
        x_tr (np.ndarray): Imputed training features.
        x_te (np.ndarray, optional): Imputed test features.
        feature_names (list): List of column names matching x_tr columns.

    Returns:
        tuple: (terms_tr, terms_te, term_names) if x_te is given,
               else (terms_tr, term_names).
    """
    if feature_names is None:
        raise ValueError("feature_names must be provided to build interactions.")

    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    def extract_terms(arr):
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

        # 6. BMI_SQ: _BMI5^2
        if "_BMI5" in name_to_idx:
            t = arr[:, name_to_idx["_BMI5"]] ** 2
            terms.append(t.astype(np.float32))
            names.append("BMI_SQ")

        # 7. AGE_SQ: _AGEG5YR^2
        if "_AGEG5YR" in name_to_idx:
            t = arr[:, name_to_idx["_AGEG5YR"]] ** 2
            terms.append(t.astype(np.float32))
            names.append("AGE_SQ")

        if len(terms) == 0:
            return np.empty((arr.shape[0], 0), dtype=np.float32), names
        return np.column_stack(terms), names

    terms_tr, names = extract_terms(x_tr)
    if x_te is not None:
        terms_te, _ = extract_terms(x_te)
        return terms_tr, terms_te, names
    return terms_tr, names


def build_clinical_interactions(x_tr, x_te=None, feature_names=None):
    """Legacy backward-compatible wrapper returning only the 5 clinical cross-terms."""
    if feature_names is None:
        raise ValueError("feature_names must be provided.")
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    def extract_5_terms(arr):
        terms = []
        names = []
        if "_AGEG5YR" in name_to_idx and "_BMI5" in name_to_idx:
            terms.append(
                (arr[:, name_to_idx["_AGEG5YR"]] * arr[:, name_to_idx["_BMI5"]]).astype(
                    np.float32
                )
            )
            names.append("AGE_x_BMI")
        if "_AGEG5YR" in name_to_idx and "_RFHYPE5" in name_to_idx:
            terms.append(
                (
                    arr[:, name_to_idx["_AGEG5YR"]] * arr[:, name_to_idx["_RFHYPE5"]]
                ).astype(np.float32)
            )
            names.append("AGE_x_HBP")
        if "_AGEG5YR" in name_to_idx and "_SMOKER3" in name_to_idx:
            terms.append(
                (
                    arr[:, name_to_idx["_AGEG5YR"]] * arr[:, name_to_idx["_SMOKER3"]]
                ).astype(np.float32)
            )
            names.append("AGE_x_SMOKE")
        if "_RFHYPE5" in name_to_idx and "_RFCHOL" in name_to_idx:
            terms.append(
                (
                    arr[:, name_to_idx["_RFHYPE5"]] * arr[:, name_to_idx["_RFCHOL"]]
                ).astype(np.float32)
            )
            names.append("HBP_x_CHOL")
        if "_TOTINDA" in name_to_idx and "GENHLTH" in name_to_idx:
            terms.append(
                (
                    (2.0 - arr[:, name_to_idx["_TOTINDA"]])
                    * arr[:, name_to_idx["GENHLTH"]]
                ).astype(np.float32)
            )
            names.append("INACTIVE_x_POORHLTH")
        return np.column_stack(terms), names

    t_tr, names = extract_5_terms(x_tr)
    if x_te is not None:
        t_te, _ = extract_5_terms(x_te)
        return t_tr, t_te, names
    return t_tr, names


def extract_missing_indicators(x_tr, x_te=None, min_ratio=0.05, max_ratio=0.85):
    """Extract binary missingness indicators for columns with missing ratio in [min_ratio, max_ratio]."""
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
    """One-hot encode nominal categorical features to eliminate spurious linear orderings."""
    if feature_names is None:
        raise ValueError("feature_names must be provided for nominal one-hot encoding.")

    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    valid_nominal = [c for c in nominal_cols if c in name_to_idx]
    nominal_indices = {name_to_idx[c] for c in valid_nominal}

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


def build_streamlined_features(
    x_tr,
    x_te=None,
    feature_names=None,
    add_bias=True,
    standardize_features=True,
):
    """Construct an elegant, high-signal 205-dimensional design matrix.

    Pipeline:
    - 197 clean base features.
    - Median imputation fit strictly on training fold.
    - 5 verified clinical Framingham cross-terms.
    - 2 quadratic risk curves (_BMI5^2, _AGEG5YR^2).
    - Standardization fit on training fold moments.
    - Intercept bias column of ones.

    Total features: 197 base + 5 cross-terms + 2 quadratic terms = 204 features (+ 1 bias = 205 dims).

    Args:
        x_tr (np.ndarray): Training features (N_tr, D).
        x_te (np.ndarray, optional): Test/validation features (N_te, D).
        feature_names (list): List of feature names.
        add_bias (bool): If True, prepends intercept ones column.
        standardize_features (bool): If True, standardizes features using training moments.

    Returns:
        tuple: (tx_tr, tx_te, out_names) if x_te is given, else (tx_tr, out_names).
    """
    if feature_names is None:
        raise ValueError("feature_names list is required.")

    x_tr_imp = x_tr.copy()
    x_te_imp = x_te.copy() if x_te is not None else None

    # 1. Median imputation fit strictly on x_tr
    med = np.nanmedian(x_tr_imp, axis=0)
    med = np.nan_to_num(med, nan=0.0)

    nan_mask_tr = np.isnan(x_tr_imp)
    for j in range(x_tr_imp.shape[1]):
        if np.any(nan_mask_tr[:, j]):
            x_tr_imp[nan_mask_tr[:, j], j] = med[j]

    if x_te_imp is not None:
        nan_mask_te = np.isnan(x_te_imp)
        for j in range(x_te_imp.shape[1]):
            if np.any(nan_mask_te[:, j]):
                x_te_imp[nan_mask_te[:, j], j] = med[j]

    # 2. Clinical non-linear cross-terms and quadratic polynomials
    if x_te_imp is not None:
        terms_tr, terms_te, term_names = build_clinical_interactions_and_polynomials(
            x_tr_imp, x_te_imp, feature_names
        )
    else:
        terms_tr, term_names = build_clinical_interactions_and_polynomials(
            x_tr_imp, feature_names=feature_names
        )
        terms_te = None

    assembled_tr = np.hstack([x_tr_imp, terms_tr])
    assembled_te = np.hstack([x_te_imp, terms_te]) if x_te_imp is not None else None
    assembled_names = list(feature_names) + list(term_names)

    # 3. Standardization fit strictly on training fold
    if standardize_features:
        mean_tr = np.mean(assembled_tr, axis=0)
        std_tr = np.std(assembled_tr, axis=0)
        std_tr[std_tr == 0.0] = 1.0

        tx_tr = (assembled_tr - mean_tr) / std_tr
        tx_te = (assembled_te - mean_tr) / std_tr if assembled_te is not None else None
    else:
        tx_tr = assembled_tr
        tx_te = assembled_te

    # 4. Prepend intercept bias column of ones
    if add_bias:
        tx_tr = np.c_[np.ones(len(tx_tr)), tx_tr]
        if tx_te is not None:
            tx_te = np.c_[np.ones(len(tx_te)), tx_te]
        assembled_names = ["BIAS"] + assembled_names

    if x_te is not None:
        return tx_tr, tx_te, assembled_names
    return tx_tr, assembled_names


def engineer_features_fold(
    x_tr,
    x_te=None,
    feature_names=None,
    add_missing_indicators=False,
    one_hot_nominal=False,
    clinical_interactions=True,
    include_polynomials=True,
    nominal_cols=("MARITAL", "EMPLOY1", "_RACE"),
    min_nan_ratio=0.05,
    max_nan_ratio=0.85,
    drop_first_nominal=True,
    standardize_features=True,
    add_bias=True,
):
    """Configurable feature engineering pipeline supporting both streamlined and ablation modes."""
    if not add_missing_indicators and not one_hot_nominal and clinical_interactions:
        # Use streamlined fast path (205 dimensions)
        return build_streamlined_features(
            x_tr=x_tr,
            x_te=x_te,
            feature_names=feature_names,
            add_bias=add_bias,
            standardize_features=standardize_features,
        )

    # Ablation mode (when missing indicators or one-hot are explicitly requested)
    curr_names = list(feature_names)
    x_tr_curr = x_tr.copy()
    x_te_curr = x_te.copy() if x_te is not None else None

    # Missingness indicator masks
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

    # Median imputation fit exclusively on x_tr
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

    # Clinical interactions and polynomials
    inter_tr, inter_te, inter_names = None, None, []
    if clinical_interactions:
        if include_polynomials:
            if x_te_curr is not None:
                inter_tr, inter_te, inter_names = (
                    build_clinical_interactions_and_polynomials(
                        x_tr_curr, x_te_curr, curr_names
                    )
                )
            else:
                inter_tr, inter_names = build_clinical_interactions_and_polynomials(
                    x_tr_curr, feature_names=curr_names
                )
        else:
            if x_te_curr is not None:
                inter_tr, inter_te, inter_names = build_clinical_interactions(
                    x_tr_curr, x_te_curr, curr_names
                )
            else:
                inter_tr, inter_names = build_clinical_interactions(
                    x_tr_curr, feature_names=curr_names
                )

    # Nominal one-hot encoding
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

    if standardize_features:
        mean_tr = np.mean(tx_tr, axis=0)
        std_tr = np.std(tx_tr, axis=0)
        std_tr[std_tr == 0.0] = 1.0

        tx_tr = (tx_tr - mean_tr) / std_tr
        if tx_te is not None:
            tx_te = (tx_te - mean_tr) / std_tr

    if add_bias:
        tx_tr = np.c_[np.ones(len(tx_tr)), tx_tr]
        if tx_te is not None:
            tx_te = np.c_[np.ones(len(tx_te)), tx_te]
        assembled_names = ["BIAS"] + assembled_names

    if x_te is not None:
        return tx_tr, tx_te, assembled_names
    return tx_tr, assembled_names
