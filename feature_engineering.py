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


def compute_woe_features(
    x_tr,
    x_te=None,
    feature_names=None,
    y_tr=None,
    alpha=20.0,
    woe_cols=(
        "_AGEG5YR",
        "GENHLTH",
        "_SMOKER3",
        "INCOME2",
        "EMPLOY1",
        "_EDUCAG",
        "DIFFWALK",
        "DIABETE3",
    ),
):
    """Compute empirical Bayesian Laplace-smoothed Log-Odds (WoE) on training set and apply out-of-fold.

    WoE formula:
        WoE(c) = ln((N_pos(c) + alpha * P(y=1)) / (N_neg(c) + alpha * P(y=-1))) - ln(P(y=1) / P(y=-1))
    """
    if feature_names is None:
        raise ValueError("feature_names is required to identify WoE columns.")
    if y_tr is None:
        raise ValueError(
            "y_tr is required to compute empirical Bayesian WoE encodings."
        )

    pos_mask = y_tr == 1
    neg_mask = (y_tr == -1) if np.any(y_tr == -1) else (y_tr == 0)
    pos_p = np.mean(pos_mask)
    neg_p = 1.0 - pos_p
    prior_lo = np.log(pos_p / (neg_p + 1e-12))

    woe_tr_list, woe_te_list = [], []
    active_woe_names = []
    for col_name in woe_cols:
        if col_name not in feature_names:
            continue
        idx = feature_names.index(col_name)
        col_tr = x_tr[:, idx]
        col_te = x_te[:, idx] if x_te is not None else None

        cats = np.unique(col_tr[~np.isnan(col_tr)])
        mapping = {}
        for c in cats:
            m_c = col_tr == c
            n_pos = np.sum(pos_mask[m_c])
            n_neg = np.sum(neg_mask[m_c])
            mapping[c] = (
                np.log((n_pos + alpha * pos_p) / (n_neg + alpha * neg_p + 1e-12))
                - prior_lo
            )

        w_col_tr = np.array(
            [0.0 if np.isnan(v) else mapping.get(v, 0.0) for v in col_tr],
            dtype=np.float32,
        )
        woe_tr_list.append(w_col_tr)
        if col_te is not None:
            w_col_te = np.array(
                [0.0 if np.isnan(v) else mapping.get(v, 0.0) for v in col_te],
                dtype=np.float32,
            )
            woe_te_list.append(w_col_te)
        active_woe_names.append(f"WOE_{col_name}")

    w_mat_tr = np.column_stack(woe_tr_list)
    w_mat_te = np.column_stack(woe_te_list) if x_te is not None else None

    # Standardize WoE columns using training statistics
    w_mean = np.mean(w_mat_tr, axis=0)
    w_std = np.std(w_mat_tr, axis=0)
    w_std[w_std == 0.0] = 1.0

    w_mat_tr = (w_mat_tr - w_mean) / w_std
    if w_mat_te is not None:
        w_mat_te = (w_mat_te - w_mean) / w_std

    if x_te is not None:
        return w_mat_tr, w_mat_te, active_woe_names
    return w_mat_tr, active_woe_names


def build_clinical_subgroup_terms(x_tr_imp, x_te_imp=None, feature_names=None):
    """Construct 10 clinical subgroup bias-correcting and risk-linearizing terms.

    1. RESILIENT_ELDERLY: (Age >= 10.0) * (GENHLTH <= 2.0)
    2. FRAIL_ELDERLY: (Age >= 10.0) * (GENHLTH >= 4.0)
    3. YOUNG_HIGH_RISK: (Age <= 6.0) * (Stroke == 1.0 | (Smoker in [1,2] & HBP == 2.0))
    4. AGE_x_GENHLTH: Age * GENHLTH
    5. LOG_PHYSHLTH: np.log1p(PHYSHLTH)
    6. LOG_MENTHLTH: np.log1p(MENTHLTH)
    7. MALE_x_AGE: (SEX == 1.0) * Age
    8. DIFF_x_AGE: (DIFFWALK == 1.0) * Age
    9. STROKE_x_AGE: (CVDSTRK3 == 1.0) * Age
    10. METABOLIC_TRIO: (DIABETE3 == 1.0) * (_RFHYPE5 == 2.0) * (_RFCHOL == 2.0)
    """
    if feature_names is None:
        raise ValueError("feature_names is required.")

    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    def extract_subgroup_matrix(arr):
        age = arr[:, name_to_idx["_AGEG5YR"]]
        genhlth = arr[:, name_to_idx["GENHLTH"]]
        stroke = arr[:, name_to_idx["CVDSTRK3"]]
        smoker = arr[:, name_to_idx["_SMOKER3"]]
        hbp = arr[:, name_to_idx["_RFHYPE5"]]
        chol = arr[:, name_to_idx["_RFCHOL"]]
        diabete = arr[:, name_to_idx["DIABETE3"]]
        physhlth = arr[:, name_to_idx["PHYSHLTH"]]
        menthlth = arr[:, name_to_idx["MENTHLTH"]]
        sex = arr[:, name_to_idx["SEX"]]
        diffwalk = arr[:, name_to_idx["DIFFWALK"]]

        # 1. RESILIENT_ELDERLY
        resilient_elderly = ((age >= 10.0) & (genhlth <= 2.0)).astype(np.float32)
        # 2. FRAIL_ELDERLY
        frail_elderly = ((age >= 10.0) & (genhlth >= 4.0)).astype(np.float32)
        # 3. YOUNG_HIGH_RISK
        stroke_yes = stroke == 1.0
        smoke_curr = (smoker == 1.0) | (smoker == 2.0)
        hbp_yes = hbp == 2.0
        young_high_risk = ((age <= 6.0) & (stroke_yes | (smoke_curr & hbp_yes))).astype(
            np.float32
        )
        # 4. AGE_x_GENHLTH
        age_x_genhlth = (age * genhlth).astype(np.float32)
        # 5. LOG_PHYSHLTH
        log_physhlth = np.log1p(np.maximum(0.0, physhlth)).astype(np.float32)
        # 6. LOG_MENTHLTH
        log_menthlth = np.log1p(np.maximum(0.0, menthlth)).astype(np.float32)
        # 7. MALE_x_AGE
        male_x_age = ((sex == 1.0).astype(np.float32) * age).astype(np.float32)
        # 8. DIFF_x_AGE
        diff_x_age = ((diffwalk == 1.0).astype(np.float32) * age).astype(np.float32)
        # 9. STROKE_x_AGE
        stroke_x_age = ((stroke == 1.0).astype(np.float32) * age).astype(np.float32)
        # 10. METABOLIC_TRIO
        metabolic_trio = ((diabete == 1.0) & (hbp == 2.0) & (chol == 2.0)).astype(
            np.float32
        )

        return np.column_stack(
            [
                resilient_elderly,
                frail_elderly,
                young_high_risk,
                age_x_genhlth,
                log_physhlth,
                log_menthlth,
                male_x_age,
                diff_x_age,
                stroke_x_age,
                metabolic_trio,
            ]
        )

    sub_tr = extract_subgroup_matrix(x_tr_imp)
    sub_te = extract_subgroup_matrix(x_te_imp) if x_te_imp is not None else None

    sub_names = [
        "RESILIENT_ELDERLY",
        "FRAIL_ELDERLY",
        "YOUNG_HIGH_RISK",
        "AGE_x_GENHLTH",
        "LOG_PHYSHLTH",
        "LOG_MENTHLTH",
        "MALE_x_AGE",
        "DIFF_x_AGE",
        "STROKE_x_AGE",
        "METABOLIC_TRIO",
    ]

    # Standardize subgroup terms using training statistics
    sub_mean = np.mean(sub_tr, axis=0)
    sub_std = np.std(sub_tr, axis=0)
    sub_std[sub_std == 0.0] = 1.0

    sub_tr = (sub_tr - sub_mean) / sub_std
    if sub_te is not None:
        sub_te = (sub_te - sub_mean) / sub_std

    if x_te_imp is not None:
        return sub_tr, sub_te, sub_names
    return sub_tr, sub_names


def build_vascular_synergy_terms(x_tr_imp, x_te_imp=None, feature_names=None):
    """Construct 8 high-mortality vascular conjunctions and 2 low-risk demographic anchors.

    High-Mortality Vascular Conjunctions (pos rate 39% - 53%, 4.4x - 6.0x lift):
    1. STROKE_x_COPD: CVDSTRK3 == 1.0 & CHCCOPD1 == 1.0 (52.9% pos rate, 6.0x lift)
    2. DIABETE_x_STROKE: DIABETE3 == 1.0 & CVDSTRK3 == 1.0 (48.9% pos rate, 5.5x lift)
    3. STROKE_x_POORHLTH: CVDSTRK3 == 1.0 & GENHLTH >= 4.0 (47.1% pos rate, 5.3x lift)
    4. COPD_x_KIDNEY: CHCCOPD1 == 1.0 & CHCKIDNY == 1.0 (46.0% pos rate, 5.2x lift)
    5. STROKE_x_DIFFWALK: CVDSTRK3 == 1.0 & DIFFWALK == 1.0 (45.4% pos rate, 5.1x lift)
    6. DIABETE_x_COPD: DIABETE3 == 1.0 & CHCCOPD1 == 1.0 (40.1% pos rate, 4.5x lift)
    7. DIABETE_x_KIDNEY: DIABETE3 == 1.0 & CHCKIDNY == 1.0 (39.5% pos rate, 4.5x lift)
    8. POORHLTH_x_KIDNEY: GENHLTH >= 4.0 & CHCKIDNY == 1.0 (39.0% pos rate, 4.4x lift)

    Low-Risk Demographic Anchors (pos rate < 2.0%, 0.18x - 0.22x lift):
    9. YOUNG_FEMALE: (_AGEG5YR <= 6.0) & (SEX == 2.0) (1.61% pos rate, 0.18x lift)
    10. YOUNG_MALE: (_AGEG5YR <= 6.0) & (SEX == 1.0) (1.97% pos rate, 0.22x lift)
    """
    if feature_names is None:
        raise ValueError("feature_names is required.")

    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    def extract_vascular_matrix(arr):
        stroke = arr[:, name_to_idx["CVDSTRK3"]] == 1.0
        copd = arr[:, name_to_idx["CHCCOPD1"]] == 1.0
        diab = arr[:, name_to_idx["DIABETE3"]] == 1.0
        kidney = arr[:, name_to_idx["CHCKIDNY"]] == 1.0
        diff = arr[:, name_to_idx["DIFFWALK"]] == 1.0
        poorhlth = arr[:, name_to_idx["GENHLTH"]] >= 4.0
        age = arr[:, name_to_idx["_AGEG5YR"]]
        sex = arr[:, name_to_idx["SEX"]]

        # 8 High-Mortality Vascular Conjunctions
        stroke_x_copd = (stroke & copd).astype(np.float32)
        diabete_x_stroke = (diab & stroke).astype(np.float32)
        stroke_x_poorhlth = (stroke & poorhlth).astype(np.float32)
        copd_x_kidney = (copd & kidney).astype(np.float32)
        stroke_x_diffwalk = (stroke & diff).astype(np.float32)
        diabete_x_copd = (diab & copd).astype(np.float32)
        diabete_x_kidney = (diab & kidney).astype(np.float32)
        poorhlth_x_kidney = (poorhlth & kidney).astype(np.float32)

        # 2 Low-Risk Demographic Anchors
        young_female = ((age <= 6.0) & (sex == 2.0)).astype(np.float32)
        young_male = ((age <= 6.0) & (sex == 1.0)).astype(np.float32)

        return np.column_stack(
            [
                stroke_x_copd,
                diabete_x_stroke,
                stroke_x_poorhlth,
                copd_x_kidney,
                stroke_x_diffwalk,
                diabete_x_copd,
                diabete_x_kidney,
                poorhlth_x_kidney,
                young_female,
                young_male,
            ]
        )

    vasc_tr = extract_vascular_matrix(x_tr_imp)
    vasc_te = extract_vascular_matrix(x_te_imp) if x_te_imp is not None else None

    vasc_names = [
        "STROKE_x_COPD",
        "DIABETE_x_STROKE",
        "STROKE_x_POORHLTH",
        "COPD_x_KIDNEY",
        "STROKE_x_DIFFWALK",
        "DIABETE_x_COPD",
        "DIABETE_x_KIDNEY",
        "POORHLTH_x_KIDNEY",
        "YOUNG_FEMALE",
        "YOUNG_MALE",
    ]

    # Standardize vascular terms using training statistics
    vasc_mean = np.mean(vasc_tr, axis=0)
    vasc_std = np.std(vasc_tr, axis=0)
    vasc_std[vasc_std == 0.0] = 1.0

    vasc_tr = (vasc_tr - vasc_mean) / vasc_std
    if vasc_te is not None:
        vasc_te = (vasc_te - vasc_mean) / vasc_std

    if x_te_imp is not None:
        return vasc_tr, vasc_te, vasc_names
    return vasc_tr, vasc_names


def build_advanced_features(x_tr, x_te=None, feature_names=None, y_tr=None, alpha=20.0):
    """Construct the rich, clinical 233-dimensional design matrix.

    Components:
    1. Base Streamlined Features (205 dims): 197 base + 5 cross-terms + 2 quad + 1 bias.
    2. Extended Empirical Bayesian WoE (8 dims): target-encoded categorical features.
    3. Clinical Subgroup Bias-Correcting Terms (10 dims): interaction terms for elderly/young subgroups.
    4. Vascular Multi-Morbidity Synergy Terms (10 dims): 8 high-lift vascular terms + 2 low-risk anchors.

    Total Dimensions: 205 + 8 + 10 + 10 = 233 dims.
    """
    if feature_names is None:
        raise ValueError("feature_names is required.")
    if y_tr is None:
        raise ValueError(
            "y_tr is required to compute empirical Bayesian WoE encodings."
        )

    # 1. Base streamlined features (205 dims)
    if x_te is not None:
        base_tr, base_te, base_names = build_streamlined_features(
            x_tr,
            x_te,
            feature_names=feature_names,
            add_bias=True,
            standardize_features=True,
        )
    else:
        base_tr, base_names = build_streamlined_features(
            x_tr,
            feature_names=feature_names,
            add_bias=True,
            standardize_features=True,
        )
        base_te = None

    # 2. Extended empirical Bayesian WoE (8 dims)
    if x_te is not None:
        woe_tr, woe_te, woe_names = compute_woe_features(
            x_tr, x_te, feature_names=feature_names, y_tr=y_tr, alpha=alpha
        )
    else:
        woe_tr, woe_names = compute_woe_features(
            x_tr, feature_names=feature_names, y_tr=y_tr, alpha=alpha
        )
        woe_te = None

    # 3. Median impute features strictly from training fold before computing interaction terms
    med = np.nanmedian(x_tr, axis=0)
    med = np.nan_to_num(med, nan=0.0)

    x_tr_imp = x_tr.copy()
    nan_mask_tr = np.isnan(x_tr_imp)
    for j in range(x_tr_imp.shape[1]):
        if np.any(nan_mask_tr[:, j]):
            x_tr_imp[nan_mask_tr[:, j], j] = med[j]

    if x_te is not None:
        x_te_imp = x_te.copy()
        nan_mask_te = np.isnan(x_te_imp)
        for j in range(x_te_imp.shape[1]):
            if np.any(nan_mask_te[:, j]):
                x_te_imp[nan_mask_te[:, j], j] = med[j]
    else:
        x_te_imp = None

    # Clinical subgroup bias-correcting terms (10 dims)
    if x_te is not None:
        sub_tr, sub_te, sub_names = build_clinical_subgroup_terms(
            x_tr_imp, x_te_imp, feature_names=feature_names
        )
    else:
        sub_tr, sub_names = build_clinical_subgroup_terms(
            x_tr_imp, feature_names=feature_names
        )
        sub_te = None

    # 4. Vascular multi-morbidity synergy terms (10 dims)
    if x_te is not None:
        vasc_tr, vasc_te, vasc_names = build_vascular_synergy_terms(
            x_tr_imp, x_te_imp, feature_names=feature_names
        )
    else:
        vasc_tr, vasc_names = build_vascular_synergy_terms(
            x_tr_imp, feature_names=feature_names
        )
        vasc_te = None

    # Assemble 233 dimensions
    tx_tr = np.column_stack([base_tr, woe_tr, sub_tr, vasc_tr])
    out_names = list(base_names) + list(woe_names) + list(sub_names) + list(vasc_names)

    if x_te is not None:
        tx_te = np.column_stack([base_te, woe_te, sub_te, vasc_te])
        return tx_tr, tx_te, out_names
    return tx_tr, out_names


def train_soft_f1_optimizer(
    tx_tr, y01, max_iters=250, lr=0.03, lambda_reg=1e-5, initial_w=None
):
    """Optimize differentiable Soft-F1 metric via Adam.

    Loss function:
        L(w) = 1 - (2 * sum(y_i * s_i)) / (sum(s_i) + sum(y_i) + eps) + lambda * ||w||^2
    where s_i = sigmoid(tx_i @ w).

    Gradient:
        dL/dw = - sum_i ((2 * y_i - F1_soft) / Denom) * s_i * (1 - s_i) * tx_i + 2 * lambda * w

    Args:
        tx_tr (np.ndarray): Design matrix of shape (N, D).
        y01 (np.ndarray): Binary labels in {0, 1} or {-1, 1} of shape (N,).
        max_iters (int): Maximum optimization steps.
        lr (float): Adam learning rate (step size).
        lambda_reg (float): L2 regularization parameter.
        initial_w (np.ndarray, optional): Starting weight vector of shape (D,).

    Returns:
        tuple: (w, loss) final weights and soft-F1 loss.
    """
    y = (y01 == 1).astype(float) if np.any(y01 == -1) else y01.astype(float)
    N, D = tx_tr.shape
    w = np.zeros(D) if initial_w is None else initial_w.copy().astype(float)

    m = np.zeros(D)
    v = np.zeros(D)
    beta1, beta2 = 0.9, 0.999
    eps_adam = 1e-8
    eps_denom = 1e-7
    sum_y = np.sum(y)

    final_loss = 0.0
    for it in range(1, max_iters + 1):
        z = np.clip(tx_tr @ w, -250.0, 250.0)
        s = 1.0 / (1.0 + np.exp(-z))

        num = 2.0 * np.dot(y, s)
        denom = np.sum(s) + sum_y + eps_denom
        f1_soft = num / denom

        c = ((2.0 * y - f1_soft) / denom) * s * (1.0 - s)
        grad = -tx_tr.T @ c + 2.0 * lambda_reg * w

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad**2)
        m_hat = m / (1.0 - beta1**it)
        v_hat = v / (1.0 - beta2**it)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        final_loss = 1.0 - f1_soft + lambda_reg * np.sum(w**2)

    return w, final_loss


class IsotonicCalibrator:
    """Non-parametric monotonic probability calibration via Pool Adjacent Violators Algorithm (PAVA)."""

    def __init__(self):
        self.knots_s = None
        self.knots_p = None

    def fit(self, scores, y):
        """Fit a monotonically increasing step function from continuous scores to probabilities."""
        scores = np.asarray(scores, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        order = np.argsort(scores)
        sorted_s = scores[order]
        sorted_y = y[order]

        # Vectorized/stack PAVA
        stack = []  # stores [block_weight, block_weighted_y]
        for yi in sorted_y:
            cur_w = 1.0
            cur_wy = yi
            while stack and (stack[-1][1] / stack[-1][0]) >= (cur_wy / cur_w):
                prev_w, prev_wy = stack.pop()
                cur_w += prev_w
                cur_wy += prev_wy
            stack.append((cur_w, cur_wy))

        # Construct knot coordinates for monotonic piecewise linear interpolation
        knots_s = []
        knots_p = []
        cum_idx = 0
        for block_w, block_wy in stack:
            n_samples = int(round(block_w))
            block_s = np.mean(sorted_s[cum_idx : cum_idx + n_samples])
            knots_s.append(block_s)
            knots_p.append(block_wy / block_w)
            cum_idx += n_samples

        self.knots_s = np.array(knots_s, dtype=np.float64)
        self.knots_p = np.array(knots_p, dtype=np.float64)
        return self

    def predict_proba(self, scores):
        """Predict calibrated probabilities in [0, 1] via monotonic interpolation."""
        scores = np.asarray(scores, dtype=np.float64)
        p = np.interp(
            scores,
            self.knots_s,
            self.knots_p,
            left=self.knots_p[0],
            right=self.knots_p[-1],
        )
        return np.clip(p, 0.0, 1.0)


class FastHistBoostedTrees:
    """Fast histogram-based gradient boosted decision trees (Depth=2) in pure NumPy."""

    def __init__(
        self,
        n_estimators=35,
        learning_rate=0.15,
        l2_reg=1.0,
        n_bins=32,
        max_features=50,
        w_pos=2.5,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.n_bins = n_bins
        self.max_features = max_features
        self.w_pos = w_pos
        self.trees = []
        self.bin_edges = []
        self.feature_indices = None
        self.base_score = 0.0

    def fit(self, X, y):
        N, D = X.shape
        y01 = y.astype(np.float64)
        sample_weights = np.where(y01 == 1.0, self.w_pos, 1.0)

        # Select top features by variance
        if self.max_features < D:
            var = np.var(X, axis=0)
            self.feature_indices = np.argsort(var)[-self.max_features :]
            X_sub = X[:, self.feature_indices]
        else:
            self.feature_indices = np.arange(D)
            X_sub = X

        K = len(self.feature_indices)
        self.bin_edges = []
        X_binned = np.empty((N, K), dtype=np.uint8)
        percentiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]

        for j in range(K):
            edges = np.unique(np.percentile(X_sub[:, j], percentiles))
            self.bin_edges.append(edges)
            X_binned[:, j] = np.digitize(X_sub[:, j], edges).astype(np.uint8)

        p_mean = np.clip(
            np.sum(sample_weights * y01) / np.sum(sample_weights), 1e-4, 1 - 1e-4
        )
        self.base_score = float(np.log(p_mean / (1.0 - p_mean)))
        F = np.full(N, self.base_score, dtype=np.float64)
        self.trees = []
        lam = self.l2_reg

        for t in range(self.n_estimators):
            p = 1.0 / (1.0 + np.exp(-F))
            g = sample_weights * (p - y01)
            h = sample_weights * np.maximum(p * (1.0 - p), 1e-6)

            best_feat, best_bin, best_gain = 0, 0, -1.0
            G_tot, H_tot = np.sum(g), np.sum(h)
            root_score = (G_tot**2) / (H_tot + lam)

            for j in range(K):
                bj = X_binned[:, j]
                n_b = len(self.bin_edges[j]) + 1
                g_b = np.bincount(bj, weights=g, minlength=n_b)
                h_b = np.bincount(bj, weights=h, minlength=n_b)
                G_L = np.cumsum(g_b[:-1])
                H_L = np.cumsum(h_b[:-1])
                G_R = G_tot - G_L
                H_R = H_tot - H_L
                gains = 0.5 * (
                    (G_L**2) / (H_L + lam) + (G_R**2) / (H_R + lam) - root_score
                )
                max_g = np.max(gains)
                if max_g > best_gain:
                    best_gain, best_feat, best_bin = (
                        max_g,
                        j,
                        int(np.argmax(gains)),
                    )

            left_mask = X_binned[:, best_feat] <= best_bin
            right_mask = ~left_mask

            def find_child_split(mask):
                if np.sum(mask) < 20:
                    g_sum, h_sum = np.sum(g[mask]), np.sum(h[mask])
                    w = -g_sum / (h_sum + lam)
                    return None, None, w, w
                g_m, h_m = g[mask], h[mask]
                G_c, H_c = np.sum(g_m), np.sum(h_m)
                score_c = (G_c**2) / (H_c + lam)
                c_feat, c_bin, c_gain = 0, 0, -1.0
                for j in range(K):
                    bj = X_binned[mask, j]
                    n_b = len(self.bin_edges[j]) + 1
                    g_b = np.bincount(bj, weights=g_m, minlength=n_b)
                    h_b = np.bincount(bj, weights=h_m, minlength=n_b)
                    G_L = np.cumsum(g_b[:-1])
                    H_L = np.cumsum(h_b[:-1])
                    G_R = G_c - G_L
                    H_R = H_c - H_L
                    gains = 0.5 * (
                        (G_L**2) / (H_L + lam) + (G_R**2) / (H_R + lam) - score_c
                    )
                    max_g = np.max(gains)
                    if max_g > c_gain:
                        c_gain, c_feat, c_bin = (
                            max_g,
                            j,
                            int(np.argmax(gains)),
                        )
                l_m = X_binned[mask, c_feat] <= c_bin
                r_m = ~l_m
                w_L = (
                    -np.sum(g_m[l_m]) / (np.sum(h_m[l_m]) + lam)
                    if np.sum(l_m) > 0
                    else 0.0
                )
                w_R = (
                    -np.sum(g_m[r_m]) / (np.sum(h_m[r_m]) + lam)
                    if np.sum(r_m) > 0
                    else 0.0
                )
                return c_feat, c_bin, w_L, w_R

            j_L, b_L, w_LL, w_LR = find_child_split(left_mask)
            j_R, b_R, w_RL, w_RR = find_child_split(right_mask)

            update = np.zeros(N, dtype=np.float64)
            if j_L is not None:
                m_LL = left_mask & (X_binned[:, j_L] <= b_L)
                m_LR = left_mask & (~(X_binned[:, j_L] <= b_L))
                update[m_LL], update[m_LR] = w_LL, w_LR
            else:
                update[left_mask] = w_LL
            if j_R is not None:
                m_RL = right_mask & (X_binned[:, j_R] <= b_R)
                m_RR = right_mask & (~(X_binned[:, j_R] <= b_R))
                update[m_RL], update[m_RR] = w_RL, w_RR
            else:
                update[right_mask] = w_RL

            F += self.learning_rate * update
            self.trees.append(
                {
                    "root_feat": best_feat,
                    "root_bin": best_bin,
                    "j_L": j_L,
                    "b_L": b_L,
                    "w_LL": w_LL,
                    "w_LR": w_LR,
                    "j_R": j_R,
                    "b_R": b_R,
                    "w_RL": w_RL,
                    "w_RR": w_RR,
                }
            )
        return self

    def predict_raw(self, X):
        X_sub = X[:, self.feature_indices]
        N, K = X_sub.shape
        X_binned = np.empty((N, K), dtype=np.uint8)
        for j in range(K):
            X_binned[:, j] = np.digitize(X_sub[:, j], self.bin_edges[j]).astype(
                np.uint8
            )

        F = np.full(N, self.base_score, dtype=np.float64)
        for tree in self.trees:
            update = np.zeros(N, dtype=np.float64)
            left_mask = X_binned[:, tree["root_feat"]] <= tree["root_bin"]
            right_mask = ~left_mask
            if tree["j_L"] is not None:
                m_LL = left_mask & (X_binned[:, tree["j_L"]] <= tree["b_L"])
                m_LR = left_mask & (~(X_binned[:, tree["j_L"]] <= tree["b_L"]))
                update[m_LL], update[m_LR] = tree["w_LL"], tree["w_LR"]
            else:
                update[left_mask] = tree["w_LL"]
            if tree["j_R"] is not None:
                m_RL = right_mask & (X_binned[:, tree["j_R"]] <= tree["b_R"])
                m_RR = right_mask & (~(X_binned[:, tree["j_R"]] <= tree["b_R"]))
                update[m_RL], update[m_RR] = tree["w_RL"], tree["w_RR"]
            else:
                update[right_mask] = tree["w_RL"]
            F += self.learning_rate * update
        return F
