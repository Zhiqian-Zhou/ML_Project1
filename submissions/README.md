# Submissions & Experiment Tracking

`run.py` writes output csv files here (`submissions/*.csv` are git-ignored).

---

## 1. Official AIcrowd Submissions Log
Log every official AIcrowd submission here so the entire team can trace which commit and hyperparameters produced each leaderboard score:

| Date | Submission ID | Commit | Model & Configuration | CV F1 (+1) | AIcrowd F1 | AIcrowd Acc | Notes & Key Changes |
|---|---|---|---|:---:|:---:|:---:|---|
| 2026-10-22 | Sub-01 (`submission_baseline_logistic.csv`) | `3d33eb3` | Regularized Logistic ($\lambda=10^{-4}, \gamma=0.1, \tau^*=0.218$) | 0.4277 | - | - | First official baseline submission (198 dims) |
| 2026-10-22 | Sub-02 (`submission_feature_engineered.csv`) | `zhihao` | Feature-Engineered Logistic ($\lambda=10^{-4}, \gamma=0.15, \tau^*=0.210$) | 0.4311 | - | - | 333 dims: MNAR indicators + Nominal OHE + 5 Clinical interactions |

---

## 2. Table 2: Progressive Feature Engineering Ablation Study (Milestone 4)
Evaluated via pure-NumPy Stratified 5-Fold Cross-Validation (seed=42) on `data/processed/` with strict zero-leakage threshold calibration on training folds.

| Step | Transformation / Hypothesis | Total Dims ($D$) | Val Accuracy | Val Precision | Val Recall | Balanced Acc | Val F1 (+1) | Calibrated $\tau^*$ | Marginal $\Delta F_1$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 0** | Baseline (197 sanitized features) | 198 | $87.22\%$ | $0.3543$ | $54.05\%$ | $72.24\%$ | $0.4277 \pm 0.0027$ | $0.218$ | - |
| **Step 1** | + Missingness Indicators ($M_{ij} \in [5\%, 85\%]$) | 312 | $87.21\%$ | $0.3540$ | $\mathbf{54.25\%}$ | $72.32\%$ | $0.4283 \pm 0.0031$ | $0.218$ | $+0.0006$ |
| **Step 2** | + Nominal One-Hot (`MARITAL`, `EMPLOY1`, `_RACE`) | 328 | $87.24\%$ | $0.3548$ | $54.24\%$ | $\mathbf{72.34\%}$ | $0.4288 \pm 0.0026$ | $0.218$ | $+0.0005$ |
| **Step 3** | + Clinical Non-linear Interactions (5 Cross-terms) | 333 | $87.54\%$ | $0.3606$ | $53.21\%$ | $72.04\%$ | $0.4298 \pm 0.0021$ | $0.224$ | $+0.0010$ |
| **Step 4** | + Tuned Hyperparameters ($\lambda^*=10^{-4}, \gamma^*=0.15$) | 333 | $\mathbf{87.69\%}$ | $\mathbf{0.3646}$ | $52.83\%$ | $71.95\%$ | $\mathbf{0.4311 \pm 0.0018}$ | $0.212$ | $+0.0013$ |

---

## 3. Local 5-Fold Stratified CV Baseline Benchmark
Evaluated across baselines to establish the lower bound performance on sanitized features:

| Exp ID | Model Architecture | Threshold $\tau$ | Val Accuracy | Val Precision | Val Recall | Balanced Acc | Val F1 (+1) | Notes / Improvements |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **B0** | Majority Class Dummy | N/A | $91.17\%$ | $0.0000$ | $0.0000$ | $50.00\%$ | $0.0000$ | Trivial all -1 baseline |
| **B1a** | Ridge ($\lambda=10^{-4}$) | Default ($0.0$) | $91.58\%$ | $0.7774$ | $0.0650$ | $53.16\%$ | $0.1198$ | Severe false-negative bias |
| **B1b** | **Ridge ($\lambda=10^{-4}$)** | **Tuned ($-0.569$)** | $87.06\%$ | $0.3490$ | $\mathbf{53.77\%}$ | $\mathbf{72.03\%}$ | $\mathbf{0.4231}$ | **+253% F1 gain via threshold tuning** |
| **B2a** | Logistic ($\lambda=10^{-4}$) | Default ($0.5$) | $91.65\%$ | $0.6192$ | $0.1421$ | $56.68\%$ | $0.2311$ | Sigmoid default misses >85% cases |
| **B2b** | **Logistic ($\lambda=10^{-4}$)** | **Tuned ($0.218$)** | $87.22\%$ | $0.3543$ | $\mathbf{54.05\%}$ | $\mathbf{72.24\%}$ | $\mathbf{0.4277}$ | **Highest baseline F1; +85% over tau=0.5** |

---

## 4. Negative Results & Pruning Rationale (What Didn't Work)

To ensure scientific rigor and avoid confirmation bias, we explicitly document empirical negative results:

1. **Indiscriminate Missingness Indicator Masks (MNAR) Acted as Noise**:
   - Appending 114 binary missingness indicators $M_{ij} = \mathbb{I}(X_{ij} \text{ was NaN})$ increased the feature dimensionality by $+57.6\%$ (from 198 to 312 dims).
   - Despite this substantial increase in complexity, cross-validation $F_1$ barely budged ($+0.0006$, from $0.4277 \to 0.4283$).
   - *Rationale*: While specific conditional skip patterns (like `HAREHAB1_ANSWERED`) carry strong deterministic signals, the vast majority of missing values in BRFSS survey questions arise from regional questionnaire module omissions (split-form surveying) rather than individual health status. Appending 114 masks primarily introduced high-dimensional collinearity, slower gradient descent convergence, and noise.

2. **Nominal One-Hot Categorical Expansion Added Sparsity Without Margin Gain**:
   - One-hot encoding `MARITAL`, `EMPLOY1`, and `_RACE` expanded the feature space by an additional 16 dimensions (328 dims) but produced a negligible $+0.0005$ F1 gain.
   - *Rationale*: Although theoretical linear distortion is removed, the low-prevalence dummy categories (e.g. `_RACE == 5` with 0.3% frequency) contributed near-zero gradient signal and induced matrix ill-conditioning.

3. **Occam's Razor & Pipeline Simplification**:
   - By pruning these 130 redundant and noisy indicator columns, we restored a clean, highly interpretable **205-dimensional design matrix**:
     - 197 clean base features.
     - 5 verified clinical non-linear cross-terms (`HBP_x_CHOL`, `AGE_x_HBP`, `AGE_x_SMOKE`, `AGE_x_BMI`, `INACTIVE_x_POORHLTH`).
     - 2 physiological quadratic risk terms (`_BMI5^2`, `_AGEG5YR^2`).
   - This compact representation eliminates feature bloat, reduces memory overhead, and runs in sub-second time.

---

## 5. Cost-Sensitive Class-Weighted Learning Benchmark (Streamlined 205 Dims)

Because the dataset exhibits severe positive class imbalance (8.83% positive), unweighted loss gradients are heavily dominated by the majority negative class. We implemented pure-NumPy sample-weighted Ridge Regression via the exact closed-form normal equations:
$$\mathbf{w}^* = (\mathbf{X}^T \mathbf{W} \mathbf{X} + 2N\lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{y}$$
where sample weight $w_i = w_{pos}$ for positive cases ($y_i = +1$), and $1.0$ for negative cases ($y_i = -1$).

### Stratified 5-Fold Cross-Validation Comparison:

| Model Architecture | Positive Weight ($w_{pos}$) | Val Accuracy | Val Precision | Val Recall | Balanced Acc | Val F1 (+1) | Calibrated $\tau^*$ | 5-Fold Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Weighted Ridge** | $1.0$ | $87.39\%$ | $0.3565$ | $53.10\%$ | $71.91\%$ | $0.4265 \pm 0.0032$ | $-0.5560$ | **7.71s** |
| **Weighted Ridge** | $1.5$ | $87.36\%$ | $0.3570$ | $53.70\%$ | $72.16\%$ | $0.4287 \pm 0.0027$ | $-0.4200$ | **8.16s** |
| **Weighted Ridge** | $2.0$ | $87.27\%$ | $0.3558$ | $\mathbf{54.34\%}$ | $\mathbf{72.40\%}$ | $0.4299 \pm 0.0030$ | $-0.3160$ | **7.59s** |
| **Weighted Ridge** | $2.5$ | $87.35\%$ | $0.3574$ | $54.18\%$ | $72.37\%$ | $0.4306 \pm 0.0031$ | $-0.2240$ | **7.09s** |
| **Weighted Ridge** | **$3.0$** | $\mathbf{87.56\%}$ | $\mathbf{0.3615}$ | $53.35\%$ | $72.11\%$ | $\mathbf{0.4309 \pm 0.0032}$ | $-0.1400$ | **7.53s** |
| **Weighted Ridge** | $5.0$ | $87.52\%$ | $0.3608$ | $53.41\%$ | $72.12\%$ | $0.4305 \pm 0.0031$ | $+0.0720$ | **8.73s** |
| **Regularized Logistic** | $1.0$ | $87.41\%$ | $0.3586$ | $53.92\%$ | $72.29\%$ | $0.4307 \pm 0.0028$ | $+0.2080$ | 52.54s |

### Key Insights:
1. **Cost-Sensitivity Matches Iterative Gradient Descent**: Weighted Ridge Regression ($w_{pos}=3.0$) matches Regularized Logistic Regression in F1 ($0.4309$ vs $0.4307$), but runs in **7.53 seconds (7.0x faster)**.
2. **Threshold Centering**: Increasing $w_{pos}$ from $1.0 \to 3.0 \to 5.0$ naturally shifts raw decision scores toward zero ($\tau^*$ shifts from $-0.556 \to -0.140 \to +0.072$), reducing score skewness caused by extreme class imbalance.

---

## 6. Critical Findings & Pitfalls Avoided (For Teammates)

Please review these findings before writing modeling or preprocessing code:

1. **Extreme Class Imbalance (8.83% Positives)**:
   - Always predicting `-1` gives **91.17% accuracy**, but **0.0000 F1**.
   - **Never judge models by accuracy alone!** AIcrowd leaderboard ranking is driven by positive class F1.
   - Cost-sensitive weighting ($w_{pos} \in [2.5, 3.0]$) directly counters majority class dominance in loss gradients.

2. **CDC BRFSS Survey Code Traps**:
   - `PHYSHLTH`, `MENTHLTH`, `POORHLTH`: Code `88` represents **0 days of illness** (healthy), NOT 88 days! Over 204,000 respondents answered `88`. Treating it as numeric treats healthy people as chronically ill.
   - `ALCDAY5`, `STRENGTH`: Code `888` means **0 drinks / 0 exercises**.
   - `7`, `9`, `77`, `99`, `777`, `999`, `7777`, `9999`: CDC codes for "Don't know / Refused". Must be converted to `np.nan` before computing numerical medians or standard deviations.
   - `HEIGHT3` & `WEIGHT2`: Unit-mixed (feet vs cm, lbs vs kg) and contains `9999` (refused). Always use the standardized CDC variables `HTM4` (height in meters), `WTKG3` (weight in kg), and `_BMI5`.

3. **Survey Branching & Target Leakage (`HAREHAB1`)**:
   - `HAREHAB1` (cardiac rehab attendance) is only asked in the survey if the respondent previously answered "Yes" to having had a heart attack (`CVDINFR4 == 1`).
   - In `x_train`, 100% of respondents who answered `HAREHAB1` (773 / 773) have label $y = +1$.
   - In `x_test`, exactly 238 respondents answered `HAREHAB1`.
   - **Handling**: Keep `HAREHAB1` and append a binary indicator column `HAREHAB1_ANSWERED`.

4. **I/O Speedup via Native Binary Cache**:
   - Reading `x_train.csv` takes ~11s.
   - Use `data/processed/x_train.npy` and `y_train.npy` (197 sanitized features, 0 sentinels). Loads in **30 ms (349x faster)**.


