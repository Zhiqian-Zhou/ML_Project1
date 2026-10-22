# Submissions & Experiment Tracking

`run.py` writes output csv files here (`submissions/*.csv` are git-ignored).

---

## 1. Official AIcrowd Submissions Log
Log every official AIcrowd submission here so the entire team can trace which commit and hyperparameters produced each leaderboard score:

| Date | Submission ID | Commit | Model & Configuration | CV F1 (+1) | AIcrowd F1 | AIcrowd Acc | Notes & Key Changes |
|---|---|---|---|:---:|:---:|:---:|---|
| 2026-10-22 | Sub-01 (`submission_baseline_logistic.csv`) | `3d33eb3` | Regularized Logistic ($\lambda=10^{-4}, \gamma=0.1, \tau^*=0.218$) | 0.4277 | - | - | First official baseline submission |

---

## 2. Local 5-Fold Stratified CV Benchmark (Ablation Ledger)
All evaluation results are evaluated via pure NumPy 5-fold stratified cross-validation on `data/processed/x_train.npy` (197 sanitized features, 328,135 samples).

| Exp ID | Model Architecture | Threshold $\tau$ | Val Accuracy | Val Precision | Val Recall | Balanced Acc | Val F1 (+1) | Notes / Improvements |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **B0** | Majority Class Dummy | N/A | $91.17\%$ | $0.0000$ | $0.0000$ | $50.00\%$ | $0.0000$ | Trivial all -1 baseline |
| **B1a** | Ridge ($\lambda=10^{-4}$) | Default ($0.0$) | $91.58\%$ | $0.7774$ | $0.0650$ | $53.16\%$ | $0.1198$ | Severe false-negative bias |
| **B1b** | **Ridge ($\lambda=10^{-4}$)** | **Tuned ($-0.569$)** | $87.06\%$ | $0.3490$ | $\mathbf{53.77\%}$ | $\mathbf{72.03\%}$ | $\mathbf{0.4231}$ | **+253% F1 gain via threshold tuning** |
| **B2a** | Logistic ($\lambda=10^{-4}$) | Default ($0.5$) | $91.65\%$ | $0.6192$ | $0.1421$ | $56.68\%$ | $0.2311$ | Sigmoid default misses >85% cases |
| **B2b** | **Logistic ($\lambda=10^{-4}$)** | **Tuned ($0.218$)** | $87.22\%$ | $0.3543$ | $\mathbf{54.05\%}$ | $\mathbf{72.24\%}$ | $\mathbf{0.4277}$ | **Highest baseline F1; +85% over tau=0.5** |

---

## 3. Critical Findings & Pitfalls Avoided (For Teammates)

Please review these findings before writing modeling or preprocessing code:

1. **Extreme Class Imbalance (8.83% Positives)**:
   - Always predicting `-1` gives **91.17% accuracy**, but **0.0000 F1**.
   - **Never judge models by accuracy alone!** AIcrowd leaderboard ranking is driven by positive class F1.
   - Default thresholds ($\tau = 0.5$ for Logistic, $\tau = 0.0$ for Ridge) severely penalize positive recall.
   - **Solution**: Dynamic threshold tuning $\tau^* \in [0.05, 0.50]$ on training folds improves F1 from $0.23 \to 0.43$ and detects $>54\%$ of true cardiovascular cases.

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

