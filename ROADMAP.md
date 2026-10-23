# Project 1 Research Ledger & Experiment Tracker

## Project Overview
- **Course**: EPFL CS-433 Machine Learning (Fall 2026)
- **Target**: Predict `_MICHD` (Coronary Heart Disease / Myocardial Infarction) from CDC BRFSS 2015 dataset
- **Class Balance**: 8.83% positive (+1), 91.17% negative (-1)
- **Evaluation Priority**: Positive Class F1-Score, Balanced Accuracy (with threshold calibration)

---

## Milestone Status Board

| Milestone | Scope | Status | Key Deliverable | Notes / Evidence |
| :--- | :--- | :---: | :--- | :--- |
| **M1** | Core ML Functions | `[x]` | `implementations.py` | 27/27 unit tests passed with 100% precision |
| **M2** | Data Hygiene & Cache | `[x]` | `preprocess.py` & `.npy` cache | Cleaned 88/888 codes, pruned 125 features, 349.2x I/O speedup |
| **M3** | Pure-NumPy CV Harness | `[x]` | `cross_validation.py` & `metrics.py` | Stratified 5-Fold, F1=0.4277 (Logistic), F1=0.4231 (Ridge) |
| **M4** | Feature Engineering | `[x]` | `feature_engineering.py` | MNAR masks, nominal one-hot, 5 clinical interactions (F1=0.4311) |
| **M5** | Bold Hypotheses & Ensembles | `[x]` | `benchmark_hypotheses.py` | 4 hypotheses evaluated: WoE, Rank Blend, 2-Layer MLP, Multi-morbidity |
| **M6** | SOTA Submission Generation | `[x]` | `run.py` & `submissions/` | `submission_sota.csv` (109,380 rows verified, peak CV F1=0.4320) |
| **M7** | 2-Page Scientific Report | `[ ]` | LaTeX PDF (`IEEEtran`) | Include baseline comparison table, ablation study & negative results |

---

## Benchmark Log (Ablation History)
*Record each experimental step quantitatively here:*

| Exp ID | Date | Description / Hypothesis | Val Acc | Val F1 (+1) | Balanced Acc | Submission Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Baseline-0 | 2026-10-22 | All negative (-1) majority class predictor | 0.9117 | 0.0000 | 0.5000 | - |
| M1-Test | 2026-10-22 | 6 canonical functions unit tests (`grading_tests/`) | N/A | N/A | N/A | 27/27 Passed |
| M2-Hygiene | 2026-10-22 | CDC survey code mapping, 197 clean features, binary cache | N/A | N/A | N/A | 349.2x I/O speedup (0.03s) |
| M3-Ridge-Untuned | 2026-10-22 | 5-Fold Ridge (lambda=1e-4) default tau=0.0 | 0.9158 | 0.1198 +/- 0.0106 | 0.5316 | - |
| M3-Ridge-Tuned | 2026-10-22 | 5-Fold Ridge (lambda=1e-4) calibrated tau=-0.569 | 0.8706 | 0.4231 +/- 0.0044 | 0.7203 | - |
| M3-Logistic-Untuned | 2026-10-22 | 5-Fold Reg Logistic (lambda=1e-4) default tau=0.5 | 0.9165 | 0.2311 +/- 0.0052 | 0.5668 | - |
| M3-Logistic-Tuned | 2026-10-22 | 5-Fold Reg Logistic (lambda=1e-4) calibrated tau=0.218 | 0.8722 | 0.4277 +/- 0.0027 | 0.7224 | Sub-01 |
| M4-Step1-Missing | 2026-10-22 | Step 0 + Informative Missingness Indicators (MNAR 5-85%) | 0.8721 | 0.4283 +/- 0.0031 | 0.7232 | - |
| M4-Step2-Nominal | 2026-10-22 | Step 1 + Nominal One-Hot (`MARITAL`, `EMPLOY1`, `_RACE`) | 0.8724 | 0.4288 +/- 0.0026 | 0.7234 | - |
| M4-Step3-Interactions | 2026-10-22 | Step 2 + 5 Clinical Non-linear Interaction Cross-Terms | 0.8754 | 0.4298 +/- 0.0021 | 0.7204 | - |
| M4-Step4-Tuned | 2026-10-22 | Step 3 + Tuned Hyperparameters (lambda=1e-4, gamma=0.15) | 0.8769 | 0.4311 +/- 0.0018 | 0.7195 | Sub-02 |
| M4-Streamlined-Ridge | 2026-10-22 | Cost-Sensitive Ridge (w_pos=3.0, 205 dims: 197 base + 5 cross + 2 quad) | 0.8756 | 0.4309 +/- 0.0032 | 0.7211 | 7.5s (Sub-02b) |
| M4-Streamlined-Log | 2026-10-22 | Regularized Logistic on Streamlined 205-dim space | 0.8741 | 0.4307 +/- 0.0028 | 0.7229 | 52.5s |
| M5-Hypothesis-A | 2026-10-23 | Empirical Bayesian Log-Odds (WoE) + Weighted Ridge (209 dims) | 0.8756 | 0.4315 +/- 0.0032 | 0.7216 | Validated (9.7s) |
| M5-Hypothesis-B | 2026-10-23 | Multi-Model Heterogeneous Rank Blend + WoE (0.45 Ridge + 0.35 Log + 0.20 Bag) | 0.8749 | **0.4320 +/- 0.0033** | 0.7230 | **Sub-03 (SOTA)** |
| M5-Hypothesis-C | 2026-10-23 | Pure-NumPy 2-Layer Neural Network (D=205 -> H=64 ReLU -> 1 Sigmoid) | 0.8778 | 0.4272 +/- 0.0020 | 0.7145 | Train F1=0.4644 (Overfitting) |
| M5-Hypothesis-D | 2026-10-23 | Clinical Multi-morbidity Stacking Score (0-7 index + Ridge) | 0.8757 | 0.4312 +/- 0.0032 | 0.7213 | Competitive (8.1s) |
| M6-Super-Blend | 2026-10-23 | Super Tri-Blend (223 dims: 205 base + 8 WoE + 10 Subgroups; Ridge + Bagging + Soft-F1) | 0.8758 | 0.4335 +/- 0.0034 | 0.7233 | Sub-04 (M6 Peak) |
| M6.5-Resampling-Blend | 2026-10-23 | Hard-Negative Bagging Tri-Blend (223 dims; weights [0.35, 0.40, 0.25]) | 0.8776 | **0.4341 +/- 0.0038** | 0.7215 | **Sub-05 (All-Time Peak)** |


