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
| **M4** | Feature Engineering | `[ ]` | `features.py` | Missing imputation, scaling, polynomial/cross terms |
| **M5** | Model Tuning & Selection | `[ ]` | `tune.py` | Regularized Logistic Regression vs Ridge grid search |
| **M6** | Reproducible Submission | `[ ]` | `run.py` & `README.md` | Match `sample_submission.csv` format |
| **M7** | 2-Page Scientific Report | `[ ]` | LaTeX PDF (`IEEEtran`) | Include baseline comparison table & ablation study |

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
| M3-Logistic-Tuned | 2026-10-22 | 5-Fold Reg Logistic (lambda=1e-4) calibrated tau=0.218 | 0.8722 | 0.4277 +/- 0.0027 | 0.7224 | - |
