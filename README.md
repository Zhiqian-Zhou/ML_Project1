# EPFL CS-433 Machine Learning (Fall 2026) — Project 1
## Predicting Coronary Heart Disease (`_MICHD`) from CDC BRFSS 2015

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Pure NumPy](https://img.shields.io/badge/Dependencies-NumPy%20Only-success.svg)](https://numpy.org/)
[![Autograder](https://img.shields.io/badge/Course%20Autograder-27%2F27%20Passed-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

**Team Members:** Zhihao Chen, Zhiqian Zhou, Guanyu Chen  
**Best Leaderboard Submission:** **F1 = 0.457**, Accuracy = 88.3% (14,001 positive predictions)  
**Clean SOTA Submission:** **F1 = 0.447**, Accuracy = 87.7% (14,876 positive predictions, Zero-Leakage Screening)

---

## 1. Project Overview

The goal of this project is binary classification on the CDC Behavioral Risk Factor Surveillance System (BRFSS 2015) dataset:
* **Data Scale:** 328,135 training respondents, 109,379 test respondents, 321 survey features.
* **Target:** Predict whether an individual has coronary heart disease or myocardial infarction (`_MICHD` $\in \{-1, +1\}$).
* **Challenge:** Severe class imbalance (**8.83% positive prevalence**, ~10:1 ratio). A naive all-negative classifier scores 91.17% accuracy but an $F_1$ score of **0.0000**. All tuning and model selection are guided by positive-class $F_1$.
* **Constraints:** Strictly pure Python standard library and NumPy (no pandas, scikit-learn, or torch).

---

## 2. Machine Learning Pipeline

### Data Preprocessing & Cleaning (`preprocessing.py`, `eda/`)
* **Sentinel Value Mapping:** Survey sentinels like `7`, `9`, `77`, `99` (don't know / refused) are converted to `NaN`. Special codes like `88` (none / zero days) and `888` (no alcohol) are mapped to `0`.
* **Skip-Logic Imputation:** Conditional survey questions (e.g., eye exams skipped by non-diabetics) are filled with domain-appropriate zero-risk values rather than generic column medians.
* **Feature Construction:** Clinically grounded non-linear interaction terms (e.g., Blood Pressure $\times$ Cholesterol) and continuous empirical Bayes risk encodings.
* **Strict Leakage Prevention:** All scalers, imputation statistics, and risk priors are computed strictly on training folds and applied out-of-fold.

### Core Baseline Methods (`implementations.py`)
We implement the 6 course-mandated optimization methods from scratch:
1. `mean_squared_error_gd` — Linear regression via gradient descent.
2. `mean_squared_error_sgd` — Linear regression via stochastic gradient descent (batch size 1).
3. `least_squares` — Least squares regression via normal equations.
4. `ridge_regression` — L2-regularized least squares regression.
5. `logistic_regression` — Logistic regression via gradient descent.
6. `reg_logistic_regression` — Regularized logistic regression via gradient descent.

*All 27 public autograder tests in `../grading_tests/` pass with 100% compliance.*

### Advanced Models (`gbm.py`, `mlp.py`)
* **Gradient Boosted Decision Trees (`gbm.py`):** Pure-NumPy Histogram Gradient Boosting with 64 bins, depth 5, and Newton leaf updates. Bagged over 3 random seeds.
* **Multi-Layer Perceptron (`mlp.py`):** Pure-NumPy Neural Network with 64 ReLU hidden units, Adam optimizer, and weight decay, averaged over 10 random initialization seeds.
* **Ensemble:** Score blend of $0.50 \times \text{GBM} + 0.50 \times \text{MLP}$ with calibrated decision threshold.

### Validation Protocol (`cross_validation.py`, `cv_final.py`)
* Stratified 5-Fold Cross-Validation preserves the exact 8.83% positive ratio in every fold.
* Decision threshold $\tau^*$ is calibrated strictly on training folds to maximize $F_1$.

---

## 3. Progressive Experimental Results

| Step | Model & Configuration | Feature Dims ($D$) | 5-Fold CV F1 | 5-Fold CV Acc | Key Impact / What Changed? |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline 0** | Naive Majority Predictor | — | 0.0000 | 91.17% | High accuracy, but zero predictive utility |
| **Baseline 1** | Ridge Regression (`implementations.py`) | 198 | 0.1198 | 91.58% | Default 0 cutoff produces severe false negatives |
| **Baseline 2** | Logistic Regression (`implementations.py`) | 198 | 0.2311 | 91.65% | Default 0.5 cutoff produces severe false negatives |
| **Calibrated** | Tuned Logistic Regression | 198 | **0.4277** | 87.22% | **+85% jump via threshold calibration ($\tau^*=0.218$)** |
| **Features** | Cleaned Survey Data + Cross-Terms | 158 | **0.4311** | 87.69% | Removes survey code spikes; adds clinical interactions |
| **Clean SOTA** | GBM (3 seeds) + MLP (10 seeds) | 164 | **0.4372** | **87.80%** | Non-linear tree + neural net ensemble (Zero-Leakage) |
| **Final Model**| GBM + MLP + Post-Event Survey Signals | 171 | **0.4553** | **88.60%** | Incorporates survey signals (Reproduces 0.457 Leaderboard) |

*For in-depth analysis of rejected hypotheses (SMOTE, class weighting, deep networks, stacking), see [`notebooks/03_negative_results.ipynb`](notebooks/03_negative_results.ipynb).*

---

## 4. Quickstart & Reproducibility

### Environment Setup
Only standard Python and NumPy are required:
```bash
# Clone the repository
git clone https://github.com/Zhiqian-Zhou/ML_Project1.git
cd ML_Project1

# Setup environment (Python 3.10)
python -m venv .venv
source .venv/bin/activate
pip install numpy pytest black
```

### Reproducing Competition Predictions
The script `run.py` trains the models and generates the exact submission files:

```bash
# 1. Reproduce Track B Leaderboard SOTA (F1 = 0.457, exactly 14,001 positives)
python run.py --with_leakage --output submissions/submission_with_leakage.csv

# 2. Reproduce Track A Clean SOTA (F1 = 0.447, exactly 14,876 positives, zero leakage)
python run.py --output submissions/submission_clean_sota.csv
```

### Running Validation & Tests
```bash
# Run 5-fold cross-validation on the final ensemble
python cv_final.py

# Run unit tests
pytest tests/

# Run EPFL course autograder tests
./run_public_tests.sh
```

---

## 5. Repository Structure

```text
ML_Project1/
├── README.md                      # Executive overview, pipeline description & quickstart
├── implementations.py             # 6 course required baseline methods (GD, SGD, LS, Ridge, Logistic, Reg-Logistic)
├── run.py                         # Standalone submission generator reproducing best predictions
├── cv_final.py                    # 5-fold cross-validation runner for the final ensemble
├── cross_validation.py            # Stratified K-fold split & threshold calibration harness
├── gbm.py                         # Pure-NumPy Histogram Gradient Boosted Decision Trees
├── mlp.py                         # Pure-NumPy Multi-Seed Multi-Layer Perceptron (Adam)
├── preprocessing.py               # BRFSS survey cleaning, imputation & feature pipeline
├── metrics.py                     # Accuracy, precision, recall, F1-score & confusion matrix
├── helpers.py                     # CSV streaming loaders & submission export utilities
├── run_public_tests.sh            # Public autograder test runner script
│
├── data/                          # Dataset folder (symlinked raw CSVs, git-ignored)
├── eda/                           # Domain codebooks & feature specifications
│   ├── column_decisions.csv       # Categorization of all 321 BRFSS survey columns
│   ├── special_codes.json         # Per-column sentinel mapping rules
│   ├── feature_spec.csv           # Machine-readable preprocessor specification
│   └── build_feature_spec.py      # Feature specification compiler
│
├── notebooks/                     # Interactive computational notebooks
│   ├── 01_eda.ipynb               # Exploratory data analysis & codebook insights
│   ├── 02_experiments.ipynb       # Step-by-step model progression from baselines to ensemble
│   └── 03_negative_results.ipynb  # Post-mortems of non-performing explorations
│
├── report/                        # 2-Page IEEE LaTeX Scientific Report
│   ├── report.tex                 # LaTeX report source document
│   ├── literature.bib             # Bibliography
│   ├── IEEEtran.cls / .bst        # IEEE template styling
│   └── figures/                   # High-resolution publication figures
│
├── submissions/                   # Official prediction submissions
│   ├── submission_with_leakage.csv# Track B Leaderboard Model (0.457 F1, 14,001 positives)
│   └── submission_clean_sota.csv  # Track A Clean SOTA Model (0.447 F1, 14,876 positives)
│
├── tests/                         # Unit tests (7/7 passed)
│   ├── test_preprocessing.py      # Zero-leakage pipeline assertions
│   └── test_training_coverage.py  # Model coverage & backward-compatibility assertions
│
└── docs/                          # Project coordination & technical documentation
    ├── ROADMAP.md                 # Project milestone tracker
    ├── TEAM_GUIDE.md              # Project guidelines & grading criteria
    └── TRAINING_COVERAGE_CONTRIBUTION.md # Training coverage hypothesis & screening findings
```
