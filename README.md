# EPFL CS-433 Machine Learning (Fall 2026) — Project 1
## Predicting Coronary Heart Disease (`_MICHD`) from CDC BRFSS 2015

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Pure NumPy](https://img.shields.io/badge/Dependencies-NumPy%20Only-success.svg)](https://numpy.org/)
[![Autograder](https://img.shields.io/badge/Course%20Autograder-27%2F27%20Passed-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

**Team Members:** Zhihao Chen, Zhiqian Zhou, Guanyu Chen  
**Track B Champion Model (Leaderboard Benchmark):** **F1 = 0.457**, Accuracy = 88.3% (14,001 positive predictions)  
**Track A Clean SOTA Model (Official Screening Tool):** **F1 = 0.447**, Accuracy = 87.7% (14,876 positive predictions, Zero-Leakage)

---

## 1. Project Overview

The goal of this project is binary classification on the CDC Behavioral Risk Factor Surveillance System (BRFSS 2015) dataset:
* **Data Scale:** 328,135 training respondents, 109,379 test respondents, 321 survey features.
* **Target:** Predict whether an individual has coronary heart disease or myocardial infarction (`_MICHD` $\in \{-1, +1\}$).
* **Challenge:** Severe class imbalance (**8.83% positive prevalence**, ~10:1 ratio). A naive all-negative classifier scores 91.17% accuracy but an $F_1$ score of **0.0000**. All tuning and model selection are guided strictly by positive-class $F_1$.
* **Constraints:** Built strictly from scratch in pure Python standard library and NumPy (no pandas, scikit-learn, or torch).

---

## 2. Machine Learning Pipeline

### Data Preprocessing & Cleaning (`preprocessing.py`, `eda/`)
* **Sentinel Value Mapping:** Survey sentinels like `7`, `9`, `77`, `99` (don't know / refused) are converted to `NaN`. Codes `88` (none / zero days) and `888` (no alcohol) are mapped to `0`. Imperial and metric measurements are standardized to centimeters and kilograms.
* **Skip-Logic Imputation:** Over 45% of questions are $>50\%$ missing due to branching logic rather than random non-response. Eleven domain rules infer implied negatives (e.g., non-hypertensive respondents never take blood pressure medication).
* **Informative Missingness & Feature Selection:** 61 binary uncertainty masks (`_dk`) are extracted as predictive features. We drop 6 constant columns, administrative IDs, columns $>95\%$ missing, and 44 collinear pairs ($|r|>0.9$), retaining 75 raw features, 2 engineered metrics, and 61 masks (138 total).
* **Clinical Interactions:** 10 pairwise non-linear clinical terms (e.g., Age $\times$ General Health, Age $\times$ Blood Pressure) and empirical Bayes smoothed target risk encodings.
* **Zero Data Leakage:** All imputation medians, one-hot categories, and scaling statistics are estimated strictly inside training folds.

### Core Baseline Methods (`implementations.py`)
We implement all 6 course-mandated optimization methods from scratch:
1. `mean_squared_error_gd` — Linear regression via gradient descent.
2. `mean_squared_error_sgd` — Linear regression via stochastic gradient descent (batch size 1).
3. `least_squares` — Least squares regression via normal equations.
4. `ridge_regression` — L2-regularized least squares regression.
5. `logistic_regression` — Logistic regression via gradient descent.
6. `reg_logistic_regression` — Regularized logistic regression via gradient descent.

*All 27 public autograder tests in `../grading_tests/` pass with 100% compliance.*

### Advanced Models (`models.py`)
* **Histogram Gradient Boosted Decision Trees (`models.py`):** Pure-NumPy Histogram GBDT with 64 quantile bins, depth 5, Newton second-order leaf steps ($\lambda=5$), row/col subsampling, and validation early stopping.
* **Multi-Layer Perceptron (`models.py`):** Pure-NumPy Neural Network with 64 ReLU hidden units, Adam optimizer, weight decay, and multi-seed logit averaging (10 random seeds).
* **Ensemble (`models.py`):** $0.50 \times \text{GBM} + 0.50 \times \text{MLP}$ blended logits with calibrated decision cutoff.

### Validation Protocol (`cross_validation.py`)
* **80% Development Split** (262,508 rows): Stratified 5-Fold Cross-Validation preserves the exact 8.83% positive prevalence in every fold.
* **20% Held-Out Split** (65,627 rows, seed 2026): Completely isolated for final validation prior to AIcrowd submission.
* **Calibrated Decision Rule:** Rather than an arbitrary 0.5 probability cutoff, models classify the top-scored fraction of individuals as positive (Top 14% for clean screening, Top 13% for leaky competition model).

---

## 3. Experimental Results (Grounding Report Table I & Table II)

### Table I: Model Comparison across Milestones (5-Fold Dev CV vs. 20% Held-Out)

| Milestone & Configuration | CV F1 Score | Held-Out F1 | Key Finding / Rationale |
| :--- | :---: | :---: | :--- |
| **Always $-1$ (Majority Class)** | 0.0000 | 0.0000 | High accuracy (91.2%), but zero predictive utility |
| **Logistic Regression (0.5 Cutoff)** | $0.2302 \pm 0.0047$ | — | Class imbalance pulls probabilities to zero; misses sick patients |
| **Least Squares / Ridge / MSE GD (Top 14%)** | $0.4216 \pm 0.0034$ | — | Normal equations and GD converge to identical linear solutions |
| **Linear Regression (SGD, unconverged)** | $0.3853 \pm 0.0085$ | — | Single-sample stochastic noise oscillates under 10:1 class skew |
| **Logistic Regression (GD, Top 14%)** | $0.4302 \pm 0.0047$ | — | Cross-entropy loss improves log-odds calibration |
| **Reg. Logistic Regression (GD, Top 14%)** | $0.4300 \pm 0.0044$ | 0.4277 | L2 penalty stabilizes weights; sets linear benchmark ceiling |
| **Histogram GBM (3 seeds, Top 14%)** | $0.4354 \pm 0.0032$ | 0.4343 | Decision trees capture non-linear thresholds and interactions |
| **Neural Network MLP (10 seeds, Top 14%)** | $0.4334 \pm 0.0026$ | 0.4310 | 64 ReLU units provide smooth continuous decision boundaries |
| **Track A Clean Ensemble (GBM + MLP)** | $\mathbf{0.4368 \pm 0.0026}$ | $\mathbf{0.4340}$ | **Official SOTA Screening Tool (AIcrowd F1 = 0.447, Acc 87.7%)** |

### Table II: Clean Screening Model vs. Leaky Competition Variant

| Variant | Strategy | 5-Fold CV F1 | Held-Out F1 | AIcrowd F1 | AIcrowd Acc. |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Track A: Clean Screening** | Excludes 7 post-event questions (Top 14%) | $0.4368 \pm 0.0026$ | 0.4340 | **0.447** | **87.7%** |
| **Track B: Leaky Benchmark** | Exploits post-event questions (Top 13%) | $0.4464 \pm 0.0032$ | 0.4445 | **0.457** | **88.3%** |
| **Performance Gain ($\Delta$)** | Target leakage inflation | $+0.0096$ | $+0.0105$ | $\mathbf{+0.0100}$ | $\mathbf{+0.6\%}$ |

> **Clinical Governance Principle:** Seven survey questions reference care *after* an event (cardiac/stroke rehab `HAREHAB1`, `STREHAB1`, secondary prevention aspirin `CVDASPRN`, `RDUCHART`, `RDUCSTRK`, `RLIVPAIN`, `ASPUNSAF`). Notably, 100% of respondents reporting cardiac rehab had heart disease (773/773 total). Because undiagnosed patients in pre-event screening are never prescribed cardiac rehab, these questions are reverse-causal. We strictly exclude them from our official model (Track A), while benchmarking the leaky model as Track B.

*For detailed post-mortems on rejected hypotheses (SMOTE, class weighting, deep networks, stacking), see [`notebooks/03_negative_results.ipynb`](notebooks/03_negative_results.ipynb).*

---

## 4. Quickstart & Reproducibility

### Environment Setup
Only standard Python, NumPy, and Matplotlib are required:
```bash
# Clone the repository
git clone https://github.com/Zhiqian-Zhou/ML_Project1.git
cd ML_Project1

# Setup virtual environment (Python 3.10)
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib pytest black
```

### Reproducing Competition Predictions (Fast Inference, ~12s)
In accordance with course project guidelines:
> *"If training takes a long time, this run.py can include only inference (loading pretrained weights and generating predictions), but then please also explain where to find the code for training these weights."*

The runner `run.py` loads pure-NumPy checkpoints from `weights/` and executes fast, deterministic inference in **~12 seconds**:

```bash
# 1. Reproduce Track B Champion Submission (Default: 0.457 F1, 14,001 positives)
python run.py

# 2. Reproduce Track A Clean Screening Submission (Official SOTA: 0.447 F1, 14,876 positives)
python run.py --clean
```

### Retraining Models from Scratch
The complete training pipeline is integrated directly into `run.py`. Utilizing process-pool parallelization across seeds, training finishes in ~4 minutes:

```bash
# 1. Retrain Track B Champion Model from scratch and export weights
python run.py --train

# 2. Retrain Track A Clean Screening Model from scratch and export weights
python run.py --train --clean
```

### Running Validation, Notebooks & Test Suite
```bash
# Run 5-fold cross-validation suite
python cross_validation.py

# Run project unit tests (11/11 passed)
pytest tests/

# Run course autograder compliance suite (27/27 passed)
pytest ../grading_tests --github_link=.
```

---

## 5. Repository Structure

```text
ML_Project1/
├── README.md                      # Executive overview, pipeline description & quickstart
├── implementations.py             # 6 course baseline methods (GD, SGD, LS, Ridge, Logistic, Reg-Logistic)
├── run.py                         # Unified runner for reproducible inference (~12s) and retraining (~4m)
├── models.py                      # Pure-NumPy OOP models (GradientBoostedTrees, MLPEnsemble, CardiacEnsemble)
├── cross_validation.py            # Stratified K-fold CV and 80/20 holdout evaluation harness
├── preprocessing.py               # OOP feature cleaning (BRFSSCleaner, TargetRiskEncoder, TabularPreprocessor)
├── metrics.py                     # Accuracy, precision, recall, F1-score & confusion matrix
├── helpers.py                     # CSV streaming loaders & submission export utilities
│
├── weights/                       # Pretrained model checkpoints (pure-NumPy .npz archives, allow_pickle=False)
│   ├── clean_model.npz            # Track A Clean SOTA weights (3 GBM seeds + 10 MLP nets)
│   └── leaky_model.npz            # Track B Leaky Champion weights
│
├── experiments/                   # Publication-grade experiment utilities and grounded logs
│   ├── plots.py                   # Zero-dependency table & chart rendering helper
│   └── results/                   # Verified experimental CSV logs backing all report tables
│       ├── model_comparison.csv   # Full 8-model milestone benchmark (Report Table I)
│       ├── final_summary.csv      # Dual-track comparison (Report Table II)
│       ├── f1_vs_rate.csv         # Calibrated cutoff curves (Report Figure 1)
│       ├── feature_importance.csv # Permutation AUC drops (Report Section IV)
│       ├── rules.csv              # High-risk clinical rules and lifts (Report Section IV)
│       ├── leaky_columns.csv      # Single-column target leakage ablation (Report Section V)
│       └── rejected.csv           # Post-mortems of rejected hypotheses (Report Section VI)
│
├── data/                          # Dataset folder (symlinked raw CSVs, git-ignored)
├── eda/                           # Domain codebooks & feature specifications
│   ├── column_decisions.csv       # Categorization of all 321 BRFSS survey columns
│   ├── special_codes.json         # Per-column sentinel mapping rules
│   └── feature_spec.csv           # Machine-readable preprocessor specification
│
├── notebooks/                     # Interactive computational notebooks
│   ├── 01_eda.ipynb               # Exploratory data analysis, skip patterns & leakage discovery
│   ├── 02_experiments.ipynb       # Live training suite (all 6 baselines, GBDT, MLP, CV) & report benchmarks
│   └── 03_negative_results.ipynb  # Rigorous post-mortems of rejected hypotheses
│
├── report/                        # 2-Page IEEE LaTeX Scientific Report
│   ├── report.tex                 # LaTeX report source document
│   ├── report.pdf                 # Compiled 2-page publication PDF
│   ├── literature.bib             # Bibliography
│   ├── IEEEtran.cls / .bst        # IEEE conference template styling
│   └── figures/                   # High-resolution figures (f1_vs_share.pdf, etc.)
│
├── submissions/                   # Verified competition predictions
│   ├── submission_with_leakage.csv# Track B Champion Model (0.457 F1, 14,001 positives)
│   └── submission_clean_sota.csv  # Track A Clean SOTA Model (0.447 F1, 14,876 positives)
│
└── tests/                         # Unit tests (11/11 passed in 0.13s)
    ├── test_preprocessing.py      # Zero-leakage pipeline assertions
    ├── test_preprocessing_contracts.py # Imputation & feature contract assertions
    └── test_training_coverage.py  # Model coverage & backward-compatibility assertions
```
