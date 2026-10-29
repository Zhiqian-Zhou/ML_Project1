# EPFL CS-433 Machine Learning (Fall 2026) — Project 1
## Predicting Coronary Heart Disease (`_MICHD`) from CDC BRFSS 2015

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Pure NumPy](https://img.shields.io/badge/Dependencies-NumPy%20Only-success.svg)](https://numpy.org/)
[![Autograder](https://img.shields.io/badge/Course%20Autograder-27%2F27%20Passed-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

**Team Members:** Zhihao Chen, Zhiqian Zhou, Guanyu Chen  
**Affiliation:** School of Computer and Communication Sciences, EPFL, Switzerland  
**Track B Champion Model (Leaderboard Benchmark):** **F1 = 0.457**, Accuracy = 88.3% (14,001 positive predictions)  
**Track A Clean SOTA Model (Official Screening Tool):** **F1 = 0.447**, Accuracy = 87.7% (14,876 positive predictions, Zero-Leakage)  

---

## 1. Project Overview

The objective of this project is binary classification on the CDC Behavioral Risk Factor Surveillance System (BRFSS 2015) dataset:
* **Data Scale:** 328,135 training respondents, 109,379 test respondents, 321 survey features.
* **Target:** Predict whether an individual has coronary heart disease or myocardial infarction (`_MICHD` $\in \{-1, +1\}$).
* **Core Challenge:** Extreme class imbalance (**8.83% positive prevalence**, ~10:1 negative-to-positive ratio; 28,969 positive vs. 299,166 negative in training data). A naive all-negative classifier achieves 91.17% accuracy but an $F_1$ score of **0.0000**. All tuning, validation, and model selection are guided strictly by positive-class $F_1$.
* **Project Constraints:** Built strictly from scratch in pure Python standard library and NumPy (strictly zero dependencies on pandas, scikit-learn, or PyTorch).

---

## 2. Machine Learning Architecture & OOP Design

### Data Preprocessing & Cleaning (`preprocessing.py`, `eda/`)
* **`BRFSSCleaner`:**
  * **Sentinel Value Mapping:** Survey sentinels like `7`, `9`, `77`, `99` (don't know / refused) are mapped to `NaN`. Codes `88` (none / zero days) and `888` (no alcohol) are mapped to `0`. Imperial and metric measurements are standardized to centimeters and kilograms.
  * **Skip-Logic Imputation:** Over 45% of questions are $>50\%$ missing due to branching survey logic rather than random non-response. Eleven domain rules infer implied negatives (e.g., non-hypertensive respondents are never prescribed blood pressure medication `BPMEDS`).
  * **Informative Missingness & Feature Selection:** 61 binary uncertainty masks (`_dk`) are extracted as predictive risk features. We drop 6 constant columns, administrative IDs, columns $>95\%$ missing, and 44 collinear pairs ($|r|>0.9$), retaining 75 raw features, 2 engineered metrics, and 61 masks (138 base features).
  * **Final Feature Matrix:** Together with 9 CDC high-AUC features, 6 domain skip indicators, 4 composite clinical terms, and 6 empirical Bayes target encodings, the delivered design matrix spans **158 features** in total.
* **`TargetRiskEncoder`:**
  * Performs smoothed empirical Bayes target encoding for high-cardinality state codes (`_STATE`), eliminating categorical sparsity without high-dimensional one-hot inflation.
* **`TabularPreprocessor`:**
  * Orchestrates the complete pipeline, fits medians and standardizers strictly within training folds, and computes non-linear pairwise clinical interaction terms with zero data leakage.

### Core Baseline Methods (`implementations.py`)
All 6 course-mandated optimization methods are implemented from scratch in pure NumPy:
1. `mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)` — Linear regression via gradient descent.
2. `mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)` — Linear regression via stochastic gradient descent (batch size 1).
3. `least_squares(y, tx)` — Least squares regression via closed-form normal equations.
4. `ridge_regression(y, tx, lambda_)` — Ridge regression with L2 regularization.
5. `logistic_regression(y, tx, initial_w, max_iters, gamma)` — Logistic regression via gradient descent with numerically stable log-loss.
6. `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)` — Regularized logistic regression with L2 penalty via gradient descent.

*All 27 public autograder tests in `../grading_tests/` pass with 100% compliance.*

### Advanced Models (`models.py`)
* **`GradientBoostedTrees`:** Pure-NumPy Histogram GBDT with 64 quantile bins, depth 5, Newton second-order leaf steps with L2 shrinkage ($\lambda=5$), row subsampling (0.7), column subsampling (0.8), missing value direction routing, and validation early stopping.
* **`MLPEnsemble`:** Pure-NumPy Multi-Layer Perceptron (64 ReLU hidden units, Adam optimizer, learning rate $10^{-3}$, weight decay $10^{-3}$, batch size 512, averaged across 10 random seeds).
* **`CardiacEnsemble`:** High-level OOP coordinator blending GBDT and MLP logits ($0.50 \times \text{GBM} + 0.50 \times \text{MLP}$), post-event feature toggle (`with_leakage`), clinical prior adjustment rules, and calibrated percentile thresholding. Provides full `.fit()`, `.predict()`, `.save()`, and `.load()`.

### Validation Protocol (`cross_validation.py`)
* **`StratifiedKFold`:** Preserves the exact 8.83% positive prevalence in every fold.
* **`HoldoutSplitter`:** Stratified 80/20 train/held-out split (262,508 development rows vs. 65,627 held-out validation rows, random seed 2026).
* **`EnsembleCrossValidator` & `LinearBaselineValidator`:** Full evaluation harnesses for cross-validation and held-out validation.
* **Calibrated Percentile Thresholding:** Rather than an arbitrary 0.5 probability cutoff, models classify the top-scored fraction of individuals as positive inside the stable 12%--16% CV plateau: **Top 13.6%** for clean screening (14,876 positives, F1 = 0.447) and **Top 12.8%** for the leaky competition model (14,001 positives, F1 = 0.457).

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
| **Track A: Clean Screening** | Excludes 7 post-event questions (Top 13.6%) | $0.4368 \pm 0.0026$ | 0.4340 | **0.447** | **87.7%** |
| **Track B: Leaky Benchmark** | Exploits post-event questions (Top 12.8%) | $0.4464 \pm 0.0032$ | 0.4445 | **0.457** | **88.3%** |
| **Performance Gain ($\Delta$)** | Target leakage inflation | $+0.0096$ | $+0.0105$ | $\mathbf{+0.0100}$ | $\mathbf{+0.6\%}$ |

> **Clinical Target Leakage Analysis:** Seven survey questions reference care *after* an event: cardiac and stroke rehabilitation (`HAREHAB1`, `STREHAB1`), daily aspirin usage, and rationales (`CVDASPRN`, `RDUCHART`, `RDUCSTRK`, `RLIVPAIN`, `ASPUNSAF`). Notably, 100% of respondents reporting cardiac rehab had heart disease (615/615 on dev, 158/158 on held-out, 773/773 total). Because undiagnosed patients in pre-event screening are never prescribed cardiac rehab, these questions are reverse-causal. We strictly exclude them from our official clinical screening tool (Track A, F1 = 0.447), while documenting the leaky model as the competition benchmark (Track B, F1 = 0.457).

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

### Dataset Placement
Unzip the project dataset CSV files into `data/dataset/`:
```text
data/dataset/
├── x_train.csv           # 328,135 rows, Id + 321 features
├── y_train.csv           # Id, _MICHD (labels in {-1, 1}, ~8.8% positives)
├── x_test.csv            # 109,379 rows, Id + 321 features
└── sample_submission.csv # Expected submission format
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
The complete training pipeline is integrated directly into `run.py`. Models can be retrained from scratch using `python run.py --train` (Track B champion) or `python run.py --train --clean` (Track A clean).

```bash
# 1. Retrain Track B Champion Model from scratch and export weights
python run.py --train

# 2. Retrain Track A Clean Screening Model from scratch and export weights
python run.py --train --clean
```

> **Note on Training Runtime:** Training execution time depends on CPU thread concurrency and hardware resources (typically between ~10 to 30 minutes on standard local workstations). Committed precomputed weights in `weights/` are provided for instant, deterministic inference (~12s).  
> **Note on Floating-Point Reproducibility:** Retraining from scratch across different machines or architectures may produce minute numerical jitter (~0.2% label differences out of 109,379 rows, i.e., ~238 rows) due to CPU floating-point accumulation order in multi-seed gradient updates and quantile bin boundaries. Loading the pre-saved checkpoints in `weights/` guarantees exact 100% bit-level reproducibility of the AIcrowd competition submissions.

### Running Cross-Validation & Evaluation
```bash
# Evaluate 5-fold cross-validation (default: with leakage)
python cross_validation.py

# Evaluate Track A clean screening model across 5 folds
python cross_validation.py --clean

# Evaluate clean model on both 5-fold CV and 20% held-out test split
python cross_validation.py --clean --holdout

# Fast smoke check on 10,000 rows (~5 seconds)
python cross_validation.py --clean --n_rows 10000
```

### Interactive Computational Notebooks
All notebooks in `notebooks/` are pre-executed with pre-rendered interactive figures and markdown tables:
* [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb): 321-column survey analysis, codebook sentinel mapping, skip logic reconstruction, informative missingness discovery, and target leakage detection.
* [`notebooks/02_experiments.ipynb`](notebooks/02_experiments.ipynb): The complete computational laboratory:
  * Live training and benchmark of all 6 baseline algorithms (`implementations.py`).
  * Live threshold tuning ($\tau \in [0.05, 0.50]$) illustrating the jump from 0.21 to 0.41 F1.
  * Live stratified 3-fold cross validation with fold-isolated preprocessing.
  * Live advanced model fitting: Histogram GBDT and Neural Network MLP ensemble.
  * Live Clean vs. Leaky dual-track comparison demonstrating the leakage boost.
  * Empirical grounding of Report Table I, Figure 1, Table II, feature importances, and clinical rules.
* [`notebooks/03_negative_results.ipynb`](notebooks/03_negative_results.ipynb): Empirical post-mortems of rejected hypotheses (SMOTE oversampling, inverse-frequency class weighting, deep architectures, stacking meta-learners).

### Test Suite Execution
```bash
# Run project unit tests (11/11 passed in 0.22s)
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
│       ├── leakage_ablation.csv   # Cumulative leakage addition ablation
│       ├── rejected.csv           # Post-mortems of rejected hypotheses (Report Section VI)
│       ├── baselines_full.csv     # Extended baseline sweep logs
│       ├── final_cv_folds.csv     # Fold-by-fold validation logs
│       └── training_coverage.json # Model training coverage registry
│
├── data/                          # Dataset folder (raw CSVs, git-ignored)
│   └── README.md                  # Instructions for placing BRFSS survey CSVs
│
├── eda/                           # Domain codebooks & feature specifications
│   ├── column_decisions.csv       # Categorization of all 321 BRFSS survey columns
│   ├── special_codes.json         # Per-column sentinel mapping rules
│   ├── feature_spec.csv           # Machine-readable preprocessor specification
│   └── build_feature_spec.py      # Spec builder utility
│
├── notebooks/                     # Interactive computational notebooks (pre-rendered)
│   ├── 01_eda.ipynb               # Exploratory data analysis, skip patterns & leakage discovery
│   ├── 02_experiments.ipynb       # Live training suite (all 6 baselines, GBDT, MLP, CV) & report benchmarks
│   └── 03_negative_results.ipynb  # Rigorous post-mortems of rejected hypotheses
│
├── report/                        # 2-Page IEEE LaTeX Scientific Report
│   ├── report.tex                 # LaTeX report source document
│   ├── report.pdf                 # Compiled 2-page publication PDF
│   ├── literature.bib             # Bibliography
│   ├── IEEEtran.cls / .bst        # IEEE conference template styling
│   └── figures/                   # High-resolution publication figures (f1_vs_share.pdf, etc.)
│
├── submissions/                   # Verified competition predictions
│   ├── submission_with_leakage.csv# Track B Champion Model (0.457 F1, 14,001 positives)
│   └── submission_clean_sota.csv  # Track A Clean SOTA Model (0.447 F1, 14,876 positives)
│
└── tests/                         # Unit tests (11/11 passed in 0.22s)
    ├── test_preprocessing.py      # Zero-leakage pipeline assertions
    ├── test_preprocessing_contracts.py # Imputation & feature contract assertions
    └── test_training_coverage.py  # Model coverage & backward-compatibility assertions
```

---

## 6. Course Requirements Compliance

| Requirement | Implementation Details | Status |
| :--- | :--- | :---: |
| **6 Mandatory Baselines** | All 6 methods implemented in `implementations.py` (`mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, `reg_logistic_regression`). | Passed (27/27) |
| **Execution Script** | `run.py` provides deterministic inference (~12s) and full training from scratch via pure NumPy checkpoints. | Verified |
| **Dependencies** | Strict pure-NumPy and Python standard library (no pandas, no scikit-learn, no torch). | Compliant |
| **Code Style** | Formatted with `black`, comprehensive docstrings, modular OOP design. | Compliant |
| **Scientific Report** | 2-page IEEE conference document in `report/report.pdf` with full empirical grounding across Table I and Table II. | Complete |
