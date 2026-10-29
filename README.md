# EPFL CS-433 Machine Learning (Fall 2026) — Project 1
## Predicting Coronary Heart Disease (`_MICHD`) from CDC BRFSS 2015

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Pure NumPy](https://img.shields.io/badge/Dependencies-NumPy%20Only-success.svg)](https://numpy.org/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

**Team Members:** Zhihao Chen, Zhiqian Zhou, Guanyu Chen  
**Submission baseline:** the leakage-enabled pipeline associated with `submissions/submission_with_leakage.csv`. The default command uses its archived checkpoint and writes `submission.csv` in the repository root, leaving both archived CSVs untouched. The archived leaky checkpoint's current output differs from that saved CSV on 458 predictions. Leaderboard records tying a score to a saved CSV are absent.

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
* **Feature Construction:** Four composite clinical indicators and continuous empirical Bayes risk encodings. The final MLP disables the optional ten pairwise interactions; both saved checkpoints contain zero fitted pairs.
* **Training-fold isolation:** Newly trained models fit composite normalization, scalers, imputation statistics, and risk priors on training rows. Training category-risk and GBM state-risk features are encoded out of fold. Saved version 1 checkpoints predate these fixes; their reported scores require revalidation for the corrected pipeline.

### Core Baseline Methods (`implementations.py`)
We implement the 6 course-mandated optimization methods from scratch:
1. `mean_squared_error_gd` — Linear regression via gradient descent.
2. `mean_squared_error_sgd` — Linear regression via stochastic gradient descent (batch size 1).
3. `least_squares` — Least squares regression via normal equations.
4. `ridge_regression` — L2-regularized least squares regression.
5. `logistic_regression` — Logistic regression via gradient descent.
6. `reg_logistic_regression` — Regularized logistic regression via gradient descent.

The external public autograder tests are not included in this repository; their pass count is unverified here.

### Advanced Models (`gbm.py`, `mlp.py`)
* **Gradient Boosted Decision Trees (`gbm.py`):** Pure-NumPy Histogram Gradient Boosting with 64 bins, depth 5, and Newton leaf updates. Bagged over 3 random seeds.
* **Multi-Layer Perceptron (`mlp.py`):** Pure-NumPy Neural Network with 64 ReLU hidden units, Adam optimizer, and weight decay, averaged over 10 random initialization seeds.
* **Ensemble:** Score blend of $0.50 \times \text{GBM} + 0.50 \times \text{MLP}$ with calibrated decision threshold.

### Validation Protocol (`cross_validation.py`, `cv_final.py`)
* Stratified 5-Fold Cross-Validation preserves the exact 8.83% positive ratio in every fold.
* Validation uses the archived fixed top-rate settings: 12.8% for the leakage-enabled model and 13.6% for the clean variant. Each held-out fold is thresholded at its own score quantile; the final CV command does not retune this rate inside a fold.

---

## 3. Progressive Experimental Results

The table below records historical experiment claims. The corrected leakage-enabled pipeline must be evaluated separately, and the stated leaderboard scores are not backed by submission records in this repository.

| Step | Model & Configuration | Feature Dims ($D$) | 5-Fold CV F1 | 5-Fold CV Acc | Key Impact / What Changed? |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline 0** | Naive Majority Predictor | — | 0.0000 | 91.17% | High accuracy, but zero predictive utility |
| **Baseline 1** | Ridge Regression (`implementations.py`) | 198 | 0.1198 | 91.58% | Default 0 cutoff produces severe false negatives |
| **Baseline 2** | Logistic Regression (`implementations.py`) | 198 | 0.2311 | 91.65% | Default 0.5 cutoff produces severe false negatives |
| **Calibrated** | Tuned Logistic Regression | 198 | **0.4277** | 87.22% | **+85% jump via threshold calibration ($\tau^*=0.218$)** |
| **Features** | Cleaned Survey Data + Cross-Terms | 158 | **0.4311** | 87.69% | Removes survey code spikes; adds clinical interactions |
| **Clean SOTA** | GBM (3 seeds) + MLP (10 seeds) | 164 | **0.4372** | **87.80%** | Historical experiment; revalidation needed after training fixes |
| **Final Model**| GBM + MLP + Post-Event Survey Signals | 171 | **0.4553** | **88.60%** | Historical experiment; leaderboard attribution unverified |

### Corrected leakage-enabled validation (checkpoint version 2.1)

`python cv_final.py --with_leakage --holdout --k_fold 5` trained five folds on the 262,508 development rows, then fitted a separate model on all development rows and scored the 65,627-row holdout. Each split used a 12.8% positive prediction rate. The five fold F1 scores were 0.4477, 0.4418, 0.4477, 0.4425, and 0.4494: mean **0.4458 ± 0.0030**. The held-out F1 was **0.4439** (precision 0.3751, recall 0.5437, accuracy 0.8797). Detailed split results are in [`experiments/results/corrected_leaky_validation.csv`](experiments/results/corrected_leaky_validation.csv). These local results measure the corrected model; they do not verify an AIcrowd score for the archived submission.

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

### Generating Competition Predictions (Fast Inference, about 20 seconds)
In accordance with course project rules:
> *"If training takes a long time, this run.py can include only inference (loading pretrained weights and generating predictions), but then please also explain where to find the code for training these weights."*

The script `run.py` loads pure-NumPy pretrained checkpoints from `weights/` and executes deterministic inference in about **20 seconds** in the measured environment:

```bash
# 1. Run the archived leakage-enabled checkpoint (writes submission.csv)
python run.py

# 2. Reproduce the saved clean screening CSV, as a separate variant
python run.py --clean
```

### Retraining Models from Scratch
The corrected training path is provided in `train.py`. It fits training-fold composite statistics and out-of-fold category and state risk encodings, so retrained checkpoints differ from the supplied version 1 checkpoints. Validate any new checkpoint before using it as a competition submission:

The final corrected leakage-enabled fit produced `weights/leaky_model_corrected.npz` (version 2.1). It selected 412, 486, and 422 GBM trees and took 517 seconds in the measured environment. Inference from this checkpoint reproduces its own training-time CSV byte for byte; its predictions differ from the archived leaky CSV on 622 rows. This is separate from the archived checkpoint's 458-row reproduction mismatch.

```bash
# Train the corrected leakage-enabled model to a new checkpoint
python train.py --with_leakage

# Alternatively, pass --train to run.py:
python run.py --train
```

### Running Validation & Tests
```bash
# Run leakage-enabled 5-fold cross-validation and a separate held-out check
python cv_final.py --with_leakage --holdout

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
├── run.py                         # Inference submission generator using pretrained weights
├── train.py                       # Complete training harness from scratch with weight export
├── model_io.py                    # Safe, portable pure-NumPy model serialization (allow_pickle=False)
├── cv_final.py                    # 5-fold cross-validation runner for the final ensemble
├── cross_validation.py            # Stratified K-fold split & threshold calibration harness
├── gbm.py                         # Pure-NumPy Histogram Gradient Boosted Decision Trees
├── mlp.py                         # Pure-NumPy Multi-Seed Multi-Layer Perceptron (Adam)
├── preprocessing.py               # BRFSS survey cleaning, imputation & feature pipeline
├── metrics.py                     # Accuracy, precision, recall, F1-score & confusion matrix
├── helpers.py                     # CSV streaming loaders & submission export utilities
├── run_public_tests.sh            # Public autograder test runner script
│
├── weights/                       # Pretrained model checkpoints (pure-NumPy .npz archives)
│   ├── clean_model.npz            # Track A Clean SOTA weights (3 GBM seeds + 10 MLP nets)
│   ├── leaky_model.npz            # Archived leakage-enabled weights
│   └── leaky_model_corrected.npz  # Corrected leakage-enabled training run
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
│   ├── submission_with_leakage.csv# Saved post-event leakage variant
│   └── submission_clean_sota.csv  # Saved clean screening variant
│
├── tests/                         # Repository regression tests
│   ├── test_preprocessing.py      # Cleaning assertions
│   ├── test_final_pipeline.py     # Fold isolation regression checks
│   └── test_training_coverage.py  # Model coverage assertions
│
└── docs/                          # Project coordination & technical documentation
    ├── ROADMAP.md                 # Project milestone tracker
    ├── TEAM_GUIDE.md              # Project guidelines & grading criteria
    └── TRAINING_COVERAGE_CONTRIBUTION.md # Training coverage hypothesis & screening findings
```
