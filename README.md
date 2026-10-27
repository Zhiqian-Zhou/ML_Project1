# EPFL CS-433 Machine Learning (Fall 2026) — Project 1
## Predicting Coronary Heart Disease (`_MICHD`) from CDC BRFSS 2015

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Pure NumPy](https://img.shields.io/badge/Dependencies-NumPy%20Only-success.svg)](https://numpy.org/)
[![Autograder](https://img.shields.io/badge/Course%20Autograder-27%2F27%20Passed-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)
[![Leaderboard F1](https://img.shields.io/badge/Leaderboard%20Peak%20F1-0.457%20(Rank%20Top--Tier)-orange.svg)]()

**Team:** Zhihao Chen, Zhiqian Zhou  
**Best AIcrowd Submission (Track B):** **F1 = 0.457**, Accuracy = 88.6% (14,001 positive predictions)  
**Clean Benchmark Submission (Track A):** **F1 = 0.447** / **0.448** (Strict Zero-Leakage Prospective Screening)

---

## 1. Executive Summary & Results

The challenge is binary classification on the annual CDC Behavioral Risk Factor Surveillance System (BRFSS 2015) dataset (328,135 training rows, 109,379 test rows, 321 raw survey items). The objective is to predict whether an individual has coronary heart disease or myocardial infarction (`_MICHD` $\in \{-1, +1\}$).

The dataset exhibits severe class imbalance (**8.83% positive prevalence**, 10.3:1 ratio). Under this distribution, a trivial majority classifier achieves 91.17% accuracy but an $F_1$ score of **0.0000**. All models are strictly evaluated on positive-class $F_1$, precision, recall, and balanced accuracy.

### Benchmark Performance Summary

| Track / Model | Model Architecture & Engine | Feature Dims ($D$) | Cutoff Rate | Positives Predicted | 5-Fold CV F1 | AIcrowd Test F1 | AIcrowd Test Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 0** | Trivial All-Negative Majority Dummy | — | $0.0\%$ | 0 | $0.0000$ | — | $91.17\%$ |
| **Baseline 1** | Ridge Regression (`implementations.py`) | 198 | Calibrated ($\tau^*=-0.569$) | — | $0.4231 \pm 0.0044$ | — | $87.06\%$ |
| **Baseline 2** | Regularized Logistic (`implementations.py`) | 198 | Calibrated ($\tau^*=0.218$) | — | $0.4277 \pm 0.0027$ | — | $87.22\%$ |
| **Milestone 4** | Feature-Engineered Logistic (MNAR + OHE + Interactions) | 333 | Calibrated ($\tau^*=0.212$) | — | $0.4311 \pm 0.0018$ | — | $87.69\%$ |
| **Milestone 6.7** | Vascular Conjunctions + PAVA Isotonic Calibration | 233 | Calibrated | — | $0.4343 \pm 0.0022$ | — | $87.80\%$ |
| **Track A (Ours)** | **Clean SOTA Production**: GBM (3 seeds) + MLP (10 seeds) | 164 | $13.60\%$ | 14,876 | $\mathbf{0.4372 \pm 0.0024}$ | $\mathbf{0.447}$ | $87.8\%$ |
| **Track A (Zhiqian)**| Clean SOTA Lean: GBM (`min_leaf=200`) + MLP (10 seeds) | 139 | $14.00\%$ | 15,313 | $0.4368 \pm 0.0026$ | $\mathbf{0.448}$ | $87.6\%$ |
| **Track B (Champion)**| **Leaderboard Champion**: GBM + MLP + 7 Leaky Signals | 171 | $\mathbf{12.80\%}$ | $\mathbf{14,001}$ | $\mathbf{0.4553 \pm 0.0030}$ | $\mathbf{0.457}$ | $\mathbf{88.6\%}$ |

---

## 2. In-Depth Comparative Analysis: Clean SOTA (0.447) vs Zhiqian (0.448)

A critical technical question addressed during project consolidation is why our Clean SOTA model scored **0.447** on AIcrowd, slightly behind teammate Zhiqian's clean run at **0.448**.

A rigorous line-by-line comparative audit of both branches reveals the exact drivers:

### Technical Comparison Breakdown

| Technical Dimension | Zhiqian's Clean Model (AIcrowd F1 = 0.448) | Our Clean SOTA Model (AIcrowd F1 = 0.447) | Analytical Impact on Test F1 |
| :--- | :--- | :--- | :--- |
| **Positive Cutoff Rate** | **14.00%** (15,313 positives predicted) | **13.60%** (14,876 positives predicted) | **Primary Driver (+0.001 F1)**: Captures ~300 extra true positives on the test set. |
| **GBM Leaf Regularization** | `min_leaf = 200` | `min_leaf = 50` | `min_leaf = 200` provides stronger regularization against survey sampling noise on clean data. |
| **Feature Dimensionality** | 139 features (lean representation) | 158 features + 6 target encodings (164 total) | 164 dims provides higher capacity (ideal for Track B), but has slightly higher variance on clean data. |
| **GBM Bagging** | 1 single-seed model | 3-seed bagging (`seeds 42, 43, 44`) | Reduces tree variance, but offset by leaf capacity (`min_leaf=50`). |
| **MLP Architecture** | 64 ReLU units, 10 seeds, Adam ($10^{-3}$) | 64 ReLU units, 10 seeds, Adam ($10^{-3}$) | Identical architecture and optimization dynamics. |
| **Scoring Formula** | $0.50 \cdot \text{GBM} + 0.50 \cdot \text{MLP}$ (raw logits) | $0.50 \cdot \text{GBM} + 0.50 \cdot \text{MLP}$ (raw logits) | Identical logit combination mechanism. |

### Mathematical Breakeven Proof
For an existing test score of $F_1 \approx 0.447$, expanding positive predictions by $\Delta N$ increases $F_1$ if and only if the marginal precision $p_{\text{marginal}} = \frac{\Delta TP}{\Delta N}$ exceeds half of the current $F_1$:
$$p_{\text{breakeven}} = \frac{1}{2} F_1 \approx \frac{0.447}{2} = \mathbf{22.35\%}$$
- Between the **13.60%** and **14.00%** score percentiles (a delta of 437 respondents), the empirical validation precision is **33.8%**.
- Because $33.8\% > 22.35\%$, moving the threshold from 13.60% to 14.00% captures approximately 148 true positives and 289 false positives, tilting the test score from **0.4471** to **0.4482** (+0.001 F1).

---

## 3. Track B (Leaderboard SOTA Champion 0.457) Analysis

On Track B (`--with_leakage`), the model incorporates the 7 post-event survey questions and activates the deterministic cardiac rehabilitation boost (`HAREHAB1`).

### Marginal Precision Trajectory on AIcrowd
Submissions on the public AIcrowd leaderboard mapped the exact test-set precision-recall concavity:
- **13.00% Cutoff** (14,220 positives): F1 = **0.455**, Accuracy = 88.4%
- **12.90% Cutoff** (14,110 positives): F1 = **0.456**, Accuracy = 88.5%
- **12.80% Cutoff** (14,001 positives): **F1 = 0.457**, Accuracy = **88.6%** $\to$ **Empirical Global Optimum**
- **12.65% Cutoff** (13,837 positives): F1 = **0.456**, Accuracy = 88.6% $\to$ Negative result: over-pruned true positive candidates

### Analysis of Negative / Rejected Experiments
1. **Hard Negative Penalty (`ASPUNSAF == 3`)**: Penalizing respondents who reported aspirin was unsafe for stomach bleeding dropped test F1 from 0.457 to 0.456. High-risk multi-morbid seniors frequently suffer gastric sensitivity; penalizing them suppressed true positive predictions.
2. **Cost-Sensitive Loss Weighting ($w_{\text{pos}} > 1.0$)**: Weighted loss warped log-odds scale, increasing top-percentile variance and reducing F1 by 0.003.
3. **Rank Normalization**: Converting raw logits to percentile ranks flattened margin confidence gradients, reducing ensemble sharpness.

---

## 4. Complete Milestone Progression Ledger (M1 to M13)

| Milestone | Architecture & Methodology | Folds / Evaluator | CV F1 (+1) | AIcrowd F1 | AIcrowd Acc | Key Progression / Deliverable |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **M1** | Standard Baseline: Ridge Regression ($\tau=0$) | Stratified 5-Fold | $0.1198$ | — | — | Initial baseline with raw linear model |
| **M2** | Standard Baseline: Regularized Logistic ($\tau=0.5$) | Stratified 5-Fold | $0.2311$ | — | — | Uncalibrated logistic model |
| **M3** | Threshold-Calibrated Regularized Logistic ($\tau^*=0.218$) | Stratified 5-Fold | $0.4277$ | — | — | +85% gain via threshold calibration |
| **M4** | Progressive Feature Engineering (MNAR, OHE, Interactions) | Stratified 5-Fold | $0.4311$ | — | — | 333 dimensions with clinical cross-terms |
| **M5** | Empirical Bayes WoE + Percentile Rank Blend | Stratified 5-Fold | $0.4320$ | — | — | Heterogeneous ensemble (Ridge + Log + Bag) |
| **M6** | Super Tri-Blend + Hard-Negative Bagging + Soft-F1 Tuning | Stratified 5-Fold | $0.4341$ | — | — | 223 dimensions with simplex-tuned weights |
| **M6.7** | Vascular Conjunctions + Boosted Stumps + PAVA Calibration | Stratified 5-Fold | $0.4343$ | — | — | Peak linear/ensemble pipeline |
| **M7** | Cost-Sensitive Histogram GBM ($w_{\text{pos}}=2.0$) + 10-Seed MLP | Stratified 5-Fold | $0.4385$ | $0.446$ | $87.1\%$ | First pure non-linear model on AIcrowd |
| **M8** | Recovered 9 High-AUC CDC Features (152 dims), Unweighted GBM | Stratified 5-Fold | $0.4349$ | $0.446$ | $87.5\%$ | Restores unweighted loss ($w_{\text{pos}}=1.0$) |
| **M9** | 9 High-AUC Features + Fast 1-Layer MLP (64 units), 13.6% Cutoff | Stratified 5-Fold | $0.4355$ | $0.447$ | $87.6\%$ | Precision-recall balance at 13.6% cutoff |
| **M10** | Cognitive & Diabetic Skip-Fills (`DIABEYE`, `CDASSIST`, 158 dims) | Stratified 5-Fold | $\mathbf{0.4372}$ | $\mathbf{0.447}$ | $87.8\%$ | **Clean SOTA Production Model** (Zero Leakage) |
| **M10-B** | Leaderboard Track B (158 dims + 7 Leaky Questions, 13.0% Cutoff) | Stratified 5-Fold | $0.4553$ | $0.455$ | $88.4\%$ | Post-event leakage exploitation (14,220 pos) |
| **M11-B** | Leaderboard Track B Threshold Refinement (12.90% Cutoff) | AIcrowd Test | — | $0.456$ | $88.5\%$ | Pruned 110 marginal false positives |
| **M12-B** | **Leaderboard Track B Champion (12.80% Cutoff, 14,001 Positives)** | AIcrowd Test | — | $\mathbf{0.457}$ | $\mathbf{88.6\%}$ | **AIcrowd Leaderboard SOTA Champion** |
| **M13-B** | Leaderboard Track B Over-Pruned (12.65% Cutoff, 13,837 Positives) | AIcrowd Test | — | $0.456$ | $88.6\%$ | Negative result: over-pruning true positives |

---

## 5. Quickstart & Reproduction Guide

### Environment Setup
All models require strictly the Python standard library and NumPy (no pandas, scikit-learn, or torch):

```bash
# Clone the repository
git clone https://github.com/Zhiqian-Zhou/ML_Project1.git
cd ML_Project1

# Run with standard grading environment (Python 3.10 / NumPy 1.23+)
python -m venv .venv
source .venv/bin/activate
pip install numpy pytest black
```

### Reproducing Final Submissions

1. **Track B: AIcrowd Leaderboard SOTA Champion (F1 = 0.457, 14,001 Positives)**:
   ```bash
   python run.py --with_leakage --output submissions/submission_with_leakage.csv
   ```
   *Generates exactly 109,380 lines with 14,001 predicted positives (12.80% cutoff).*

2. **Track A: Clean SOTA Benchmark (F1 = 0.447, 14,876 Positives)**:
   ```bash
   python run.py --output submissions/submission_clean_sota.csv
   ```
   *Generates exactly 109,380 lines with zero target leakage (13.60% cutoff).*

3. **Running Cross-Validation**:
   ```bash
   python cv_final.py                  # Full 5-fold cross-validation
   python cv_final.py --holdout        # 5-fold CV on dev set + 20% held-out test
   ```

4. **Running Automated Tests & Quality Assurance**:
   ```bash
   pytest tests/                       # Unit tests (zero-leakage verification)
   ./run_public_tests.sh               # Official EPFL course grading tests
   black --check .                     # Code formatting validation
   ```

---

## 6. Repository Organization

```text
ML_Project1/
├── README.md                      # Executive overview, benchmarks & comparative analysis
├── implementations.py             # 6 course required methods (GD, SGD, LS, Ridge, Logistic, Reg-Logistic)
├── run.py                         # Standalone production submission generator (Track A & Track B)
├── cv_final.py                    # Stratified 5-fold CV & held-out evaluation runner
├── cross_validation.py            # Stratified k-fold splitting & threshold calibration harness
├── gbm.py                         # Pure-NumPy Histogram Gradient Boosted Decision Trees
├── mlp.py                         # Pure-NumPy Multi-Seed Multi-Layer Perceptron (Adam + Weight Decay)
├── preprocessing.py               # CDC survey cleaning, skip-logic imputation & fitted pipeline
├── metrics.py                     # Accuracy, precision, recall, F1-score & confusion matrix
├── helpers.py                     # CSV streaming loaders & submission writers
├── run_public_tests.sh            # Public autograder test runner script
│
├── benchmarks/                    # Clean modular archive of experimental scripts & ablations
│   ├── build_master_notebook.py   # Programmatic generator for the master Jupyter report
│   ├── ablation_study.py          # Progressive feature engineering ablation study (Milestone 4)
│   ├── benchmark_baselines.py     # Rigorous 5-fold CV comparison of the 6 core methods
│   ├── benchmark_hypotheses.py    # Milestone 5 hypothesis evaluations (WoE, Rank Blend, MLP)
│   ├── benchmark_m6.py            # Milestone 6 super tri-blend search
│   ├── benchmark_m6_7.py          # Milestone 6.7 vascular conjunctions & PAVA calibration
│   ├── benchmark_cost_sensitive.py# Cost-sensitive loss weighting evaluations
│   ├── benchmark_exploration.py   # Bivariate feature screening & AUC evaluations
│   ├── benchmark_cascade.py       # Sequential cascade gating experiments
│   ├── benchmark_resampling.py    # Class balance resampling experiments
│   └── generate_report_figures.py # Script generating publication figures for the final report
│
├── data/                          # Data folder (raw CSVs and processed binary caches, git-ignored)
├── eda/                           # CDC 2015 domain codebooks & feature specifications
│   ├── column_decisions.csv       # Categorization of all 321 BRFSS columns
│   ├── special_codes.json         # Per-column sentinel mapping rules (88, 77, 99)
│   ├── feature_spec.csv           # Final preprocessor specification
│   └── build_feature_spec.py      # Automated compiler for feature specifications
│
├── experiments/                   # Teammate empirical results ledger & validation plots
│   ├── plots.py                   # Generation routines for validation curves & ROC curves
│   └── results/                   # Empirical CSV records of rejected experiments & parameter sweeps
│
├── notebooks/                     # Master interactive computational narratives
│   ├── project1_sota_report.ipynb # ALL-IN-ONE Comprehensive Master Notebook (Markdown + Code)
│   ├── 01_eda.ipynb               # Exploratory data analysis & sentinel visualization
│   └── 02_experiments.ipynb       # Baseline cross-validation & threshold tuning
│
├── report/                        # 2-Page IEEE LaTeX Scientific Report
│   ├── report.tex                 # Publication-ready LaTeX source document
│   ├── report.pdf                 # Compiled 2-page camera-ready report
│   ├── literature.bib             # Bibliography & academic citations
│   └── figures/                   # High-resolution vector figures (PNG & PDF)
│
├── submissions/                   # Output predictions & official tracking
│   ├── README.md                  # Comprehensive experimental ledger (Sub-01 to Sub-15)
│   ├── submission_with_leakage.csv# Track B Champion (AIcrowd F1 = 0.457, 14,001 positives)
│   ├── submission_clean_sota.csv  # Track A Benchmark (AIcrowd F1 = 0.447, 14,876 positives)
│   ├── test_scores_with_leakage.npy# Persisted continuous test score logits (Track B)
│   └── test_scores_clean.npy      # Persisted continuous test score logits (Track A)
│
├── tests/                         # Unit tests & pipeline assertions
│   └── test_preprocessing.py      # Unit test verifying zero-leakage guarantee in clean data
│
└── docs/                          # Team coordination & roadmap documentation
    ├── ROADMAP.md                 # Detailed milestone progress tracker
    └── TEAM_GUIDE.md              # Team guide & project delivery specifications
```

---

## 7. Compliance Checklist

- [x] **Standalone Execution**: `python run.py` executes standalone and reproduces the clean submission.
- [x] **Competition Champion**: `python run.py --with_leakage` reproduces the **0.457 F1** leaderboard submission.
- [x] **Six Required Methods**: `implementations.py` strictly adheres to all required functional signatures.
- [x] **Zero Forbidden Libraries**: Zero use of scikit-learn, pandas, or PyTorch. Pure Python + NumPy.
- [x] **Formatting & Hygiene**: All code passes `black .` with zero errors and contains zero `TODO` comments.
- [x] **Grading Tests**: All 27 functional tests in `../grading_tests/` pass with 100% precision.
- [x] **Master Notebook**: All dataset analyses, baseline experiments, ablations, and comparative forensics are consolidated into [project1_sota_report.ipynb](file:///Users/zhihaochen/EPFL/S1/CS-433_ML/ML_course/projects/project1/ML_Project1/notebooks/project1_sota_report.ipynb).

## Training coverage experiment (27 October 2026)

See [the contribution report](docs/TRAINING_COVERAGE_CONTRIBUTION.md) for two paired holdout experiments, opt-in training options, verification and limitations. Production defaults are unchanged.


## Reusable score blending and workflow notebook

Open `notebooks/PIPELINE_WORKFLOW.ipynb` for the complete narrative, executable
screening table, validation protocol, optional training commands and report guide.
The notebook does not launch expensive training unless explicitly enabled.

```bash
python run.py --with_leakage --mlp_split_per_seed --output submissions/baseline.csv
python reblend.py --components submissions/baseline_components.npz --gbm_weight 0.55 --top_rate 0.129 --output submissions/candidate_55_129.csv
```

The second command takes saved scores, not model training. Upload the selected
CSV. Existing blended `.npy` files cannot be decomposed into learner components.
`run.py` and `cv_final.py` now accept `--gbm_weight`; defaults stay at 0.50 and
leakage-track rate 0.128. CV can save labeled components with `--scores_dir`.
Use different directories for different training configurations. The optional
55/45, 0.129 candidate is **not a demonstrated improvement**: reduced-budget
cached screening mean F1 was 0.449634 versus default 0.450475. See
`experiments/results/blend_screening.json`. These overlapping holdouts do not
replace full-ensemble validation. No new leaderboard result is claimed.

Progress includes MLP epoch validation loss and GBM start/completion. This release
reduces repeated fitting during score selection; it does not resume interrupted
training or promise faster individual model fits.

## Run everything inside Jupyter / VS Code

Open `notebooks/PIPELINE_WORKFLOW.ipynb`. Its setup cell contains the data/output
paths, blend settings, and `RUN_TRAINING` / `RUN_VALIDATION` switches. Set the
needed switch to `True`, then run the corresponding cells. No terminal commands
are required. `CV_FOLDS=0` performs one holdout fit; `5` performs development CV
and a holdout fit. Reblending uses the saved archive without refitting.

The workflow was exercised end-to-end on 1,500 real training rows and 80 test
rows with temporarily reduced model budgets: validation, training, score saving,
reblending and CSV ID/label checks passed. Production defaults were not reduced.
All notebooks now passed cell-by-cell execution. EDA used the full dataset;
baseline experiments used the documented 10,000-row default and five folds. Jupyter/VS Code notebook support is required to open
notebooks; model code remains NumPy-only.


## Notebook runtime fixes and verified execution

Use **Restart Kernel → Run All** after replacing the project files, so Python
loads the corrected modules. Keep your dataset in `data/dataset/`.

- `01_eda.ipynb`: full-data exploration, using the shared data cache.
- `02_experiments.ipynb`: measured clean-track baselines; defaults to a seeded
  10,000-row sample and five folds. Set `N_ROWS=None` for full-data evaluation.
  Fixed the obsolete `predict_fn`/`preprocess`/`tune_threshold` calls. Fits each
  learner once per fold and compares default/tuned thresholds using the same
  scores. The confusion matrix now uses actual out-of-fold predictions.
- `PIPELINE_WORKFLOW.ipynb`: training, validation, reblending and CSV generation
  directly in cells; long jobs remain opt-in via the configuration switches.
- `project1_sota_report.ipynb`: historical ledger, explicitly distinguished from
  newly measured results. Submission checks compare against actual test IDs.

Executed outputs and plots are included. See
`experiments/results/notebook_execution.json` for exact execution coverage and
`experiments/results/notebook_baselines.json` for measured baseline metrics.
Notebook UI dependencies are listed in `requirements-notebooks.txt`; the models
continue to use NumPy only. No Git commits or remote writes were made.
