# CS-433 Project 1: Predicting coronary heart disease (MICHD) from BRFSS 2015

**Team:** _(name, SCIPER, GitHub user, ×3)_
**AIcrowd team name:** _(team name)_. **Best AIcrowd submission:** _(id)_, F1 0.448, accuracy _(acc.)_

## Task

Binary classification. From the 321 features of the CDC Behavioral Risk Factor
Surveillance System 2015 survey, predict whether a person has coronary heart
disease or myocardial infarction (`_MICHD`, labels in {-1, 1}).

- Data: 328,135 training rows and 109,379 test rows.
- Only 8.8% of the training labels are positive.
- Metric: F1 of the positive class (AIcrowd ranking). Accuracy is also reported.

## Results

| Model | 5-fold CV F1 (mean ± std) |
|---|---|
| Regularized logistic regression (`cross_validation.py`) | 0.428 |
| Gradient boosted trees (`gbm.py`) | 0.435 |
| Neural network, mean of 10 (`mlp.py`) | 0.433 |
| **Final: GBM + MLP, top 14% predicted +1** | **0.4359 ± 0.0034** |

- Held-out 20% (never used for selection): F1 0.4353.
- AIcrowd public leaderboard: F1 0.448.

## Setup

The models use only the Python standard library and NumPy, as the project
rules require. matplotlib is used only for the plots in the EDA notebook.

```bash
# Same environment as the graders (Python 3.9, numpy 1.23.1)
conda env create --file=<course-repo>/projects/project1/grading_tests/environment.yml --name=project1-grading
conda activate project1-grading
```

## Data

Download the data from the
[AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1),
or unzip `dataset_to_release.zip` from the course repo, so that you have:

```
data/dataset/x_train.csv
data/dataset/y_train.csv
data/dataset/x_test.csv
```

Do not rename the files. The `data/` folder is git-ignored.

On the first run the CSVs are parsed once and cached in `data/cache/raw.npz`
(about 1.1 GB). Delete that file to rebuild the cache.

## Reproducing our submission

```bash
python run.py                       # = --data_path data/dataset --output submissions/submission.csv
```

- **Runtime:** about 5 minutes on a laptop CPU (12 cores), plus about 1 minute
  the first time, to parse the CSVs.
- **Output:** `submissions/submission.csv` (`Id,Prediction`, 109,379 rows).
- **Determinism:** every random step has a fixed seed, so the file is
  identical at every run and identical to our best AIcrowd submission.

Cross-validation of the final model (about 25 minutes):

```bash
python cv_final.py                  # 5-fold CV on all training rows
python cv_final.py --holdout        # CV on the 80% dev rows + F1 on the 20% held-out rows
```

## Repository structure

| File | Content |
|---|---|
| `implementations.py` | The 6 required methods: `mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, `reg_logistic_regression` |
| `run.py` | Trains the final model on all training rows and writes the submission |
| `cv_final.py` | k-fold cross-validation of the final model (same steps as `run.py` inside every fold) and the held-out check |
| `preprocessing.py` | Cleaning (special codes, units, skip-logic fills, derived features) and the fitted pipeline (column selection, imputation, one-hot, missing indicators, log1p, standardization, interactions) |
| `gbm.py` | Gradient boosted decision trees in NumPy (histogram splits, Newton leaves, early stopping) |
| `mlp.py` | One-hidden-layer neural network in NumPy (ReLU, Adam, weight decay, early stopping, mean of 10 seeds) |
| `cross_validation.py` | Stratified k-fold split, CV loop with the preprocessor fitted on the training folds, grid search, linear baselines |
| `metrics.py` | Accuracy, precision, recall, F1, threshold search |
| `helpers.py` | Data loading and submission writing, provided by the course (unchanged) |
| `eda/` | `column_decisions.csv` (kind of each of the 321 columns), `special_codes.json` (per-column codebook rules), `feature_spec.csv` (built by `build_feature_spec.py`, read by `preprocessing.py`) |
| `notebooks/01_eda.ipynb` | Exploratory data analysis, which generates the figures in `report/figures/` |
| `tests/` | Unit test: leaky columns never reach the model input (`pytest tests`) |
| `report/` | 2-page LaTeX report (IEEE template) |

## Method

### 1. Cleaning (`preprocessing.clean_data`, stateless, row by row)
- **Special codes, per column, from the 2015 codebook (`eda/special_codes.json`):**
  - 7/9, 77/99 and 777/999 ("don't know / refused") become NaN;
  - 88 and 888 ("none") become 0;
  - the rule is set per column because in some columns 7 or 99 are real values.
- **Units:** weight is converted to kg and height to cm; frequency codes
  (1xx per day / week, 2xx per week / month, ...) are converted to one unit.
- **Skip-logic fills:** questions that were not asked because of an earlier
  answer get the implied value. For example, "taking BP medication" becomes
  "no" when the person has no high blood pressure.
- **Derived features:** a few, such as number of adults in the household and
  years with diabetes.
- **"Don't know" flags:** one per question, set when the answer was "don't
  know / refused".
- **Dropped columns:** survey administration and design columns, near-empty
  columns and redundant recodes.
- **Leaky questions:** the questions asked only to people who already had a
  heart attack (cardiac rehabilitation, aspirin for the heart, ...) are
  dropped. They are HAREHAB1, CVDASPRN, RDUCHART, RDUCSTRK, RLIVPAIN,
  ASPUNSAF and STREHAB1. They would raise CV F1 (+0.01) only by leaking the
  label. `tests/test_preprocessing.py` checks that they never reach the
  model input.

### 2. Fitted pipeline (`fit_preprocessor` / `transform`)
Every statistic is computed on the training rows only, and inside
cross-validation on the training folds only:
- column selection (at most 95% NaN, not constant);
- yes/no recoded to 1/0;
- median/mode imputation;
- one-hot encoding of nominal columns (the reference is the mode; rare levels
  are grouped);
- missing-value indicators;
- log1p of count columns;
- standardization;
- optional pairwise interactions;
- a bias column.

The MLP uses this pipeline without interactions (about 130 columns).

### 3. Models
- **`gbm.py`:** logistic loss with Newton leaf values.
  - Inputs: the cleaned features, where NaN is a separate bin and the split
    learns where to send it.
  - Parameters: 64 bins, depth 5, learning rate 0.05, row subsample 0.7,
    column subsample 0.8, min 200 rows per leaf, L2 = 5.
  - Early stopping uses an inner 20% split.
  - The state of residence (`_STATE`) is replaced by its smoothed rate of
    heart disease, fitted on the training rows only.
- **`mlp.py`:** 64 ReLU units, Adam (learning rate 1e-3, batch 512), L2
  weight decay 1e-3, early stopping on an inner 20% split. The output is the
  mean logit of 10 networks with different seeds.
- **Final score:** 0.5·GBM logit + 0.5·MLP logit. The 14% of rows with the
  highest scores are predicted +1. That share maximized F1 in
  cross-validation, and F1 is flat between 12% and 16%.

### 4. Validation
- **Folds:** stratified 5-fold cross-validation (`cross_validation.build_k_indices`).
  The preprocessor, the target encoding, both models and the decision rule
  are fitted inside each fold.
- **Hold-out:** a stratified 20% hold-out (`cv_final.holdout_split`, seed
  2026) was kept aside during model selection and scored only for the final
  candidates.
- **No choices on AIcrowd or on the test set:** no hyperparameter was chosen
  on the AIcrowd leaderboard or on the test set.
- **Rejected alternatives:** none improved CV F1 by at least 0.001 on 4 of
  the 5 folds. Details are in the report. They include:
  - class weights, focal loss, ridge, linear SVM;
  - deeper or bigger trees and networks, refitting on 100% of the rows;
  - feature selection, PCA, extra engineered features and target encodings;
  - stacking, random forest and k-NN members, per-group thresholds.

## Running the tests

From the course repo folder `projects/project1/grading_tests/`:

```bash
pytest --github_link <path-or-github-link-to-this-repo> .
```

Our own test: `pytest tests`. Formatting: `black .` (black 22.6.0).
