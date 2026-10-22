# CS-433 Project 1 — Predicting coronary heart disease (MICHD) from BRFSS 2015

> **Status: work in progress (template).** Sections marked _(fill in)_ must be
> completed before the deadline. See [TEAM_GUIDE.md](TEAM_GUIDE.md) for the task
> split and checklist.

**Team:** _(fill in: name — SCIPER — GitHub user, ×3)_
**AIcrowd team name:** _(fill in)_ · **Best AIcrowd submission:** _(fill in: id, F1, accuracy)_

## Task

Binary classification: predict whether a person has coronary heart disease or
myocardial infarction (`_MICHD`, labels in {-1, 1}) from 321 features of the
CDC Behavioral Risk Factor Surveillance System 2015 survey.
Metric: F1 score (AIcrowd ranking), accuracy also reported.

## Repository structure

| File | Content |
|---|---|
| `implementations.py` | The 6 required methods (auto-graded): `mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, `reg_logistic_regression` |
| `run.py` | Reproduces our best AIcrowd submission end to end |
| `helpers.py` | Data loading and submission writing, provided by the course (unchanged) |
| `preprocessing.py` | Cleaning, special codes, imputation, encoding, standardization, feature expansion |
| `metrics.py` | Accuracy, precision, recall, F1, threshold tuning |
| `cross_validation.py` | Stratified k-fold CV and hyperparameter grid search |
| `notebooks/01_eda.ipynb` | Exploratory data analysis and figures |
| `notebooks/02_experiments.ipynb` | Baselines, CV experiments, ablation study |
| `report/` | 2-page LaTeX report (IEEE template) and figures |
| `data/` | Dataset location (not committed, see below) |

## Setup

Only Python standard library and NumPy are used for the models (matplotlib only
for plots), as required by the project rules.

```bash
# Same environment as the graders (Python 3.9, numpy 1.23.1)
conda env create --file=<course-repo>/projects/project1/grading_tests/environment.yml --name=project1-grading
conda activate project1-grading
```

## Data

Download the data from the
[AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
(or unzip `dataset_to_release.zip` from the course repo) so that you have:

```
data/dataset/x_train.csv
data/dataset/y_train.csv
data/dataset/x_test.csv
```

Do not rename the files. The `data/` folder is git-ignored.

## Reproducing our submission

```bash
python run.py --data_path data/dataset --output submissions/submission.csv
```

Expected runtime: _(fill in)_. Output: `submissions/submission.csv` with columns
`Id,Prediction`, identical to our best AIcrowd submission.

## Method summary _(fill in)_

1. **Preprocessing:** _(special codes, dropped columns, imputation, encoding, standardization)_
2. **Model:** _(final model and hyperparameters, how they were chosen)_
3. **Validation:** _(k-fold CV setup, threshold tuning)_
4. **Results:** _(CV F1 mean ± std, AIcrowd F1)_

## Running the public tests

From the course repo folder `projects/project1/grading_tests/`:

```bash
pytest --github_link <path-or-github-link-to-this-repo> .
```

Before every commit: `black .` (formatting is checked) and make sure no `.py`
file contains the word "to" + "do" (also checked by the tests).
