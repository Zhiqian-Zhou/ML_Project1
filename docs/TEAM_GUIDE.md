# Team guide — what has to be done

**Deadline: Thursday 29 October 2026, 16:00 (CET)** — report PDF + GitHub link on
[mlcourse.epfl.ch](http://mlcourse.epfl.ch). The AIcrowd competition closes at
the same time. Project 1 is not graded, but it is the rehearsal for Project 2 (30%).

## 1. Deliverables

| # | Deliverable | Where | Requirements |
|---|---|---|---|
| 1 | Code | This **public** GitHub repo | `README.md`, `implementations.py`, `run.py` at the root. Readable, commented, docstrings everywhere, `black`-formatted. Only numpy + standard library (matplotlib/seaborn only for plots). **No pandas, no sklearn, no external data.** |
| 2 | Report | PDF on mlcourse.epfl.ch | LaTeX (`report/`), **max 2 pages**, references may go on a 3rd page, **no appendix**. |
| 3 | Predictions | AIcrowd | `Id,Prediction` csv with values in {-1, 1}. Max 5 submissions/day **per team**. Ranking does not count. |
| 4 | Final submission | mlcourse.epfl.ch (HotCRP) | Upload report PDF + link to a specific commit of this repo (`https://github.com/<user>/<repo>/tree/<commit-hash>`, press `y` on GitHub to get it). |

## 2. Grading (from the project description)

- **Code:** read by 2 TAs + automatic correctness check of the 6 functions.
- **Report:** half = methods from class correctly implemented and explained;
  half = own contribution: creativity, reproducibility, **solid baselines**,
  **ablation study**, clarity.
- Code and report are checked for plagiarism.

## 3. Rules for `implementations.py` (checked by tests)

- Exact signatures; `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)`.
- Return `(w, loss)` with the **last** `w` and the loss **at that w**; `loss` is a numpy scalar (`loss.ndim == 0`, no `.item()`).
- 1-D arrays only: `y (N,)`, `w (D,)`.
- MSE = `1/(2N)·Σe²`; SGD with batch size 1.
- Ridge: solve `(XᵀX + 2Nλ I) w = Xᵀy`; returned loss **without** penalty.
- Logistic: labels in {0, 1}; regularized gradient adds `2λw`; returned loss **without** penalty.
- `max_iters = 0` → return `initial_w` and its loss.
- `np.linalg.lstsq` forbidden. No file may contain the word "to"+"do".

## 4. Task split (proposal — put your names)

| Who | Area | Files |
|---|---|---|
| **A** _(name)_ | 6 required methods + public tests passing | `implementations.py` |
| **B** _(name)_ | Data exploration + preprocessing pipeline | `preprocessing.py`, `notebooks/01_eda.ipynb` |
| **C** _(name)_ | Metrics, cross-validation, baselines, ablation | `metrics.py`, `cross_validation.py`, `notebooks/02_experiments.ipynb` |
| All | Final model, `run.py`, report | `run.py`, `report/` |

## 5. Checklist

### Setup (day 1)
- [ ] Everyone has a GitHub account added as collaborator to this repo
- [ ] Repo is **public** (required by the project description)
- [ ] Everyone created an AIcrowd account with the **epfl.ch** email; team created on AIcrowd
- [ ] Everyone created an account on mlcourse.epfl.ch
- [ ] Conda env `project1-grading` created; data in `data/dataset/`

### Code
- [ ] `implementations.py`: 6 functions implemented, all public tests pass locally
- [ ] Preprocessing: special codes → NaN, drop useless/empty columns, impute (train stats only), missing indicators, one-hot, standardize
- [ ] Metrics: F1, accuracy, threshold tuning
- [ ] Stratified 5-fold CV + grid search over λ, γ, max_iters, degree, threshold
- [ ] Baselines evaluated with CV: always −1, least squares, ridge, logistic, regularized logistic
- [ ] Imbalance handling tried (threshold tuning, class weights / undersampling)
- [ ] Ablation table: F1 change for each preprocessing/model step
- [ ] `run.py` reproduces the best AIcrowd csv exactly (fixed seeds)
- [ ] `README.md` completed; `black .` run; no leftover markers

### Report
- [ ] Introduction, Models and Methods, Results, Discussion, Summary
- [ ] All hyperparameters and exact feature transformations stated
- [ ] Results as mean ± std over folds; baselines + ablation table
- [ ] AIcrowd test F1/accuracy reported
- [ ] ≤ 2 pages (+ references), no appendix, PDF compiled

### Submission (before 29 Oct 16:00)
- [ ] Public tests pass against the GitHub commit link
- [ ] PDF + commit link uploaded on mlcourse.epfl.ch with all 3 authors

## 6. Git workflow

- One branch per task (`feature/implementations`, `feature/preprocessing`, ...), merge to `main` via pull request.
- Pull before you start, commit small, run `black .` before each commit.
- Never commit data or submission csv files (`.gitignore` takes care of it).
- Notebooks import from the `.py` modules; do not copy functions into notebooks.

## 7. Useful links

| Link | Use |
|---|---|
| [AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1) | Data, submissions, leaderboard |
| [mlcourse.epfl.ch](http://mlcourse.epfl.ch) | Final submission |
| [BRFSS 2015 codebook](https://www.cdc.gov/brfss/annual_data/annual_2015.html) | Meaning of the 321 columns and special codes |
| [Course repo — project1](https://github.com/epfml/ML_course/tree/main/projects/project1) | Description, tests, LaTeX template |
| [Ed forum](https://edstem.org/eu/courses/3637) | Questions |
| [Scientific writing guide](http://arxiv.org/pdf/1609.00037) | Writing the report |

Lab code to reuse (course repo): `labs/ex02/solution` (GD/SGD), `labs/ex03/solution`
(least squares, ridge, build_poly), `labs/ex04/solution` (cross-validation),
`labs/ex05/solution` (logistic regression).
