# Training coverage contribution — 27 October 2026

## Provenance and remote status

Prepared against the uploaded `ML_Project1-zhihao.zip` snapshot, whose ZIP comment identifies `92bb5d475086bf3e754692e4bd2fa7b1ee00de35`. The GitHub connection returned 404 for `Zhiqian-Zhou/ML_Project1` and did not list it among the connected account’s accessible repositories. **No remote commit history was inspected and no branch was pushed.** Reconcile this patch with the latest shared branch before applying it.

## Observation and hypothesis

The existing GBM uses only the inner 80% of each supplied training set for tree updates; the remaining 20% selects stopping time and is not subsequently used for refitting. All ten existing MLP seeds also share the same 80/20 split. Hypothesis: increasing training-row coverage can reduce model variance without adding new features or external data.

## Changes

- `gbm.py`: optional `refit_full=True` selects rounds on an internal split, then refits that many trees and the state encoding on all supplied training rows. Its pilot also fits bin edges on internal training rows only, so this candidate combines full refitting with stricter pilot bin fitting. The default remains False.
- `mlp.py`: optional `split_per_seed=True` varies the stratified validation split with each network seed. The first network uses the old split. The default remains False.
- `run.py` / `cv_final.py`: expose both experimental flags. Experimental default filenames are separate from production filenames. Create the output directory before saving continuous scores, preventing a failure after expensive training.
- `benchmarks/benchmark_training_coverage.py`: paired holdouts, fixed operating rate, saved scores and row IDs, paired bootstrap intervals, and machine-readable results.
- `tests/test_training_coverage.py`: coverage, reproducibility, compatibility and output-name checks.

## Measured screening results

Sample: 60,000 randomly selected training rows (seed 99). Each stratified holdout has 45,000 training and 15,000 validation rows. Outer split seeds: 20261027 and 20261028. Track B features and existing HAREHAB1 boost, top rate 0.128 fixed in advance; 50/50 GBM/MLP logit blend. One GBM seed (42), at most 300 trees; three MLP seeds. This is deliberately smaller than production (three GBMs and ten MLPs).

| Variant | Split 1 F1 | Split 2 F1 | Mean F1 | Mean delta |
|---|---:|---:|---:|---:|
| baseline | 0.456915 | 0.438516 | 0.447715 | +0.000000 |
| gbm_refit_only | 0.457528 | 0.441582 | 0.449555 | +0.001840 |
| mlp_varied_only | 0.459368 | 0.441582 | 0.450475 | +0.002760 |
| both | 0.456302 | 0.439129 | 0.447715 | +0.000000 |

The MLP-only change gains approximately 0.00276 F1 on average, with improvement on both splits. GBM refitting gains approximately 0.00184 but roughly doubles fitting work. Combining them has zero average F1 improvement here. **Prioritize MLP split diversity for full-budget evaluation; retain existing production defaults.**

Every candidate’s per-split paired 95% bootstrap interval crosses zero (see JSON). These intervals condition on fitted models and fixed predictions; they do not include model retraining uncertainty. The two holdouts overlap and are not independent folds. This screening does not establish statistical significance, a new AIcrowd score, or a winning model.

## Reproduce

```bash
python -m pytest tests -q
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python benchmarks/benchmark_training_coverage.py --with_leakage
# Full-budget evaluation before choosing a submission:
python cv_final.py --with_leakage --mlp_split_per_seed
# Generate the candidate after reviewing full-budget validation:
python run.py --with_leakage --mlp_split_per_seed
```

The candidate CSV is `submissions/submission_with_leakage_mlp_splits.csv`; scores are saved to `submissions/test_scores_with_leakage_mlp_splits.npy`. No candidate CSV was generated or submitted during this contribution. The test dataset was not used for candidate selection.

## Verification

Seven tests pass (the existing two plus five new tests). Changed Python files pass Black and the diff has no whitespace errors. Both model defaults produce exactly the same synthetic-fixture scores as modules extracted directly from the original uploaded archive. The six course algorithms and helpers.py are unchanged.

## Audit findings for the next step

- The README’s “global optimum” and numerical attribution of leaderboard changes are stronger than the recorded evidence supports. Keep leaderboard observations distinct from causal or independent validation claims.
- The six supervised risk encodings are fitted before the internal model stopping splits; thus the inner early-stopping metrics are not completely label-independent. External holdout labels remain excluded. Consider cross-fitted encodings in a separate ablation rather than changing the champion silently.
- The new options remain hypotheses until evaluated at the full production budget. Preserve the current best submission and its exact score vector.
- Confirm the shared branch name and enable repository access before integrating; the uploaded snapshot may lag your teammate’s latest commits.
