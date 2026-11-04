# Statistical Validation V2

Validated statistics and their test status (all executed on `research-revision`).

| Component | Status | Validation |
|---|---|---|
| Krippendorff's α | ✅ correct | coincidence matrix; = Scott's π for 2 raters (max diff 3.9e-16 over 300 datasets); perfect=1, complete=−1, single-category=NaN |
| Fleiss' κ | ✅ correct | single-rating units excluded (no inflation); perfect=1 |
| Cohen's κ / Scott's π / weighted κ | ✅ | independently implemented; κ≠π on unbalanced marginals |
| HC3 SEs + leverage + Cook's D | ✅ | rank HC3 t=−2.13 (vs conventional −1.79); max leverage 0.169; max Cook's D 0.071 |
| Breusch–Pagan | ✅ | p=0.099 (no strong heteroscedasticity on log scale) |
| VIF / condition number | ✅ | VIF=1.0; cond 597.9 (scale, not collinearity) |
| Duan smearing | ✅ | smear ≥ 1 (Jensen); LOO smear 15.4 vs in-sample; PACE score unchanged (0.0792) |
| ESS / Gini / w_max | ✅ | ESS=1/Σw²; Gini standard; w_max normalized |
| Bootstrap CI | ✅ | full-pipeline refitting (resample records, refit estimator), per-method |
| Permutation test | ✅ | plus-one Monte Carlo; now includes PACE vs IPE and PACE vs NIPS |
| Wilcoxon paired | ✅ | on trial-level MAE |

## Test results (executed)

- `tests/test_core.py` — 13/13 pass
- `tests/test_interrater.py` — 13/13 pass
- `experiments/run_exhaustive_verification.py` — 22/22 pass, 0 warnings

No test is reported as passing unless it was actually run. No second-rater
annotations were fabricated; inter-rater agreement remains **not computed**
pending real data.
