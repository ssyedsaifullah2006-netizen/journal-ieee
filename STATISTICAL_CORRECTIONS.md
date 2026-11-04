# Statistical Corrections (research-revision)

Independent corrections to the reliability statistics and exposure-model
inference. All changes are on `research-revision`; every claim is backed by a
run or a mathematical identity.

## 1. Krippendorff's alpha — corrected (was wrong)

**Before:** `krippendorff_alpha()` assumed nominal alpha equals Cohen's kappa
for two raters and substituted Fleiss' kappa for >2 raters. Both shortcuts are
incorrect in general.

**Correction:** implemented the coincidence-matrix formulation with the correct
expected-disagreement term `D_e = (N^2 - sum n_c^2) / N^2`. The prior code used
`N(N-1)` in the denominator, which was the concrete arithmetic bug.

**Key identity (validated):** for two raters with no missing data, nominal
Krippendorff's alpha equals **Scott's pi** (pooled marginals), NOT Cohen's kappa
(which uses rater-specific marginals). Verified over 300 random datasets:
max |alpha − Scott's pi| = 3.9e-16.

**Edge cases:** perfect agreement → 1.0; complete (binary) disagreement → −1.0;
single-category (zero-variability) → NaN (undefined); missing ratings and ≥3
raters handled by the coincidence matrix (units with <2 ratings contribute no
coincidence and are skipped).

## 2. Fleiss' kappa — corrected (was inflating agreement)

**Before:** units with a single valid rating were assigned `P_i = 1.0` (treated
as perfectly consistent), artificially inflating agreement.

**Correction:** units with fewer than two valid ratings are **excluded**. When
the number of valid ratings varies across units, the per-unit `n_i` is used
(a documented generalization that is exact only for constant `n_i`).

## 3. OLS inference — HC3 + influence diagnostics added

`ols_inference()` now reports, alongside conventional SEs and Breusch–Pagan:
- HC3 heteroscedasticity-consistent standard errors and t-statistics,
- maximum leverage, high-leverage count (threshold 2p/n), and maximum Cook's D.

**Observed on the real corpus:** conventional rank t = −1.79 vs HC3 rank t =
−2.13 (marginally significant under HC3); arrival t = −12.07 (conventional) /
−11.06 (HC3), strongly significant. Max leverage 0.169, max Cook's D 0.071 —
no single observation dominates the fit.

## 4. Validation harness

`tests/test_interrater.py` (13 checks) and `tests/test_core.py` (13 checks) pass;
the QA verifier passes 22/22. The external `krippendorff` package could not be
installed in the sandbox (PyPI blocked), so validation relies on the
Scott's-pi identity and analytic edge cases, which are exact.
