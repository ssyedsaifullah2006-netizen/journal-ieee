# PACE Mathematical Validation

Audit of the estimator's mathematical components. Distinguishes **mathematical
properties** (provable) from **empirical observations** (measured).

## Components

| Component | Property | Status |
|---|---|---|
| Weight normalization | Σw = 1 by construction | ✅ exact |
| ESS | 1/Σw² ∈ [1, N] | ✅ exact |
| Gini | standard formula, clipped to [0,1] | ✅ exact |
| MAD standardization | z = (δ−median)/(1.4826·MAD) | ✅ standard robust scale |
| Sigmoid gate | w ∈ (0,1) strictly | ✅ exact |
| Depth damping | 0.8^depth ∈ (0,1] | arbitrary hyperparameter |
| Residual clip | clip(z, ±20) | arbitrary hyperparameter |
| Duan smearing | valid iff homoscedastic i.i.d. ε | assumption, untested (BP p=0.099) |

## Key finding: sigmoid is NOT bounded-influence in the claimed sense

The sigmoid maps each residual to (0,1), but after **normalization** the maximum
weight share `w_max* = max w / Σw` is **not** bounded away from 1. If one
residual is extreme, its sigmoid → 1 while others → 0, so `w_max* → 1`. The
manuscript already states this correctly ("does not mathematically bound the
normalized weight share"); here it is confirmed as a mathematical fact, not an
assertion.

## Empirical observations (synthetic + real)

- Scenario 1 (viral outlier): PACE reduces outlier weight share 0.057 → 0.041
  (−28%), but still above equal weighting (1/50 = 0.02). So PACE *dampens* but
  does not *eliminate* outlier influence.
- Real corpus: rank coefficient marginally significant (HC3 t = −2.13), arrival
  strongly significant (t = −11.06). Condition number 597.9 (scale, not
  collinearity — VIF = 1.0). Max Cook's D 0.071 (no dominating observation).
- PACE-LOO ≈ PACE on the corpus (0.0792) — the smearing estimate is not the
  deciding factor; the sigmoid gating is robust to it.

## Sensitivity (from expand_simulations.py, E1/E3)

- **Engagement distribution:** PACE's MAE advantage over the heuristic shrinks
  as likes become heavy-tailed (uniform 0.012 → Pareto 0.057).
- **Measurement error:** under 10–30% label flips, NC-IPS is more robust than
  PACE; at 50% noise all estimators converge to the unweighted error.
- **Edge cases:** N=1 and all-zero-likes degrade to the unweighted mean (safe).
  NC-IPS shows a small N=2 bias (−0.125).

## Conclusion

PACE is mathematically well-defined and its normalization/ESS/Gini are exact.
Its robustness claims must be stated as **empirical and scenario-specific**, not
as formal bounded-influence guarantees. No such guarantee is claimed here.
