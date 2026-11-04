# Novelty & Prior-Art Review (V2)

Honest separation of established techniques, methodological contribution,
empirical contribution, and limitations.

## 1. Established techniques (not novel)

- Position bias / examination hypothesis and IPS: Joachims et al. 2017; Wang et
  al. 2018; Craswell et al. 2008.
- Robust regression / bounded influence: Huber 1981; Hampel et al. 1986;
  Rousseeuw & Croux 1993 (MAD).
- Retransformation bias correction: Duan 1983.
- Lexicon sentiment: VADER (Hutto & Gilbert 2014), SentiStrength (Thelwall 2010).
- Engagement-weighted sentiment heuristics: widely used in commercial analytics.

## 2. Actual methodological contribution (incremental)

PACE combines, for **continuous sentiment consensus** (not click/rank
prediction): a log-linear rank+arrival exposure surface, Duan's smearing
retransformation, MAD-studentized residual gating via a sigmoid, and a tree-depth
damping factor. Each ingredient is established; the combination is **incremental**
rather than a new theory. There is no defensible "first-of-its-kind" claim.

## 3. Empirical contribution (the stronger, defensible part)

- A reproducible, chance-corrected audit showing **exposure-conditioning is
  sensitive and not universally superior** (multi-trial + endogeneity + noise
  simulations).
- A **lexicon reliability audit**: VADER at 65.0% accuracy, κ = 0.402 vs. a
  majority-class baseline, 46% zero-inflation, and 23.9% positive→negative
  misclassification on informal discourse; multi-video zero-inflation ranges
  11–50%.
- The observation that **NC-IPS-style weighting overcorrects** under rank–
  sentiment confounding (bias −0.176 vs PACE −0.016).

## 4. Limitations and unresolved questions

- Single-annotator accuracy audit; no inter-rater reliability yet.
- Single-video human annotation (multi-video is VADER-scored only).
- PACE's advantage is scenario- and measurement-specific.
- No transformer-based sentiment baseline (lexicon vs. neural comparison is
  discussed, not run).

## Conclusion

Position the paper as a **reproducibility + audit methodology** contribution,
not a novel estimator. Do not claim novelty for established components.
