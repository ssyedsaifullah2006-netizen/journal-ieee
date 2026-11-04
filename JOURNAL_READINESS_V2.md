# Journal-Readiness Assessment (V2)

## Verdict: **B — Improved but not submission-ready**

The statistical implementation and experiments are now corrected and validated;
the remaining blockers are **data/annotation gaps**, not implementation errors.

## What is now fixed and validated (this revision)

- Krippendorff's alpha: correct coincidence-matrix implementation (was wrong;
  D_e denominator and the Cohen's-kappa substitution both corrected). Validated
  against Scott's pi (max diff 3.9e-16 over 300 datasets) and analytic edge cases.
- Fleiss' kappa: single-rating units no longer inflated to perfect agreement.
- OLS inference: HC3 standard errors and influence diagnostics added.
- Endogeneity simulation: redesigned with independent controls; 7 DGPs; 500
  trials; bias/MAE/RMSE/SE/CI reported.
- NC-IPS baseline: identified as mislabeled (it is inverse-propensity × log-likes,
  not Joachims' NC-IPS).
- Corpus: SHA-256 provenance verified; no fabrication.

## Why not "A" (research-ready)

1. **No second rater** → inter-rater reliability cannot be computed; the
   accuracy audit remains single-annotator (N=100, one video).
2. **Multi-video human annotation absent** — the four-video extension is
   VADER-scored only.
3. **Manuscript must be updated** to (a) relabel the NC-IPS baseline honestly
   and (b) reflect the corrected statistics and the endogeneity findings.

## What further research is required

- A second/third independent annotator (the single highest-leverage step).
- Human labels for the additional videos.
- A transformer-based sentiment baseline (lexicon vs. neural), if the semantic
  contribution is to be made stronger.

## What the framework can honestly claim

PACE is a defensible **observational diagnostic** that meaningfully reduces
exposure-induced bias among weighted estimators and outperforms naive inverse-
propensity weighting under confounding — but it does **not** outperform the
unweighted mean for the participant-mean estimand. The lexicon audit is a real,
reproducible empirical contribution. Neither is a "universally superior" result,
and the manuscript should continue to avoid that claim.

## Bottom line

Correct and honest now; **not yet independently validated at the annotation
level**. A second annotator is the gating item for "A — research-ready."
