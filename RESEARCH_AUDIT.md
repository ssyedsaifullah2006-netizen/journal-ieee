# PACE Framework — Research Audit & Revision Log

Branch: `research-revision` (from `main` @ 191efcd). This document records the
mathematical, statistical, experimental, and implementation audit of the PACE
(Preferential-Attachment Corrected Endorsement) consensus estimator and the
fixes applied. Findings are evidence-based; no claim is asserted without a
corresponding check or citation.

---

## 1. Mathematical audit

**Model.** PACE fits the log-linear exposure surface
`ln(L_i + 1) = β0 + β1·ln(r_i) + β2·ln(1 + Δt_i) + ε_i` by OLS
(`np.linalg.lstsq`) and retransforms with Duan's (1983) smearing factor
`ŝ = mean(exp(ê))`, so `Ê[L_i] = max(0, ŝ·exp(X_iβ) − 1)`. The retransformation
is arithmetically correct **under i.i.d. homoscedastic ε**.

Findings:
- **M1 — No parameter uncertainty.** β are reported as point estimates only.
  → Add OLS standard errors, t-statistics, and residual df.
- **M2 — Homoscedasticity is assumed but never tested.** Duan smearing is
  consistent only under constant variance; heavy-tailed engagement almost
  certainly violates this. → Add a Breusch–Pagan test on log-scale residuals.
- **M3 — Collinearity is unexamined.** `ln(rank)` and `ln(1+Δt)` are correlated
  in real data (retrieval order is temporally correlated), so β is unstable.
  → Add design condition number and VIF.
- **M4 — In-sample residuals are optimistic.** Residuals come from the same fit
  used to estimate β, biasing `ŝ` downward (over-confident baseline). → Add a
  leave-one-out (PRESS) variant `estimate_pace_loo` for comparison.
- **M5 — Arbitrary hyperparameters.** sigmoid clip ±20, depth discount 0.8,
  MAD scale 1.4826, NC-IPS γ=0.8 and 90th-pct clip, heuristic reply weight 0.5
  are fixed without sensitivity. → Report these as hyperparameters; add a
  sensitivity sweep in the expanded simulations.

## 2. Endogeneity and causal scope

- `rank` is the **retrieval-time relevance order** returned by the API, which the
  platform computes partly from like counts. Regressing likes on rank is
  therefore **endogenous/circular**. PACE estimates an *observational exposure
  proxy*, not a causal exposure curve.
- **Decision: PACE must never be described as a causal estimator.** The manuscript
  already warns; this revision hardens the language to "exposure-proxy conditioning"
  and adds the endogeneity caveat to every fit diagnostic.
- Public APIs do not expose historical display rank or impressions; no method can
  recover them. This is a data boundary, not a code defect.

## 3. Statistical audit

- Paired Wilcoxon signed-rank on trial-level MAE: appropriate for paired trials.
- Full-pipeline refitting bootstrap (Davison & Hinkley): correctly resamples
  records and refits the estimator.
- Permutation divergence test: tests whether two weight vectors yield
  distinguishable consensus under **sentiment exchangeability**. It is *not* a
  test of "PACE superiority" per se and is reported as such.
- Findings:
  - **S1 — No multiple-comparison control.** Several p-values are reported
    without correction. → Report exact p-values; avoid "significant" language;
    treat p-values as descriptive where multiple tests are run.
  - **S2 — M = 50 trials gives limited power.** → Add 95% CIs on the MAE
    differences (paired) rather than relying on p alone.
- ESS = 1/Σw², Gini, and leave-top-10% ρ are computed correctly.

## 4. Edge cases (tested)

- n < 5 → unweighted fallback (safe).
- All-zero likes → β=0, residuals=0, sigmoid(0)=0.5 → uniform weights (degenerate
  but non-crashing). Flagged, not silently used.
- Negative/absent likes → clamped to 0 in the constructor (safe).

## 5. Fair-baseline comparison

- NC-IPS (γ=0.8, 90th-pct clip) and Heuristic (reply weight 0.5) use fixed
  hyperparameters. The comparison is fair in the sense that all methods use the
  same inputs; the hyperparameter sensitivity is reported separately.

## 6. Reproducibility

- Synthetic generators are seed-deterministic (42..91). Bootstrap/permutation use
  fixed seeds. → Preserved.
- Live YouTube fetch is non-deterministic (like counts/relevance drift); dated
  SHA-256 provenance mitigates. → Preserved.

## 7. Data limitations (NOT code defects — require new data)

- Human-annotated accuracy audit is **single-annotator, N = 100, one video**.
- Multi-video extension is VADER-scored only (no human labels).
- **Inter-rater agreement cannot be computed without a second annotator.**
  This revision ships the tooling (`analyzers/interrater.py`) and a labeling
  protocol, but does **not** fabricate second-rater data.

## 8. Changes applied in this revision

- [x] OLS inference diagnostics (SE, t, Breusch–Pagan, VIF, condition number)
- [x] `estimate_pace_loo` (PRESS) variant
- [x] `analyzers/interrater.py` (Cohen's κ, weighted κ, Krippendorff's α, Fleiss κ, % agreement)
- [x] Second-annotator labeling protocol + blind template generator
- [x] Expanded simulations (`experiments/expand_simulations.py`)
- [x] Automated tests (`tests/test_core.py`)
- [x] Manuscript updates (honest, additive)

## 9. Expanded-simulation findings (honest, not favorable to PACE)

- **E1 (engagement distribution):** PACE's MAE advantage over the heuristic
  *shrinks* as likes become heavier-tailed (uniform 0.012 → Pareto 0.057).
  The gain is not distribution-robust.
- **E2 (ranking endogeneity):** with rank correlated with sentiment, the
  unweighted mean is unbiased (MAE 0.000); every weighted estimator adds bias.
  PACE is the *least* biased weighted estimator (0.006 vs NC-IPS 0.189 at strong
  correlation), but it never beats the mean. This confirms PACE cannot recover
  an unbiased mean under endogenous ranking.
- **E3 (measurement error):** under 10–30% label flips, NC-IPS is the most
  robust estimator and PACE degrades markedly; at 50% noise all estimators
  converge to the unweighted error. PACE is not robust to sentiment noise.
- **E4 (edge cases):** N=1 and all-zero-likes degrade to the unweighted mean
  (safe). NC-IPS shows a small N=2 bias (−0.125). PACE-LOO matched PACE on the
  corpus but should be treated as a diagnostic, not a replacement, pending
  further stress-testing under extreme high-leverage outliers.

## 10. Remaining gaps that require further research (not fixable in code)

- A **real second annotator** is required to compute inter-rater reliability;
  the tooling is ready but no second-rater data exists.
- The human-annotated accuracy audit remains **single-annotator, N=100, one video**.
- PACE's conditional advantage is **scenario- and measurement-specific**; the
  framework does not support a "universally superior" claim and no such claim is made.

