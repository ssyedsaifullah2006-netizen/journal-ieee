# Baseline Methodology V2 (IPS vs IPE)

## Definitions (implemented and validated)

1. **Unweighted Mean** — `estimate_unweighted()`. μ̂ = (1/N) Σ s_i.
2. **Heuristic Log-Likes** — `estimate_heuristic_log_likes()`.
   w_i = 1 + ln(1+L_i) + 0.5·ln(1+R_i), normalized.
3. **IPE (Inverse-Propensity Engagement-Weighted)** — `estimate_position_ips()`.
   w_i ∝ clip(rank_i^{γ}) · ln(1+L_i). This is an engagement-weighted hybrid;
   it is NOT Joachims' NC-IPS. (Previously mislabeled NC-IPS.)
4. **NIPS (Normalized Clipped IPS)** — `estimate_normalized_ips()`.
   w_i ∝ clip(rank_i^{γ}); μ̂ = Σ w_i s_i / Σ w_i. Genuine normalized IPS
   (Joachims et al. 2017) for the sentiment mean.
5. **PACE** — `estimate_pace()`.

## Propensity assumptions and limitations (documented)

- Propensity is modeled as p_i ∝ rank_i^{-γ} with γ = 0.8 and a 90th-percentile
  clip. γ and the clip are **prespecified**, not tuned to the test outcome.
- **YouTube retrieval rank is NOT historical exposure or impression
  probability.** The API returns relevance order, which is itself partly a
  function of likes (endogeneity). Both IPS and IPE therefore estimate against a
  *retrieval-time proxy*, not a causal exposure probability. No causal claim is
  made; "exposure" is observational proxy only.
- The inverse-propensity estimators require positivity (p_i > 0) and correct
  propensity specification. Neither is verifiable from the public API; the
  results are therefore descriptive, not unbiased-causal estimates.

## Validation

- All five estimators run in `run_all_and_compare()`, each with its own
  full-pipeline bootstrap CI and (for PACE) permutation divergence tests vs.
  unweighted/heuristic/IPE/NIPS.
- Endogeneity simulation (7 DGPs, M=500) confirms: NIPS and IPE overcorrect under
  rank–sentiment confounding; PACE is least biased among weighted estimators;
  unweighted is unbiased for the participant-mean estimand.
