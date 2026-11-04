# Baseline Comparison Report (NC-IPS audit)

## Finding: the "NC-IPS" baseline is mislabeled

The implementation in `estimate_position_ips()` computes

```
raw_w_i = clip(1 / rank_i^{-gamma}) * log1p(likes_i)
```

i.e. **inverse-rank-propensity × log-likes**, then normalizes and takes the
weighted mean of sentiment.

This is **not** the Normalized Clipped Inverse Propensity Scoring (Joachims et
al. 2017) that it cites. Standard (normalized) IPS for a mean estimand is

```
mu_hat = sum_i (s_i / p_i) / sum_i (1 / p_i)
```

where `p_i` is the observation/selection propensity. It does **not** multiply by
`log1p(likes)`. Multiplying by engagement converts the estimator into an
*engagement-weighted* quantity that targets a different estimand and is not
unbiased for the sentiment mean.

## Consequences

1. The label "NC-IPS" must be corrected in the manuscript, e.g. to
   **"Inverse-Propensity Engagement-Weighted (IPE)"**, or a genuine normalized
   IPS must be implemented and compared.
2. This explains the endogeneity simulation result: under rank–sentiment
   confounding (scenario B), the "NC-IPS" baseline produced a large bias of
   −0.176 — because inverse-rank weights amplify low-ranked (negative) comments,
   and the `log1p(likes)` factor does not restore unbiasedness.

## Fair-comparison assessment

- All methods receive the same inputs (sentiment, likes, replies, rank, arrival).
- Hyperparameters (γ = 0.8, 90th-percentile clip, heuristic reply weight 0.5) are
  **prespecified**, not tuned to the test outcome — this is appropriate.
- The unweighted mean is the correct reference for the participant-mean estimand
  and is unbiased by construction (verified: bias 0.000 in all seven endogeneity
  scenarios).

## Recommendation

Rename the baseline honestly, or implement a genuine normalized IPS and report
both. Do **not** claim the current implementation matches Joachims' NC-IPS.
