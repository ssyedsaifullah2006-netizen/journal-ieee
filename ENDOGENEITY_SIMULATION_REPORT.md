# Endogeneity Simulation Report

`experiments/endogeneity_simulation.py` replaces the flawed `run_e2()` (which
set `arrival_hours = rank * 0.2`, making rank and arrival perfectly collinear).

## Design

Seven DGPs with **independently controlled** sentiment, rank, arrival, likes,
replies, and exposure propensity. **Target estimand (declared before running):**
the unweighted mean of true sentiment (participant consensus). M = 500 trials,
N = 80, independent seeds. Reported: bias, MAE, RMSE, SE, 95% CI of bias.

Scenarios:
- A: rank independent of sentiment and arrival.
- B: rank correlated with sentiment.
- C: arrival correlated with sentiment.
- D: rank and arrival independently correlated with sentiment.
- E: rank and arrival correlated with each other, independent of sentiment.
- F: likes drive retrieval rank (endogeneity).
- G: latent popularity drives both rank and arrival.

## Results (bias; full table in the script output)

| Scenario | Unw | Heur | IPE | NIPS | PACE | PACE-LOO |
|---|---|---|---|---|---|---|
| A (no confounding) | 0.000 | +0.001 | +0.002 | +0.001 | +0.002 | +0.002 |
| B (rank~sent) | 0.000 | +0.059 | −0.176 | **−0.217** | −0.016 | −0.017 |
| C (arrival~sent) | 0.000 | +0.010 | +0.014 | +0.001 | **−0.001** | −0.001 |
| D (both~sent) | 0.000 | +0.067 | −0.156 | **−0.205** | −0.018 | −0.020 |
| E (rank~arrival) | 0.000 | +0.001 | +0.001 | −0.001 | +0.002 | +0.002 |
| F (likes→rank) | 0.000 | −0.000 | +0.001 | +0.001 | +0.000 | +0.000 |
| G (latent popularity) | 0.000 | −0.001 | +0.002 | +0.002 | +0.000 | +0.000 |

## Interpretation (why each estimator behaves as it does)

1. **Unweighted is unbiased (0.000) in every scenario** — by construction, since
   the estimand is the mean of the same sentiments it averages. No weighting can
   beat it for *this* estimand.
2. **The genuine NIPS overcorrects even more than IPE under rank–sentiment
   confounding** (bias −0.217 vs −0.176 in B; −0.205 vs −0.156 in D), because
   pure inverse-rank propensity assigns the most extreme weights to low-ranked
   (here negative-sentiment) comments. The engagement term in IPE partially
   dampens this.
3. **PACE is the least-biased weighted estimator under confounding** (−0.016 in
   B, −0.018 in D) because its exposure baseline absorbs rank/arrival variation.
   It is **nearly unbiased under pure arrival confounding** (C: −0.001).
4. **PACE-LOO ≈ PACE** throughout — the LOO variant does not materially change
   results at N=80.

## Conclusion

PACE meaningfully reduces (but does not eliminate) exposure-induced bias among
*weighted* estimators, and it substantially outperforms NC-IPS under
confounding. However, it **never beats the unweighted mean** for the participant
-mean estimand. This is a defensible, honest result and does not support any
"universally superior" claim.
