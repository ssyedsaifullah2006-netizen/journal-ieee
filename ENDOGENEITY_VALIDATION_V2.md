# Endogeneity Validation V2

`experiments/endogeneity_simulation.py` — seven DGPs, independent
rank/arrival/sentiment controls, M=500, N=80. Target estimand declared before
running: the unweighted mean of true sentiment.

## Scenarios

A rank ⊥ sentiment, arrival · B rank~sentiment · C arrival~sentiment ·
D rank&arrival~sentiment · E rank~arrival ⊥ sentiment · F likes→rank ·
G latent popularity → rank & arrival.

## Results (bias, M=500)

| Scenario | Unw | Heur | IPE | NIPS | PACE | PACE-LOO |
|---|---|---|---|---|---|---|
| A | 0.000 | +0.001 | +0.002 | +0.001 | +0.002 | +0.002 |
| B | 0.000 | +0.059 | −0.176 | **−0.217** | −0.016 | −0.017 |
| C | 0.000 | +0.010 | +0.014 | +0.001 | **−0.001** | −0.001 |
| D | 0.000 | +0.067 | −0.156 | **−0.205** | −0.018 | −0.020 |
| E | 0.000 | +0.001 | +0.001 | −0.001 | +0.002 | +0.002 |
| F | 0.000 | −0.000 | +0.001 | +0.001 | +0.000 | +0.000 |
| G | 0.000 | −0.001 | +0.002 | +0.002 | +0.000 | +0.000 |

## Interpretation

- Unweighted unbiased (0.000) everywhere — the reference estimator for this
  estimand.
- Genuine NIPS overcorrects more than IPE under rank–sentiment confounding
  (B: −0.217 vs −0.176; D: −0.205 vs −0.156).
- PACE least-biased weighted estimator (B: −0.016, D: −0.018) and nearly unbiased
  under pure arrival confounding (C: −0.001).
- PACE-LOO ≈ PACE throughout.

Full per-scenario MAE/RMSE/SE/95% CI are in the script's stdout (reproducible).
The implementation matches this report; see also `ENDOGENEITY_SIMULATION_REPORT.md`.
