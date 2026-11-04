# Final Research Audit (research-revision)

Consolidated audit of the PACE framework after the IPS/IPE correction.

## Corrections applied (this pass)

1. **IPS/IPE separation.** `estimate_normalized_ips()` is the genuine Joachims
   normalized clipped IPS (inverse-rank propensity, no engagement term).
   `estimate_position_ips()` is retained as IPE (inverse-propensity × log1p(likes)),
   explicitly labeled IPE. `run_all_and_compare()` and
   `compute_full_pipeline_bootstrap_ci()` now use the correct estimator per method
   and include all five methods (Unweighted, Heuristic, IPE, NIPS, PACE).
2. **Main benchmark** (`reproduce_all_experiments.py`) now computes NIPS and
   reports IPE/NIPS separately in the summary table and trial CSV.
3. **Manuscript** Table I and text distinguish IPE and NIPS throughout.

## Key numerical results (reproduced, M=50 / N=100)

| Scenario | Unw | Heur | IPE | NIPS | PACE |
|---|---|---|---|---|---|
| 1. Viral Outlier | 0.030 | 0.086 | 0.008 | 0.007 | 0.061 |
| 2. Late Groundswell | 0.075 | 0.107 | 0.103 | 0.100 | **0.010** |
| 3. Democratic | **0.000** | 0.009 | 0.024 | 0.029 | 0.021 |
| YouTube (μ̂) | 0.114 | 0.123 | 0.116 | 0.123 | 0.079 |

NIPS and IPE are similar (both inverse-propensity based); NIPS is slightly better
in Scenarios 1–2 but worse in Scenario 3. Neither reproduces the genuine Joachims
estimator until now; the manuscript now labels them correctly.

## Honest conclusions (unchanged, reinforced)

- Unweighted is unbiased for the participant-mean estimand (bias 0.000 across 7
  endogeneity scenarios).
- Under rank–sentiment confounding, NIPS overcorrects more than IPE (bias −0.217
  vs −0.176), while PACE is least biased (−0.016).
- PACE is a conditional observational diagnostic, not a universally superior
  estimator. No causal identification is claimed.
