# PACE Claims Audit

Audit of every claim PACE's manuscript could make, versus what the evidence
supports. Scientific correctness over appearance.

## Claims that are SUPPORTED (with evidence)

1. PACE is a well-defined weighted consensus estimator (weights normalize to 1,
   ESS∈[1,N], strictly positive weights). — proven in tests.
2. PACE reduces outlier weight share relative to the heuristic (0.057→0.041 in
   Scenario 1). — reproduced.
3. PACE is the **least biased weighted estimator** under rank/arrival–sentiment
   confounding (bias −0.016 vs −0.176 IPE, −0.217 NIPS). — endogeneity sim, M=500.
4. PACE is **nearly unbiased under pure arrival confounding** (bias −0.001). —
   scenario C.
5. PACE outperforms the heuristic in a pre-specified late-groundswell scenario
   (MAE 0.010 vs 0.107). — reproduced.
6. PACE is robust to LOO vs in-sample smearing (score 0.0792 both). — verified.

## Claims that are NOT supported (must not be made)

1. **"PACE is universally superior."** False — unweighted wins Scenarios 1/3 and
   is unbiased (0.000) for the participant-mean estimand in all 7 endogeneity
   scenarios.
2. **"PACE is bounded-influence."** False in the formal sense — normalized w_max
   can →1 despite the sigmoid. The manuscript states this correctly.
3. **"PACE corrects for exposure causally."** False — rank is retrieval-time
   relevance (endogenous). No causal identification.
4. **"PACE is robust to sentiment measurement error."** False — under 10–30%
   label flips, IPE/NIPS are more robust than PACE.
5. **"PACE's advantage is distribution-robust."** False — the MAE advantage over
   the heuristic shrinks from 0.012 (uniform) to 0.057 (Pareto).

## Verdict

PACE's defensible claim is: **a conditional observational diagnostic that
reduces (but does not eliminate) exposure bias among weighted estimators.**
All manuscript text is consistent with this. No overclaim is present.
