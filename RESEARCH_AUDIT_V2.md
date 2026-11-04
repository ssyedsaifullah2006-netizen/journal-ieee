# Research Audit V2 (research-revision)

Index of the independent correction-and-validation pass. Detailed evidence is in
the companion documents.

## Corrections (with validation)

1. **Krippendorff's alpha** — `analyzers/interrater.py`. Corrected the
   coincidence-matrix `D_e` term (`N(N-1)` → `N²`) and removed the false
   "α == Cohen's kappa" shortcut. α now equals Scott's pi for 2 raters (verified
   to 3.9e-16 over 300 datasets). See `STATISTICAL_CORRECTIONS.md`,
   `INTER_RATER_VALIDATION.md`.
2. **Fleiss' kappa** — units with <2 ratings excluded (no longer given perfect
   agreement). Same docs.
3. **OLS inference** — HC3 SEs, leverage, Cook's D added. `PACE_MATHEMATICAL_VALIDATION.md`.
4. **Endogeneity simulation** — `experiments/endogeneity_simulation.py` (7 DGPs,
   independent controls, M=500). `ENDOGENEITY_SIMULATION_REPORT.md`.
5. **NC-IPS relabeling** — identified as inverse-propensity × log-likes, not
   Joachims' NC-IPS. `BASELINE_COMPARISON_REPORT.md`.
6. **Corpus validation** — SHA-256 match, 100 unique rows, rank is retrieval
   order (not historical display). `EMPIRICAL_REPRODUCIBILITY_REPORT.md`.

## Key honest findings

- Unweighted mean is unbiased (0.000) for the participant-mean estimand in all
  seven endogeneity scenarios; no weighting beats it.
- NC-IPS-style weighting overcorrects under rank–sentiment confounding
  (bias −0.176 vs PACE −0.016).
- PACE is nearly unbiased under pure arrival confounding (bias −0.001).
- PACE's sigmoid is **not** formally bounded-influence (normalized w_max can →1).

## Status

Corrected, validated, honest. Remaining gap: a real second annotator (no data
fabrication permitted). Final verdict: **B — improved, not submission-ready**
(see `JOURNAL_READINESS_V2.md`).
