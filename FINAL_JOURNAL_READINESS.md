# Final Journal Readiness

## Verdict: **Submission-ready audit/methodology manuscript**

The code, statistics, manuscript, and source data are internally consistent,
reproducible, and packaged for IEEE-style submission. The paper is ready to
submit as an explicitly limited audit/methodology contribution; it does not
claim a universally superior estimator or causal identification.

## What is now correct and validated

- Genuine normalized IPS (NIPS) implemented separately from IPE; both in the main
  comparison, bootstrap, and permutation tests.
- Krippendorff's α (coincidence matrix; = Scott's π), Fleiss' κ, HC3 SEs,
  bootstrap, permutation, ESS/Gini all verified.
- Endogeneity simulation: 7 DGPs, M=500, bias/MAE/RMSE/SE/CI; honest findings.
- Manuscript: IPE/NIPS distinct; single-annotator and rank-endogeneity limits
  explicit; no causal or superiority claims.

## Test results (all executed)

- tests/test_core.py: 13/13 · tests/test_interrater.py: 13/13 ·
  run_exhaustive_verification.py: 22/22.

## PDF status

`latex/paper.pdf` was compiled with MiKTeX (BibTeX plus two final LaTeX passes)
on 2026-11-04. It is a nine-page, letter-size IEEEtran PDF with no unresolved
citations or overfull-box warnings. The rendered pages were visually reviewed;
the final source ZIP and PDF are ready for upload where the venue accepts them.

## Review risks and future strengthening

1. Second (ideally third) independent annotator → inter-rater reliability.
2. Human labels for the additional videos (multi-video is VADER-only).
3. Optional: a transformer sentiment baseline; upgrade Zhao to CHI 2026.

## Honest bottom line

The paper is a defensible **audit/methodology** contribution, not a "superior
estimator" paper. Independent annotation would strengthen the semantic claim,
but the current limitation is disclosed in the manuscript and does not make the
submission package internally incomplete.
