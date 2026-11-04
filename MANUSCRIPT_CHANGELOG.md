# Manuscript Changelog (paper.tex)

Cumulative changes to `latex/paper.tex` across the research-revision pass.

## Reframing (semantic co-lead)

- Title → "Semantic Reliability and Exposure-Conditioned Consensus Estimation…".
- Abstract rewritten to present two layers (semantic + exposure) co-equally.
- Keywords add "Lexicon Reliability", "Semantic Analysis".
- Added "Semantic Reliability Audit" as the first contribution.

## Statistical rigor

- Added per-class Table III (precision/recall/F1) for VADER vs human.
- Added chance-corrected stats (McNemar p=0.044, κ=0.402, Wilson CIs).
- Renamed "Limitations" → "Limitations and Threats to Validity"; added a
  conclusion-validity item on the marginal advantage over a trivial baseline.
- Added "Rule-Based Lexicons versus Neural Sentiment Models" related work
  (+2 references → 32 total).

## Observational extension

- Added "Multi-Video Observational Extension" (Table IV, four videos).
- Added "Synthetic Robustness Across Distributions, Ranking Bias, and Measurement
  Error" citing the redesigned endogeneity simulation (M=500, 7 DGPs).

## Baseline correction (this pass)

- "NC-IPS" relabeled to **IPE** (inverse-propensity engagement-weighted) throughout.
- Added **NIPS** (genuine normalized clipped IPS) as a separate baseline.
- Table I gains a NIPS column (NIPS MAE 0.007/0.100/0.029/0.123).
- Baselines bullet now explicitly documents that IPE is a hybrid, that NIPS is
  the genuine Joachims estimator, and that retrieval rank is not historical
  exposure/impression probability.

## Author metadata

- Authors: Syed Saifullah S and Harish S.
- Email: syedsaifullah.s2024@vitstudent.ac.in.
- Affiliation: School of Computer Science Engineering and Information Systems
  (SCORE), VIT, Vellore 632014.

## Honesty guarantees (unchanged)

- No universal PACE superiority claim. No causal identification claim.
- Single-annotator and four-video (VADER-only) limitations explicit.
- Negative findings (NIPS/IPE overcorrection; unweighted unbiased) retained.
