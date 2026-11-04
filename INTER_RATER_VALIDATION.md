# Inter-Rater Validation

## Status: NO second-rater data exists

The repository contains a **single-annotator** gold standard
(`data/youtube_corpus_100_comments_human_labeled.csv`, N=100). No second rater's
labels are present, and none were fabricated. Consequently:

- **No inter-rater agreement statistic is reported** for the semantic audit.
- The accuracy figures (65.0%, κ=0.402 vs. a majority-class baseline) are
  single-annotator and remain a limitation (disclosed in the manuscript).

## Tooling (ready, not yet applied to real second-rater data)

`analyzers/interrater.py` provides, each independently implemented and validated:
- percent agreement,
- Cohen's kappa (2 raters, rater-specific marginals),
- Scott's pi (2 raters, pooled marginals),
- weighted Cohen's kappa (linear/quadratic),
- Fleiss' kappa (≥2 raters; units with <2 ratings excluded),
- Krippendorff's alpha (coincidence matrix, nominal; = Scott's pi for 2 raters).

`experiments/second_rater_protocol.md` defines a blind annotation protocol;
`experiments/generate_second_rater_template.py` emits a blind labeling sheet
(comment text only, no labels).

## Validation evidence

- alpha == Scott's pi (2-rater nominal) over 300 random datasets: max diff 3.9e-16.
- perfect agreement = 1.0; complete binary disagreement = −1.0; single-category = NaN.
- Cohen's kappa != Scott's pi on unbalanced-marginal data (classic distinction).
- Fleiss' kappa = 1.0 on perfect agreement; single-rating units excluded (no inflation).

## Recommended design for a real reliability study

- **Two (ideally three) raters**, blind to each other and to VADER scores.
- **Sample-size justification:** for a 3-class task, a 95% CI for Cohen's κ of
  width ±0.10 around κ≈0.5 requires on the order of 200–400 doubly-coded units
  (Cicchetti 1981); the current N=100 is adequate only for a coarse estimate.
- Report **percent agreement + Cohen's κ + Scott's π (or Krippendorff's α)**,
  and state the missing-data handling explicitly.
- Do **not** label the second rater automatically; any such automation would be
  fabrication and is prohibited.

## Conclusion

The single-annotator limitation is retained explicitly. No agreement statistics
are fabricated or reported. The infrastructure to compute them correctly is in
place pending real second-rater data.
