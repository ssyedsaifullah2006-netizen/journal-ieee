# Pre-Submission Evidence Gate: PACE Manuscript

**Current status: NOT READY FOR JOURNAL SUBMISSION.**

This document is an evidence gate, not a promotional report. A script passing
structural checks does not establish corpus provenance, causal exposure
identification, real-world accuracy, or publisher-verified bibliography.

## What is currently reproducible

- The synthetic generators, deterministic seeds, trial-level CSV tables, and
  ablation table reproduce from `experiments/reproduce_all_experiments.py`.
- The code dynamically fits the log-linear model; it does not contain the
  earlier hard-coded regression coefficients.
- The manuscript contains an algorithm block and its cited BibTeX keys resolve
  to entries in `latex/references.bib`.
- The stored YouTube snapshot has 100 rows, no missing cells, and VADER scores
  that can be recomputed from its text.

## Evidence gaps that block the previous submission claim

1. **Corpus provenance is unverified.** The legacy CSV has a recorded hash but
   no contemporaneous retrieval timestamp or raw API collection record. It is
   illustrative only. Run `experiments/fetch_real_youtube_corpus.py` with a
   valid API key to create a dated provenance manifest before claiming an
   official API collection.
2. **The retrieved rank is not historical exposure.** API relevance order at
   retrieval time cannot reconstruct impressions or the display position under
   which likes accumulated. The paper must not present PACE as causal exposure
   correction on this data.
3. **The real-world snapshot has no consensus ground truth.** It can support
   descriptive semantic and weight diagnostics, not accuracy or causal claims.
4. **Current fit calibration is poor.** The verifier reports the ratio of
   in-sample expected to observed total likes. Until calibration and holdout
   diagnostics are satisfactory, the snapshot cannot support a calibrated
   engagement-model claim.
5. **Synthetic simulations are not external validation.** Their fixed ground
   truths are useful for controlled behavior checks only. PACE is not uniformly
   best in the supplied tables.
6. **Bibliography quality still needs publisher-level review.** The local check
   only detects missing keys, not incorrect metadata, retractions, or DOI
   resolution.

## Required release gate

Run:

```bash
python experiments/run_exhaustive_verification.py
python experiments/reproduce_all_experiments.py
```

The first command intentionally exits non-zero when warnings remain. Submission
requires: a regenerated dated corpus manifest, documented API collection,
holdout/calibration analysis, an independently labelled semantic benchmark, a
manual publisher-record bibliography review, and a successfully compiled final
PDF.
