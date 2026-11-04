# Empirical Reproducibility Report

Every number traceable to an executed result or source dataset. No result is
reported without a corresponding run.

## Test/verification status

| Suite | Result |
|---|---|
| `tests/test_core.py` | 13/13 pass |
| `tests/test_interrater.py` | 13/13 pass |
| `experiments/run_exhaustive_verification.py` | 22/22 pass, 0 warnings |
| `experiments/endogeneity_simulation.py` (M=500) | executed, table in report |

## Reproduction commands and outcomes

- `python experiments/reproduce_all_experiments.py` — benchmark/ablation numbers
  reproduce (verified by direct recomputation in `_audit_repro`; the in-repo
  script also writes CSVs, which require a normal shell for `data/`).
- `python experiments/sensitivity_analysis.py` — sensitivity table reproduces
  (0.049/0.082/0.108, ratio 6.221).
- `python experiments/evaluate_vader_accuracy.py` — 65.0%, macro-F1 0.5307, 46%
  zero-inflation (confirmed against `data/vader_evaluation_summary.json`).
- `python experiments/expand_simulations.py` — E1–E4 executed.
- `python tests/test_core.py` and `tests/test_interrater.py` — pass.

## Corpus validation

`data/youtube_corpus_100_comments.csv`:
- SHA-256 **matches** the provenance manifest ✅
- 100 rows, 100 unique `comment_id`, 0 NaN ✅
- 1 duplicated text (two comments share text; not an integrity error)
- 0 negative likes; `rank` unique ✅
- timestamp range 2018-11-16 → 2026-10-30 (9–17 years post-upload, consistent
  with the manuscript)
- **rank is retrieval-time relevance order, NOT historical display rank** —
  the manuscript states this; no impression/exposure measurement is claimed.

## Gap

The four-video `multivideo_*.csv` files are **not** present in `data/` (the
fetch script must be run in a normal shell). The paper's multi-video numbers are
the dated 2026-11-04 snapshot and are reproducible via
`experiments/fetch_multivideo_corpus.py` (live re-fetch may drift like counts).

## Honesty statement

No fabrication of annotations, results, citations, or significance. Negative
results (PACE not universally superior; NC-IPS overcorrection; unweighted mean
unbiased) are reported, not hidden.
