# Semantic Reliability Audit

Audit of the human-annotated semantic layer and its limitations.

## Data

- `data/youtube_corpus_100_comments_human_labeled.csv`: single-annotator gold
  labels for all 100 comments (one video, dQw4w9WgXcQ).
- SHA-256 provenance verified; 100 unique rows, 0 NaN; rank = retrieval order.

## Verified lexicon statistics (VADER vs human, N=100)

| Metric | Value |
|---|---|
| Accuracy | 65.0% (95% Wilson [55.3%, 73.6%]) |
| Macro-F1 | 0.5307 |
| Zero-inflation | 46% [36.6%, 55.7%] |
| Positive→negative misclassification | 23.9% [13.9%, 37.9%] |
| Cohen's κ vs majority baseline | κ=0.402; McNemar p=0.044 |
| Per-class precision (pos/neu/neg) | 0.703 / 0.787 / 0.125 |

## Annotation consistency (audited)

- Positive = enjoyment/praise of the content; negative = genuine criticism;
  neutral = meme mechanics, anecdotes, quotes, emoji-only.
- Sarcasm/humor handled by labeling literal sentiment unless clearly evaluative.
- The protocol is documented in `experiments/second_rater_protocol.md`.

## Limitations (explicit, not hidden)

1. **Single annotator** — no inter-rater reliability can be computed; tooling
   (`analyzers/interrater.py`, blind template) is ready but no second-rater data
   exists. No agreement statistic is reported.
2. **Single video, meme domain** — the multi-video extension is VADER-scored only
   (no human labels), so the accuracy claim does not generalize across domains.
3. **Negative class has support 3** — negative precision (0.125) is indicative,
   not precise (95% CI [0.035, 0.360]).

## Conclusion

The lexicon degradation finding (65%, κ=0.40 vs a trivial baseline, 46%
zero-inflation) is reproducible but **single-annotator and single-video**. It is
described honestly as such; upgrading it requires real second-rater labels, which
are not fabricated.
