# Second-Rater Annotation Protocol (PACE Semantic Audit)

This protocol defines how a **second, independent human rater** labels the
YouTube corpus so inter-rater reliability (Cohen's κ, weighted κ, Krippendorff's
α) can be computed. It exists to enable a valid reliability study; it does **not**
substitute for the actual annotation.

## Why this matters

The current semantic audit is **single-annotator** (`data/youtube_corpus_100_comments_human_labeled.csv`).
A single rater cannot establish reliability. To upgrade the audit from "case study"
to "reliability-validated," a second rater must label the **same** comments
**blind** (without seeing the first rater's labels or the VADER scores).

## Procedure

1. Generate a blind labeling sheet (comment text only, no labels):
   ```bash
   python experiments/generate_second_rater_template.py
   ```
   This writes `data/second_rater_template.csv` with columns
   `comment_id, text, rater2_label` (the label column empty).

2. The second rater fills `rater2_label` for every row using the rules below,
   **without** consulting the first rater, the VADER output, or any model output.

3. Compute agreement against the first rater's labels:
   ```bash
   python -c "import pandas as pd; from analyzers.interrater import agreement_report; \
   d=pd.read_csv('data/youtube_corpus_100_comments_human_labeled.csv'); \
   t=pd.read_csv('data/second_rater_completed.csv'); \
   print(agreement_report(d['human_label'], t['rater2_label'], ['positive','neutral','negative']))"
   ```

## Annotation rules (3-class)

- **positive** — genuine enjoyment, praise, admiration, affection, or endorsement
  of the song/video/content (including slang praise: *"banging"*, *"fire"*,
  *"so sick"*, *"his voice kills"*, *"masterpiece"*, *"staple"*).
- **negative** — genuine complaint, criticism, frustration, or negative
  self-feeling directed at the content or its presentation.
- **neutral** — factual statements, meme mechanics/descriptions, prank
  anecdotes ("I rickrolled my class"), link/QR-code stories, lyric quotes,
  non-evaluative observations, or unintelligible/emoji-only text.

Guidelines:
- Judge **what the comment expresses**, not whether you agree with it.
- Sarcasm/irony: label the literal sentiment unless clearly evaluative.
- Emoji-only or placeholder text → `neutral`.
- When genuinely uncertain, choose `neutral` and add a note.

## Interpretation thresholds (Landis & Koch, 1977)

| Cohen's κ | Interpretation |
|---|---|
| < 0.0 | Poor |
| 0.00–0.20 | Slight |
| 0.21–0.40 | Fair |
| 0.41–0.60 | Moderate |
| 0.61–0.80 | Substantial |
| 0.81–1.00 | Almost perfect |

Report κ (nominal agreement) and weighted κ (ordinal) alongside percent
agreement. Do **not** report a single-annotator accuracy as a reliability
estimate.
