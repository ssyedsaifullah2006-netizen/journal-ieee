# Sentiment annotation protocol (YouTube corpus, N = 100)

Label each comment with exactly one of `positive`, `neutral`, `negative`, describing the
commenter's stance toward the song/video (not toward people or events in the story they tell).

- **positive** — enjoyment or praise of the song/video/Rick Astley (including slang such as
  "banging", "fire", "kills"), nostalgia or affection, or enthusiastic celebration of a milestone.
- **negative** — genuine frustration, complaint, criticism of the song/video, or negative
  self-feeling about the experience.
- **neutral** — factual statements, descriptions of the "rickroll" meme mechanics, link-sharing
  anecdotes, questions, non-evaluative quotes, and non-text/emoji-only placeholders.

Rules for the second rater: label from the text alone, independently, without discussing
labels with the first annotator, and without running VADER. Do not look at
`youtube_corpus_100_comments_human_labeled.csv` until you have finished.

Workflow:

1. Second rater fills the `label` column of `data/annotation_sheet_second_rater.csv`.
2. Run `python experiments/compute_agreement.py` to obtain Cohen's kappa, raw agreement, the
   disagreement list, and VADER accuracy against each rater.
3. Report kappa in the paper (Section V-E and the Limitations item on reference labels), and
   resolve disagreements by discussion if a consensus label set is wanted.
