# Pre-Submission Audit: PACE Manuscript

**Status: manuscript, code and data are consistent and reproducible. Five author-side items (below) must be completed before upload.** Nothing here is a guarantee of acceptance; it records what was checked and what was not.

## Verified in this repository

- **Numbers reproduce.** `reproduce_all_experiments.py`, `sensitivity_analysis.py` and `evaluate_vader_accuracy.py` were re-run from a clean copy; every CSV/JSON in `data/` was byte-identical to the committed version.
- **Automated checks.** `experiments/run_exhaustive_verification.py` runs 23 checks. In an environment without the NLTK VADER lexicon one check (re-scoring the corpus with VADER) cannot run and reports a warning; with the lexicon installed it should pass.
- **Corpus provenance.** Retrieved via YouTube Data API v3 on 2026-10-30 (07:09 UTC); SHA-256 manifest in `data/youtube_corpus_100_comments.provenance.json`. Commenter display names were pseudonymized afterwards (`experiments/pseudonymize_authors.py`); the manifest records both the original and the post-processing hash and says so.
- **Claims match the evidence.** The paper no longer claims universal superiority, a fixed "90.7%" error reduction (it depends on the choice of target Y*), rank-order invariance of PACE, or a "gold-standard" annotation. Scenario-generation limits, ranking endogeneity, lexicon failure and single-annotator labels are disclosed.
- **Bibliography.** Eight entries were corrected against publisher/index records (wrong authors, venue, pages or DOI; one entry, Vardasbi et al., pointed to a different paper than the one cited, and the sentence describing it was rewritten). See the table below.

## Reference verification detail

| Status | Entries |
|---|---|
| Title, authors, venue, year and DOI matched to a publisher/index record I read | joachims2017unbiased, zubiaga2016stance (no DOI exists; ACL URL is inferred), cakmak2024bias, shajari2025safeguarding, wang2018position, agarwal2019addressing, vardasbi2020when, craswell2008experimental, tsagkias2009predicting, thelwall2018social, thelwall2010sentiment, severyn2016multilingual, horvitz1952generalization, muchnik2013social, noelle1974spiral, hutto2014vader, pontiki2016semeval, zhao2025spiral, davison1997bootstrap, huber1981robust, hampel1986robust |
| Matched on title/authors/venue but **DOI or a field not independently resolved** (click each before submission) | duan1983smearing (volume/issue/pages from a secondary list), wu2018beyond (pages), borghol2012untold (DOI), rosenbaum1983central (DOI), salganik2006experimental (issue number), gomez2008statistical (DOI, pages), ernst2004permutation (DOI), wilcoxon1945individual (DOI, issue), rousseeuw1993alternatives (JSTOR DOI 10.2307/2291267 taken from the repository record; issue number) |

Crossref and doi.org were not reachable from the review environment, so no DOI was resolved programmatically and no retraction database was queried.

## Author-side items still open

1. **Second annotator.** The 100 reference labels come from one annotator (disclosed in the paper). Have another person label `data/annotation_sheet_second_rater.csv` following `ANNOTATION_GUIDELINES.md`, run `python experiments/compute_agreement.py`, and add Cohen's kappa to Section V-E and the matching limitation. Only 3 of 100 comments are labeled negative, so negative-class precision (12.5%) rests on three comments.
2. **Resolve the nine bibliography fields above** (click the DOIs / check the issue and page numbers).
3. **Compile in Overleaf** with the official `IEEEtran.cls`/`IEEEtran.bst` and check page count and layout against the target journal's current author guidelines.
4. **IEEE policy items:** disclose any use of AI tools for text or code in the Acknowledgments if applicable; add funding/conflict-of-interest statements if the journal requires them.
5. **Data release.** Confirm that redistributing comment text complies with the YouTube API Services Terms. Raw commenter handles remain in earlier git commits; to remove them, rewrite history (e.g. `git filter-repo`) or publish the data from a fresh repository.

## Reproduce

```bash
python experiments/reproduce_all_experiments.py
python experiments/sensitivity_analysis.py
python experiments/evaluate_vader_accuracy.py
python experiments/generate_publication_figures.py
python experiments/run_exhaustive_verification.py
```
