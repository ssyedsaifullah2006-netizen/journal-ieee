# PACE IEEE Journal Submission Package

This folder contains the complete, compilable IEEE journal manuscript package for:
**"Exposure-Conditioned Engagement Adjustment for Robust Consensus Estimation in Ranked Social Discussion Streams"**
Author: **Sayed Saifullah** (Vellore Institute of Technology, India)

---

## Files Included

- `paper.tex`: Complete IEEE Transactions LaTeX source document in two-column format.
- `references.bib`: BibTeX bibliography. Verify each bibliographic record against its publisher before submission; presence in this file is not publisher validation.
- `figures/`: High-resolution (300 DPI) publication-grade PNG figures:
  - `fig1_monte_carlo_mae.png`: Monte Carlo estimation error across $M=50$ trials.
  - `fig2_ablation_study.png`: Component ablation results across $M=50$ trials.
  - `fig3_weight_distributions.png`: Weight shares comparing Heuristic vs. PACE in Scenarios 1 & 2.
  - `fig4_youtube_case_study.png`: Real-world YouTube case study fit and weight dispersion.

---

## Instructions for Submission

### Option 1: Overleaf (Recommended)
1. Zip the entire `latex/` folder (or select `paper.tex`, `references.bib`, and the `figures/` folder).
2. Go to [Overleaf](https://www.overleaf.com), click **New Project** $\to$ **Upload Project**, and upload the zip file.
3. Overleaf will automatically detect `IEEEtran.cls` (built into Overleaf) and compile `paper.tex` into a publication-ready PDF.

### Option 2: Local Compilation (TeX Live / MiKTeX)
Run the standard LaTeX compilation sequence from within the `latex` directory:
```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

---

## Verification & Reproducibility Limits
To regenerate all empirical numbers and figures from scratch:
```bash
# 1. Run all 50 Monte Carlo trials per scenario
python experiments/reproduce_all_experiments.py

# 2. Regenerate 300 DPI publication figures
python experiments/generate_publication_figures.py
```
All trial-level data are saved in `data/monte_carlo_trial_level_results.csv` and `data/ablation_trial_level_results.csv`.

The included YouTube CSV (`youtube_corpus_100_comments.csv`) is a static 100-comment snapshot retrieved via the YouTube Data API v3 on 2026-10-30 with a contemporaneous SHA-256 provenance manifest (`youtube_corpus_100_comments.provenance.json`). It provides an illustrative semantic case study and is accompanied by a gold-standard human-annotated sentiment audit (`youtube_corpus_100_comments_human_labeled.csv`). However, it does not establish historical display rank, user impressions, or causal exposure, and its retrieval-time relevance order serves as an exposure proxy rather than a causal instrument.
