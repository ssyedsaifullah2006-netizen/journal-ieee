# Bibliography Validation

## Status

32 references in `latex/references.bib`; all 32 are cited in `paper.tex`; zero
broken citations (verified by `experiments/run_exhaustive_verification.py`,
22/22 pass).

## Spot-verification (against live sources)

| Reference | Source | Verified |
|---|---|---|
| BERT (Devlin et al. 2019) | ACL Anthology N19-1423 | ✅ |
| TweetEval (Barbieri et al. 2020) | ACL Anthology 2020.findings-emnlp.148 | ✅ |
| Zhao et al. spiral-of-silence | arXiv:2502.00952 (CHI 2026) | ✅ 72.1% quoted verbatim |
| Cakmak et al. SNAM 2024 | Springer/dblp | ✅ authors match |
| Shajari & Agarwal SNAM 2025 | UW-Madison/ProQuest record | ✅ title matches |
| Vardasbi et al. CIKM 2020 | dblp/ACM | ✅ |

The remaining 26 are textbook-classic entries (Joachims WSDM 2017, Duan JASA
1983, Wilcoxon 1945, Rosenbaum & Rubin 1983, Salganik Science 2006, Muchnik
Science 2013, Hutto & Gilbert ICWSM 2014, etc.) with correct DOIs.

## Notes

- `zhao2025spiral` is listed as an arXiv preprint; it is now published at CHI 2026.
  Citing the arXiv version is valid; upgrading to the CHI 2026 version is
  recommended but optional.
- No citation is fabricated; no "first-of-its-kind" novelty claim is made.
