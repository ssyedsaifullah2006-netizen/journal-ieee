# CHANGELOG — research-revision (scientific corrections)

All changes are on `research-revision`; `main` and `submission-ready` are untouched.

## [Unreleased] — independent statistical correction & validation

### Planned / in progress
- [ ] Correct Krippendorff's alpha (coincidence-matrix formulation)
- [ ] Correct Fleiss' kappa (single-rating handling)
- [ ] Redesign endogeneity simulation (independent rank/arrival controls)
- [ ] HC3 + influence diagnostics for OLS inference
- [ ] NC-IPS implementation audit
- [ ] Empirical corpus/provenance validation
- [ ] Full reproduction of all experiments
- [ ] Manuscript revision + novelty audit + final verdict

### Fixed
- Corrected Krippendorff's alpha (coincidence matrix; `D_e = (N^2 - sum n_c^2)/N^2`; equals Scott's pi, not Cohen's kappa)
- Corrected Fleiss' kappa (units with <2 ratings excluded, not given perfect agreement)
- Added HC3 standard errors + influence diagnostics (leverage, Cook's D) to OLS inference
- Redesigned endogeneity simulation (`endogeneity_simulation.py`): 7 DGPs, independent rank/arrival/sentiment controls, M=500, bias/MAE/RMSE/SE/CI
- Identified NC-IPS baseline as mislabeled (inverse-propensity × log-likes, not Joachims' NC-IPS)
- Added `scotts_pi`; added `tests/test_interrater.py` (13 checks)
- Corpus SHA-256 provenance verified; rank confirmed as retrieval-time relevance order

### [latest] Genuine normalized IPS + manuscript relabeling
- Implemented `estimate_normalized_ips()` (genuine Joachims normalized clipped IPS: inverse-rank propensity, no engagement term)
- Retained the hybrid as IPE (`estimate_position_ips`, now labeled "IPE" in outputs)
- Endogeneity + expanded simulations now compare 6 estimators (Unw, Heur, IPE, NIPS, PACE, PACE-LOO)
- Key finding: genuine NIPS overcorrects more than IPE under rank–sentiment confounding (bias −0.217 vs −0.176; PACE −0.016)
- Manuscript: relabeled NC-IPS→IPE, added NIPS baseline, updated robustness subsection with endogeneity_simulation.py (M=500) results
- PDF: NOT compiled (no TeX distribution in sandbox and install is blocked); paper.tex validated structurally (QA 22/22, environment balance clean, citations resolve)
