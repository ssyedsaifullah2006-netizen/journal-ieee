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
