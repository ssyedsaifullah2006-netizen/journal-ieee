# Journal-Readiness Assessment (research-revision branch)

Prepared as an independent reviewer would assess the PACE manuscript. This is an
honest assessment, not a marketing statement.

## 1. What is solid (verified, not asserted)

- **Reproducibility.** Every reported number (benchmark, ablation, sensitivity,
  VADER audit, chance-corrected statistics, multi-video table) was recomputed
  directly from the code and datasets. Deterministic seeds + SHA-256 provenance.
  QA verifier: 22/22. New regression suite: 13/13.
- **Citations.** All 32 references are real (verified against ACL Anthology,
  arXiv, Springer, and Google Scholar); zero fabricated citations. The "72.1%"
  spiral-of-silence statistic is quoted verbatim from Zhao et al.
- **Honesty.** The manuscript reports negative results (PACE is not universally
  superior; all four estimators are indistinguishable on real data, p = 0.366;
  VADER is only marginally above a majority-class baseline, κ = 0.402).
- **Statistical hygiene.** Chance-corrected baselines (McNemar, Cohen's κ),
  Wilson/bootstrap confidence intervals, a threats-to-validity section, and the
  expanded robustness simulations added in this revision.

## 2. Novelty / prior-art (do not overclaim)

- The individual components — position-bias correction (Joachims' IPS), robust
  M-estimation (Huber), Duan's smearing — are established. PACE's novelty is the
  **specific combination** (log-linear rank+arrival exposure surface + smearing +
  MAD-studentized sigmoid residual gating + tree-depth damping) applied to
  *continuous sentiment consensus* rather than click/rank prediction.
- This is an **incremental methodological contribution**, not a new theory. The
  more valuable and defensible contribution is the **empirical audit**:
  (a) exposure-conditioning is sensitive and not universally superior, and
  (b) rule-based lexicons degrade sharply on informal discourse.
- Recommendation: position the paper as a **reproducibility + audit methodology
  paper**, which is a legitimate and increasingly valued genre. Do not position
  PACE as a "better estimator."

## 3. Remaining gaps (explicitly identified, not hidden)

1. **No inter-rater reliability.** The human-annotated accuracy audit is
   single-annotator. A second (ideally third) independent rater is REQUIRED to
   compute Cohen's κ / Krippendorff's α between human raters. Tooling is ready
   (`analyzers/interrater.py`, `experiments/second_rater_protocol.md`) but the
   data does not exist yet.
2. **Single-video accuracy audit.** The 65%-accuracy/κ=0.402 claim rests on one
   meme-domain video (N=100). The multi-video extension is VADER-scored only.
3. **Conditional advantage.** PACE helps in a pre-specified late-groundswell
   scenario but degrades under heavy-tailed engagement (E1), is not unbiased
   under ranking endogeneity (E2), and is not robust to sentiment label noise
   (E3). These are now documented in the manuscript.
4. **No neural baseline.** A transformer-based sentiment model is discussed but
   not run. A reviewer may request it; it requires compute and would change the
   semantic-audit framing (lexicon → model comparison).

## 4. Verdict

**Suitable for submission to a reputable venue as an audit/methodology paper** —
e.g., a computational-social-science or reproducibility-focused venue, or IEEE
TCSS as a methods paper — **provided** items 1–2 are addressed (a second rater)
or the single-annotator limitation is accepted and the framing stays explicitly
"audit." The framework **does not support** a "PACE is superior" claim, and the
manuscript correctly avoids one.

**Not yet "100% publishable"** in the sense of guaranteed acceptance — no paper
is — but it is internally consistent, reproducible, and honestly framed. The
single highest-leverage next step is collecting a second rater's labels.

## 5. Further research required (if the framework must stand on stronger legs)

- Second/third independent annotators → inter-rater reliability.
- Multi-video human annotation → extend the accuracy audit beyond one domain.
- A transformer-based sentiment baseline → situate VADER's failure modes.
- Theoretical analysis of the sigmoid-gated estimator (bias/variance/influence
  function) if a methods contribution is claimed.
