"""Self-contained regression + invariance tests for the PACE framework.

Run:  python tests/test_core.py
Exits non-zero on any failure. No external test framework required.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analyzers.consensus_framework import ConsensusEstimatorSuite, CommentRecord, ols_inference
from analyzers.interrater import cohens_kappa, fleiss_kappa, krippendorff_alpha, percent_agreement, scotts_pi

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


# --- mathematical invariants ---
def test_invariants():
    print("mathematical invariants")
    recs = [CommentRecord(str(i), f"c{i}", 0.6 if i % 2 == 0 else -0.6,
                          "positive" if i % 2 == 0 else "negative",
                          int(10 * (i + 1)), 1, i + 1, i * 0.5, 0) for i in range(30)]
    s = ConsensusEstimatorSuite(recs)
    p = s.estimate_pace()
    w = np.array(p["weights"])
    check("weights sum to 1", np.isclose(w.sum(), 1.0, atol=1e-9))
    check("weights strictly positive", np.all(w > 0))
    ess = p["effective_sample_size"]
    check("ESS in [1, N]", 1.0 <= ess <= len(recs))
    smear = p["fitted_parameters"]["duan_smearing_factor"]
    check("Duan smear >= 1", smear >= 1.0)
    inf = p["fit_diagnostics"]["inference"]
    check("inference dict present", inf is not None and "beta_se" in inf)
    check("beta SE finite", np.all(np.isfinite(inf["beta_se"])))


# --- backward compatibility: paper numbers must reproduce ---
def test_backward_compat():
    print("backward compatibility")
    df = pd.read_csv(ROOT / "data" / "youtube_corpus_100_comments.csv")
    recs = [CommentRecord(str(r["comment_id"]), str(r["text"]), float(r["sentiment_score"]),
                          str(r["sentiment_label"]), int(r["likes"]), int(r["replies"]),
                          int(r["rank"]), float(r["arrival_hours"]), 0) for _, r in df.iterrows()]
    mu = ConsensusEstimatorSuite(recs).estimate_pace()["consensus_score"]
    check("YouTube mu_pace == 0.079", abs(mu - 0.079) < 0.001)


# --- edge cases: no crashes ---
def test_edge_cases():
    print("edge cases")
    s1 = ConsensusEstimatorSuite([CommentRecord("a", "only", 0.5, "positive", 10, 0, 1, 1.0, 0)])
    check("N=1 does not crash", s1.estimate_pace()["consensus_score"] == 0.5)
    zeros = [CommentRecord(str(i), f"z{i}", 0.6, "positive", 0, 0, i + 1, i, 0) for i in range(20)]
    check("all-zero likes -> uniform", abs(ConsensusEstimatorSuite(zeros).estimate_pace()["consensus_score"] - 0.6) < 1e-9)


# --- inter-rater agreement ---
def test_interrater():
    print("inter-rater agreement")
    a = ["pos", "neg", "neu"] * 10
    check("perfect kappa == 1", cohens_kappa(a, a) == 1.0)
    codes = np.column_stack([np.array([0, 1, 2] * 10), np.array([0, 1, 2] * 10)])
    check("krippendorff == scotts_pi (2-rater nominal)", krippendorff_alpha(codes) == scotts_pi(a, a))
    check("percent agreement in [0,1]", 0.0 <= percent_agreement(a, ["pos", "neu", "neu"] * 10) <= 1.0)
    check("fleiss perfect == 1", fleiss_kappa([[0, 0, 0], [1, 1, 1], [2, 2, 2]]) == 1.0)


if __name__ == "__main__":
    test_invariants()
    test_backward_compat()
    test_edge_cases()
    test_interrater()
    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
