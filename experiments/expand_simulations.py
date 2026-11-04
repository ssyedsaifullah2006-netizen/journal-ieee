"""
Expanded robustness simulations for the PACE framework.

Adds four scenario families beyond the original three:
  E1  engagement-distribution sensitivity (uniform / lognormal / power-law / Pareto)
  E2  ranking bias (endogeneity): rank is correlated with sentiment
  E3  sentiment measurement error (label flips and continuous noise)
  E4  edge cases: small N, all-zero likes, outlier at extreme ranks

Each run is seed-deterministic. Every estimator (Unweighted, Heuristic,
NC-IPS, PACE, and the PACE-LOO diagnostic variant) is evaluated against a
declared ground-truth Y*, reporting Mean MAE +/- SE over M trials. The point is
robustness, not superiority: negative results are reported, never hidden.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from analyzers.consensus_framework import ConsensusEstimatorSuite, CommentRecord


def _estimators(suite):
    u = suite.estimate_unweighted()["consensus_score"]
    h = suite.estimate_heuristic_log_likes()["consensus_score"]
    i = suite.estimate_position_ips()["consensus_score"]
    p = suite.estimate_pace()["consensus_score"]
    l = suite.estimate_pace_loo()["consensus_score"]
    return u, h, i, p, l


# ----------------------------------------------------------------------------
# E1: engagement-distribution sensitivity (late-groundswell structure, Y*=-0.1)
# ----------------------------------------------------------------------------
def _scenario2_like(seed, like_kind, n=60):
    """Scenario-2 comment structure with a chosen like distribution."""
    rng = np.random.default_rng(seed)
    recs = []
    # 10 early hype (positive)
    for i in range(10):
        s = float(np.clip(rng.normal(0.70, 0.05), -1, 1))
        recs.append(CommentRecord(f"h{i}", f"hype {i}", s, "positive",
                                  _draw_likes(like_kind, rng, 150, 350, i), 5, i + 1, 0.1 + i * 0.2, 0))
    # 24 casual neutral
    for i in range(10, 34):
        recs.append(CommentRecord(f"m{i}", f"casual {i}", 0.10, "neutral",
                                  _draw_likes(like_kind, rng, 10, 40, i), 1, i + 1, 2.0 + i * 0.5, 0))
    # late critical groundswell (rank 35)
    recs.append(CommentRecord("gw", "critical issue", -0.85, "negative", 800, 45, 35, 24.0, 0))
    # 25 late agreeing
    for i in range(35, 60):
        s = float(np.clip(rng.normal(-0.40, 0.05), -1, 1))
        recs.append(CommentRecord(f"l{i}", f"agree {i}", s, "negative",
                                  _draw_likes(like_kind, rng, 20, 60, i), 2, i + 1, 25.0 + (i - 35) * 0.3, 0))
    return recs


def _draw_likes(kind, rng, lo, hi, idx):
    if kind == "uniform":
        return int(rng.integers(lo, hi))
    if kind == "lognormal":
        mu = np.log((lo + hi) / 2)
        return int(np.clip(rng.lognormal(mu, 0.7), 1, 2000))
    if kind == "powerlaw":
        return int(np.clip((rng.pareto(2.0) + 1) * lo, 1, 5000))
    if kind == "pareto":
        return int(np.clip(rng.pareto(3.0) * lo, 1, 10000))
    raise ValueError(kind)


def run_e1(M=30):
    print("\n[E1] Engagement-distribution sensitivity (late groundswell, Y*=-0.1)")
    kinds = ["uniform", "lognormal", "powerlaw", "pareto"]
    for kind in kinds:
        rows = np.zeros((M, 5))
        for m in range(M):
            recs = _scenario2_like(1000 + m, kind)
            rows[m] = [abs(e - (-0.1)) for e in _estimators(ConsensusEstimatorSuite(recs))]
        mean = rows.mean(0)
        se = rows.std(0) / np.sqrt(M)
        names = ["Unw", "Heur", "IPS", "PACE", "PACE-LOO"]
        line = "  ".join(f"{n}={mean[i]:.3f}±{se[i]:.3f}" for i, n in enumerate(names))
        print(f"  {kind:<9}: {line}")


# ----------------------------------------------------------------------------
# E2: ranking bias (endogeneity) — rank is positively correlated with sentiment
# ----------------------------------------------------------------------------
def run_e2(M=30):
    print("\n[E2] Ranking bias: positive comments sorted to the top (Y*=true mean)")
    for corr in [0.0, 0.5, 0.9]:
        rows = np.zeros((M, 5))
        for m in range(M):
            rng = np.random.default_rng(2000 + m)
            n = 80
            true_s = rng.normal(0.1, 0.5, n)
            true_s = np.clip(true_s, -1, 1)
            if corr > 0:
                # rank = order of (sentiment + noise), so rank correlates with sentiment
                order = np.argsort(-(true_s + rng.normal(0, (1 - corr) * 0.6 + 0.01, n)))
            else:
                order = rng.permutation(n)
            ranks = np.empty(n, int)
            ranks[order] = np.arange(1, n + 1)
            # likes decay with rank (platform-like), independent of sentiment
            likes = np.maximum(1, (2000 * ranks ** -0.8)).astype(int) + rng.integers(0, 20, n)
            recs = [CommentRecord(f"c{i}", f"t{i}", float(true_s[i]),
                                  "positive" if true_s[i] > 0.05 else ("negative" if true_s[i] < -0.05 else "neutral"),
                                  int(likes[i]), 0, int(ranks[i]), float(ranks[i]) * 0.2, 0)
                    for i in range(n)]
            y_star = float(true_s.mean())
            rows[m] = [abs(e - y_star) for e in _estimators(ConsensusEstimatorSuite(recs))]
        mean = rows.mean(0); se = rows.std(0) / np.sqrt(M)
        line = "  ".join(f"{n}={mean[i]:.3f}±{se[i]:.3f}" for i, n in enumerate(["Unw", "Heur", "IPS", "PACE", "PACE-LOO"]))
        print(f"  corr={corr:.1f}: {line}")


# ----------------------------------------------------------------------------
# E3: sentiment measurement error
# ----------------------------------------------------------------------------
def run_e3(M=30):
    print("\n[E3] Sentiment measurement error (Scenario-2, Y*=-0.1)")
    from experiments.reproduce_all_experiments import generate_scenario_2_trial
    for flip in [0.0, 0.1, 0.3, 0.5]:
        rows = np.zeros((M, 5))
        for m in range(M):
            recs, y = generate_scenario_2_trial(3000 + m)
            if flip > 0:
                rng = np.random.default_rng(4000 + m)
                for r in recs:
                    if rng.random() < flip:
                        r.sentiment_score = -r.sentiment_score
                        r.sentiment_label = ("positive" if r.sentiment_score >= 0.05
                                             else "negative" if r.sentiment_score <= -0.05 else "neutral")
            rows[m] = [abs(e - (-0.1)) for e in _estimators(ConsensusEstimatorSuite(recs))]
        mean = rows.mean(0); se = rows.std(0) / np.sqrt(M)
        line = "  ".join(f"{n}={mean[i]:.3f}±{se[i]:.3f}" for i, n in enumerate(["Unw", "Heur", "IPS", "PACE", "PACE-LOO"]))
        print(f"  flip={flip:.1f}: {line}")


# ----------------------------------------------------------------------------
# E4: edge cases
# ----------------------------------------------------------------------------
def run_e4():
    print("\n[E4] Edge cases")
    def show(name, recs, y_star):
        if len(recs) == 0:
            print(f"  {name}: empty (skipped)"); return
        s = ConsensusEstimatorSuite(recs)
        ests = _estimators(s)
        print(f"  {name}: " + "  ".join(
            f"{n}={ests[i]:.3f} (MAE {abs(ests[i]-y_star):.3f})"
            for i, n in enumerate(["Unw", "Heur", "IPS", "PACE", "PACE-LOO"])))

    show("N=1", [CommentRecord("a", "only", 0.5, "positive", 10, 0, 1, 1.0, 0)], 0.5)
    show("N=2 balanced", [CommentRecord("a", "p", 0.5, "positive", 10, 0, 1, 1.0, 0),
                          CommentRecord("b", "n", -0.5, "negative", 10, 0, 2, 1.5, 0)], 0.0)
    show("all-zero likes", [CommentRecord(str(i), f"c{i}", 0.6, "positive", 0, 0, i + 1, i, 0)
                            for i in range(20)], 0.6)
    # single viral outlier at rank 1 vs rank 50 (unique ranks)
    for rk in [1, 50]:
        recs = []
        for i in range(50):
            rank = i + 1 if i + 1 < rk else i + 2  # leave rank `rk` to the outlier
            recs.append(CommentRecord(f"c{i}", f"p{i}", 0.6, "positive", 10, 0, rank, i * 0.3, 0))
        recs.append(CommentRecord("out", "viral", -0.9, "negative", 50000, 500, rk, 0.1, 0))
        recs.sort(key=lambda r: r.rank)
        show(f"outlier at rank {rk}", recs, 0.6)


if __name__ == "__main__":
    run_e1(30)
    run_e2(30)
    run_e3(30)
    run_e4()
