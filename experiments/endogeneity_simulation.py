"""
Endogeneity simulation — independently controlled rank, arrival, and sentiment.

Replaces the flawed run_e2() in expand_simulations.py, which set
arrival_hours = rank * 0.2 (perfect rank-arrival collinearity).

Seven data-generating processes (DGPs), each with an explicit target estimand
(the unweighted mean of true sentiment, i.e. the participant consensus):

  A  rank independent of sentiment and arrival
  B  rank correlated with sentiment
  C  arrival correlated with sentiment
  D  rank and arrival independently correlated with sentiment
  E  rank and arrival correlated with each other, independent of sentiment
  F  likes drive retrieval rank (ranking endogeneity)
  G  latent popularity drives both rank and arrival

For each scenario, M Monte Carlo trials estimate the five estimators and report
bias, MAE, RMSE, standard error, and a 95% confidence interval of the bias.
Nothing here is constructed to favor PACE; the unweighted mean is the natural
reference because it is unbiased for the declared estimand by construction.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from analyzers.consensus_framework import ConsensusEstimatorSuite, CommentRecord

NAMES = ["Unweighted", "Heuristic", "NC-IPS", "PACE", "PACE-LOO"]


def _estimates(recs):
    s = ConsensusEstimatorSuite(recs)
    return np.array([
        s.estimate_unweighted()["consensus_score"],
        s.estimate_heuristic_log_likes()["consensus_score"],
        s.estimate_position_ips()["consensus_score"],
        s.estimate_pace()["consensus_score"],
        s.estimate_pace_loo()["consensus_score"],
    ])


def _records(s, rank, arrival, likes):
    replies = np.maximum(0, (likes * 0.1).astype(int))
    recs = []
    for i in range(len(s)):
        lab = "positive" if s[i] >= 0.05 else ("negative" if s[i] <= -0.05 else "neutral")
        recs.append(CommentRecord(f"c{i}", f"t{i}", float(s[i]), lab,
                                  int(max(0, likes[i])), int(replies[i]),
                                  int(rank[i]), float(arrival[i]), 0))
    return recs


def _poisson_likes(rank, arrival, rng, base=50.0, gamma=0.8, lam=0.3):
    expo = (rank.astype(float) ** -gamma) * ((1.0 + arrival) ** -lam)
    return rng.poisson(base * expo) + rng.integers(0, 5, len(rank))


def scenario_A(seed, n=80):
    rng = np.random.default_rng(seed)
    s = np.clip(rng.normal(0.1, 0.5, n), -1, 1)
    rank = rng.permutation(np.arange(1, n + 1))
    arrival = rng.uniform(0.1, 48.0, n)
    likes = _poisson_likes(rank, arrival, rng)
    return _records(s, rank, arrival, likes), float(s.mean())


def scenario_B(seed, n=80):
    rng = np.random.default_rng(seed)
    s = np.clip(rng.normal(0.1, 0.5, n), -1, 1)
    order = np.argsort(-(s + rng.normal(0, 0.1, n)))
    rank = np.empty(n, int); rank[order] = np.arange(1, n + 1)
    arrival = rng.uniform(0.1, 48.0, n)
    likes = _poisson_likes(rank, arrival, rng)
    return _records(s, rank, arrival, likes), float(s.mean())


def scenario_C(seed, n=80):
    rng = np.random.default_rng(seed)
    s = np.clip(rng.normal(0.1, 0.5, n), -1, 1)
    rank = rng.permutation(np.arange(1, n + 1))
    arrival = np.clip(rng.uniform(0.1, 48.0, n) - 20.0 * s + 10.0, 0.05, 100.0)
    likes = _poisson_likes(rank, arrival, rng)
    return _records(s, rank, arrival, likes), float(s.mean())


def scenario_D(seed, n=80):
    rng = np.random.default_rng(60 + seed)
    s = np.clip(rng.normal(0.1, 0.5, n), -1, 1)
    order = np.argsort(-(s + rng.normal(0, 0.2, n)))
    rank = np.empty(n, int); rank[order] = np.arange(1, n + 1)
    arrival = np.clip(rng.uniform(0.1, 48.0, n) - 20.0 * s + 10.0, 0.05, 100.0)
    likes = _poisson_likes(rank, arrival, rng)
    return _records(s, rank, arrival, likes), float(s.mean())


def scenario_E(seed, n=80):
    rng = np.random.default_rng(seed)
    s = np.clip(rng.normal(0.1, 0.5, n), -1, 1)
    latent = rng.normal(0, 1, n)
    order = np.argsort(-latent)
    rank = np.empty(n, int); rank[order] = np.arange(1, n + 1)
    arrival = np.clip(20.0 - 10.0 * latent, 0.05, 100.0)
    likes = _poisson_likes(rank, arrival, rng)
    return _records(s, rank, arrival, likes), float(s.mean())


def scenario_F(seed, n=80):
    rng = np.random.default_rng(seed)
    s = np.clip(rng.normal(0.1, 0.5, n), -1, 1)
    arrival = rng.uniform(0.1, 48.0, n)
    base_rank = np.arange(1, n + 1)
    likes = _poisson_likes(base_rank, arrival, rng)
    order = np.argsort(-likes)
    rank = np.empty(n, int); rank[order] = np.arange(1, n + 1)
    return _records(s, rank, arrival, likes), float(s.mean())


def scenario_G(seed, n=80):
    rng = np.random.default_rng(seed)
    s = np.clip(rng.normal(0.1, 0.5, n), -1, 1)
    pop = rng.gamma(2.0, 2.0, n)
    base_rank = np.arange(1, n + 1)
    arrival0 = rng.uniform(0.1, 48.0, n)
    expo = (base_rank ** -0.8) * ((1 + arrival0) ** -0.3)
    likes = rng.poisson(50 * expo * (1 + 0.3 * pop)) + rng.integers(0, 5, n)
    order = np.argsort(-likes)
    rank = np.empty(n, int); rank[order] = np.arange(1, n + 1)
    arrival = np.clip(20.0 - 5.0 * pop, 0.05, 100.0)
    return _records(s, rank, arrival, likes), float(s.mean())


SCENARIOS = {
    "A rank|sent,arrival": scenario_A,
    "B rank~sentiment": scenario_B,
    "C arrival~sentiment": scenario_C,
    "D rank&arrival~sentiment": scenario_D,
    "E rank~arrival|sent": scenario_E,
    "F likes->rank": scenario_F,
    "G latent popularity": scenario_G,
}


def run(M=500, n=80):
    print(f"Endogeneity simulation: M={M} trials, N={n}, target = unweighted mean of true sentiment\n")
    print(f"{'scenario':<24} {'estimator':<12} {'bias':>8} {'MAE':>8} {'RMSE':>8} {'SE':>8} {'95% CI (bias)'}")
    print("-" * 95)
    for name, fn in SCENARIOS.items():
        err = np.zeros((M, 5))
        for m in range(M):
            recs, truth = fn(seed=10000 + m, n=n)
            err[m] = _estimates(recs) - truth
        bias = err.mean(0)
        mae = np.abs(err).mean(0)
        rmse = np.sqrt((err ** 2).mean(0))
        se = err.std(0) / np.sqrt(M)
        lo = bias - 1.96 * se
        hi = bias + 1.96 * se
        for j, est in enumerate(NAMES):
            print(f"{name:<24} {est:<12} {bias[j]:>+8.4f} {mae[j]:>8.4f} {rmse[j]:>8.4f} "
                  f"{se[j]:>8.4f} [{lo[j]:+.4f},{hi[j]:+.4f}]")
        print("-" * 95)


if __name__ == "__main__":
    run(500, 80)
