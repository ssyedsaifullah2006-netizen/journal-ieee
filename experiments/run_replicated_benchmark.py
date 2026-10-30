"""
Replicated Monte Carlo Benchmark (M = 50 independent simulation trials per scenario).
Computes:
1. Mean Absolute Error (MAE) against Ground Truth Y* (Mean +/- Standard Error across 50 runs).
2. Paired Wilcoxon Signed-Rank Test on MAE: PACE vs Heuristic.
3. Maximum Normalized Weight Share (w_max / sum(w)) to audit weight concentration.
4. Multi-trial Ablation Study reporting Mean MAE +/- SE across 50 runs.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from analyzers.consensus_framework import ConsensusEstimatorSuite, CommentRecord


def generate_scenario_1_trial(seed: int) -> tuple:
    """Scenario 1: Early Viral Outlier (Y* = 0.600)"""
    np.random.seed(seed)
    comments = []
    # Comment 0: Viral troll outlier
    comments.append(CommentRecord("c_0", "Extreme viral outrage", -0.9, "negative", likes=15000, replies=250, rank=1, arrival_hours=0.1, depth=0))
    # Comments 1-49: Organic positive community
    for i in range(1, 50):
        s_val = float(np.clip(np.random.normal(0.60, 0.05), -1.0, 1.0))
        comments.append(CommentRecord(
            f"c_{i}", f"Positive review {i}", s_val, "positive",
            likes=int(np.random.randint(5, 50)), replies=int(np.random.randint(0, 3)),
            rank=i + 1, arrival_hours=0.5 + i * 0.4, depth=0
        ))
    return comments, 0.600


def generate_scenario_2_trial(seed: int) -> tuple:
    """Scenario 2: Late Organic Groundswell (Y* = -0.100)"""
    np.random.seed(seed)
    comments = []
    # 10 early hype comments
    for i in range(10):
        comments.append(CommentRecord(
            f"c_early_{i}", f"Generic hype {i}", float(np.clip(np.random.normal(0.70, 0.05), -1.0, 1.0)), "positive",
            likes=int(np.random.randint(150, 350)), replies=5,
            rank=i + 1, arrival_hours=0.1 + i * 0.2, depth=0
        ))
    # 24 casual neutral comments
    for i in range(10, 34):
        comments.append(CommentRecord(
            f"c_mid_{i}", f"Casual comment {i}", 0.10, "neutral",
            likes=int(np.random.randint(10, 40)), replies=1,
            rank=i + 1, arrival_hours=2.0 + i * 0.5, depth=0
        ))
    # Comment 34: Late groundswell critical critique
    comments.append(CommentRecord("c_groundswell", "Critical issue detected", -0.85, "negative", likes=800, replies=45, rank=35, arrival_hours=24.0, depth=0))
    # 25 late agreeing comments
    for i in range(35, 60):
        comments.append(CommentRecord(
            f"c_late_{i}", f"Agreeing comment {i}", float(np.clip(np.random.normal(-0.40, 0.05), -1.0, 1.0)), "negative",
            likes=int(np.random.randint(20, 60)), replies=2,
            rank=i + 1, arrival_hours=25.0 + (i - 35) * 0.3, depth=0
        ))
    return comments, -0.100


def generate_scenario_3_trial(seed: int) -> tuple:
    """Scenario 3: Democratic Balanced Discussion (Y* = 0.000)"""
    np.random.seed(seed)
    raw_s = np.array([0.50] * 25 + [-0.50] * 25)
    perm_s = np.random.permutation(raw_s)
    comments = []
    for i in range(50):
        s_val = float(perm_s[i])
        lbl = "positive" if s_val > 0 else "negative"
        expected_l = int(1000 * ((i + 1) ** (-0.7)) * ((1 + i * 0.3) ** (-0.2)))
        noise_l = max(0, expected_l + int(np.random.randint(-15, 15)))
        comments.append(CommentRecord(
            f"c_demo_{i}", f"Democratic opinion {i}", s_val, lbl,
            likes=noise_l, replies=int(np.random.randint(0, 5)),
            rank=i + 1, arrival_hours=i * 0.3, depth=0
        ))
    return comments, 0.000


def run_replicated_benchmark(n_trials: int = 50):
    print(f"Running Replicated Monte Carlo Benchmark (M = {n_trials} independent trials)...")
    
    generators = {
        "1. Early Viral Outlier": generate_scenario_1_trial,
        "2. Late Organic Groundswell": generate_scenario_2_trial,
        "3. Democratic Consensus": generate_scenario_3_trial
    }

    summary_rows = []
    ablation_rows = []

    for sc_name, gen_fn in generators.items():
        mae_unw_list = []
        mae_heur_list = []
        mae_ips_list = []
        mae_pace_list = []
        
        w_max_heur_list = []
        w_max_pace_list = []
        
        rho_heur_list = []
        rho_pace_list = []
        ess_pace_list = []

        # Ablation tracking
        mae_notime_list = []
        mae_norank_list = []
        mae_nosmear_list = []

        for m in range(n_trials):
            seed = 42 + m
            comments, y_star = gen_fn(seed)
            suite = ConsensusEstimatorSuite(comments)
            
            # Estimates
            m_unw = suite.estimate_unweighted()
            m_heur = suite.estimate_heuristic_log_likes()
            m_ips = suite.estimate_position_ips()
            m_pace = suite.estimate_pace()
            rob = suite._evaluate_robustness()

            # MAEs
            mae_unw_list.append(abs(m_unw["consensus_score"] - y_star))
            mae_heur_list.append(abs(m_heur["consensus_score"] - y_star))
            mae_ips_list.append(abs(m_ips["consensus_score"] - y_star))
            mae_pace_list.append(abs(m_pace["consensus_score"] - y_star))

            # Weight Concentration: Max normalized weight share w_max / sum(w)
            w_max_heur_list.append(float(np.max(m_heur["weights"])))
            w_max_pace_list.append(float(np.max(m_pace["weights"])))

            # Robustness & ESS
            rho_heur_list.append(rob["heuristic_stability"])
            rho_pace_list.append(rob["pace_stability"])
            ess_pace_list.append(m_pace["effective_sample_size"])

            # Ablations for this trial
            # 1. No time
            log_likes = np.log1p(suite.likes)
            log_ranks = np.log(suite.ranks)
            X_nt = np.column_stack([np.ones(suite.n), log_ranks])
            beta_nt, _, _, _ = np.linalg.lstsq(X_nt, log_likes, rcond=None)
            pred_nt = np.exp(X_nt @ beta_nt) - 1.0
            res_nt = suite.likes - np.maximum(0, pred_nt)
            mad_nt = np.median(np.abs(res_nt - np.median(res_nt))) + 1e-6
            z_nt = np.clip((res_nt - np.median(res_nt)) / (1.4826 * mad_nt), -20.0, 20.0)
            w_nt = (1.0 / (1.0 + np.exp(-z_nt))) / np.sum(1.0 / (1.0 + np.exp(-z_nt)))
            mae_notime_list.append(abs(float(np.sum(w_nt * suite.sentiments)) - y_star))

            # 2. No rank
            log_arrivals = np.log1p(suite.arrivals)
            X_nr = np.column_stack([np.ones(suite.n), log_arrivals])
            beta_nr, _, _, _ = np.linalg.lstsq(X_nr, log_likes, rcond=None)
            pred_nr = np.exp(X_nr @ beta_nr) - 1.0
            res_nr = suite.likes - np.maximum(0, pred_nr)
            mad_nr = np.median(np.abs(res_nr - np.median(res_nr))) + 1e-6
            z_nr = np.clip((res_nr - np.median(res_nr)) / (1.4826 * mad_nr), -20.0, 20.0)
            w_nr = (1.0 / (1.0 + np.exp(-z_nr))) / np.sum(1.0 / (1.0 + np.exp(-z_nr)))
            mae_norank_list.append(abs(float(np.sum(w_nr * suite.sentiments)) - y_star))

            # 3. No smear
            pred_ns = np.exp(np.column_stack([np.ones(suite.n), log_ranks, log_arrivals]) @ np.array([2.0, -0.6, -0.2]))
            res_ns = suite.likes - np.maximum(0, pred_ns)
            mad_ns = np.median(np.abs(res_ns - np.median(res_ns))) + 1e-6
            z_ns = np.clip((res_ns - np.median(res_ns)) / (1.4826 * mad_ns), -20.0, 20.0)
            w_ns = (1.0 / (1.0 + np.exp(-z_ns))) / np.sum(1.0 / (1.0 + np.exp(-z_ns)))
            mae_nosmear_list.append(abs(float(np.sum(w_ns * suite.sentiments)) - y_star))

        # Paired Wilcoxon signed-rank test on MAE (PACE vs Heuristic across 50 trials)
        stat, p_val_mae = stats.wilcoxon(mae_heur_list, mae_pace_list, alternative="greater")

        summary_rows.append({
            "Scenario": sc_name,
            "N": len(comments),
            "Ground Truth Y*": y_star,
            "Unweighted MAE": f"{np.mean(mae_unw_list):.3f} +/- {np.std(mae_unw_list)/np.sqrt(n_trials):.3f}",
            "Heuristic MAE": f"{np.mean(mae_heur_list):.3f} +/- {np.std(mae_heur_list)/np.sqrt(n_trials):.3f}",
            "NC-IPS MAE": f"{np.mean(mae_ips_list):.3f} +/- {np.std(mae_ips_list)/np.sqrt(n_trials):.3f}",
            "PACE MAE": f"{np.mean(mae_pace_list):.3f} +/- {np.std(mae_pace_list)/np.sqrt(n_trials):.3f}",
            "Paired Wilcoxon p-val": f"{p_val_mae:.5f}",
            "Heuristic w_max*": f"{np.mean(w_max_heur_list):.3f}",
            "PACE w_max*": f"{np.mean(w_max_pace_list):.3f}",
            "Heuristic rho": f"{np.mean(rho_heur_list):.4f}",
            "PACE rho": f"{np.mean(rho_pace_list):.4f}",
            "PACE ESS": f"{np.mean(ess_pace_list):.1f}"
        })

        ablation_rows.append({
            "Scenario": sc_name,
            "Full PACE MAE": f"{np.mean(mae_pace_list):.3f} +/- {np.std(mae_pace_list)/np.sqrt(n_trials):.3f}",
            "w/o Temporal MAE": f"{np.mean(mae_notime_list):.3f} +/- {np.std(mae_notime_list)/np.sqrt(n_trials):.3f}",
            "w/o Rank MAE": f"{np.mean(mae_norank_list):.3f} +/- {np.std(mae_norank_list)/np.sqrt(n_trials):.3f}",
            "w/o Smearing MAE": f"{np.mean(mae_nosmear_list):.3f} +/- {np.std(mae_nosmear_list)/np.sqrt(n_trials):.3f}"
        })

    # Real-World YouTube Case Study (Video ID: dQw4w9WgXcQ)
    csv_path = ROOT_DIR / "data" / "youtube_corpus_100_comments.csv"
    df_yt = pd.read_csv(csv_path)
    yt_records = [
        CommentRecord(str(r["comment_id"]), str(r["text"]), float(r["sentiment_score"]), str(r["sentiment_label"]), int(r["likes"]), int(r["replies"]), int(r["rank"]), float(r["arrival_hours"]), 0)
        for _, r in df_yt.iterrows()
    ]
    yt_suite = ConsensusEstimatorSuite(yt_records)
    yt_unw = yt_suite.estimate_unweighted()
    yt_heur = yt_suite.estimate_heuristic_log_likes()
    yt_ips = yt_suite.estimate_position_ips()
    yt_pace = yt_suite.estimate_pace()
    yt_rob = yt_suite._evaluate_robustness()
    yt_pval = yt_suite._permutation_divergence_test(yt_pace["weights"], yt_heur["weights"])

    summary_rows.append({
        "Scenario": "4. Real YouTube: Rick Astley",
        "N": 100,
        "Ground Truth Y*": "Observational",
        "Unweighted MAE": f"{yt_unw['consensus_score']:.3f} (mu)",
        "Heuristic MAE": f"{yt_heur['consensus_score']:.3f} (mu)",
        "NC-IPS MAE": f"{yt_ips['consensus_score']:.3f} (mu)",
        "PACE MAE": f"{yt_pace['consensus_score']:.3f} (mu)",
        "Paired Wilcoxon p-val": f"{yt_pval:.5f} (perm)",
        "Heuristic w_max*": f"{np.max(yt_heur['weights']):.3f}",
        "PACE w_max*": f"{np.max(yt_pace['weights']):.3f}",
        "Heuristic rho": f"{yt_rob['heuristic_stability']:.4f}",
        "PACE rho": f"{yt_rob['pace_stability']:.4f}",
        "PACE ESS": f"{yt_pace['effective_sample_size']:.1f}"
    })

    df_summary = pd.DataFrame(summary_rows)
    df_ablation = pd.DataFrame(ablation_rows)

    summary_path = ROOT_DIR / "data" / "replicated_benchmark_results.csv"
    ablation_path = ROOT_DIR / "data" / "replicated_ablation_results.csv"

    df_summary.to_csv(summary_path, index=False)
    df_ablation.to_csv(ablation_path, index=False)

    print("\n================================================================================")
    print("       REPLICATED MONTE CARLO BENCHMARK (M = 50 RUNS, MEAN +/- SE)              ")
    print("================================================================================")
    print(df_summary.to_string(index=False))

    print("\n================================================================================")
    print("       REPLICATED MULTI-SCENARIO ABLATION (M = 50 RUNS, MAE +/- SE)             ")
    print("================================================================================")
    print(df_ablation.to_string(index=False))

    return df_summary, df_ablation


if __name__ == "__main__":
    run_replicated_benchmark()
