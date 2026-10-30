"""
Full Independent Reproducibility & Statistical Audit Script
Author: Sayed Saifullah (VIT Semester 5 AI Project)

Outputs:
1. data/monte_carlo_trial_level_results.csv: 150 individual trial-level observations (50 trials x 3 scenarios).
2. data/ablation_trial_level_results.csv: 150 individual trial-level ablation observations.
3. data/replicated_benchmark_results.csv: Summary table with Mean +/- SE and Paired Wilcoxon Tests.
4. data/replicated_ablation_results.csv: Summary ablation table with Mean +/- SE.
5. Evaluates real-world YouTube corpus (data/youtube_corpus_100_comments.csv) with refitting bootstrap.
"""

import sys
import os
import math
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from analyzers.consensus_framework import ConsensusEstimatorSuite, CommentRecord


def generate_scenario_1_trial(seed: int) -> tuple:
    """Scenario 1: Early Viral Outlier (N=50, Y* = 0.600)"""
    np.random.seed(seed)
    comments = []
    # Rank 1: Early viral negative comment (15k likes at t=0.1h)
    comments.append(CommentRecord("c_0", "Extreme viral outrage", -0.9, "negative", likes=15000, replies=250, rank=1, arrival_hours=0.1, depth=0))
    # Ranks 2-50: Organic positive comments (likes 5-50)
    for i in range(1, 50):
        s_val = float(np.clip(np.random.normal(0.60, 0.05), -1.0, 1.0))
        comments.append(CommentRecord(
            f"c_{i}", f"Positive review {i}", s_val, "positive",
            likes=int(np.random.randint(5, 50)), replies=int(np.random.randint(0, 3)),
            rank=i + 1, arrival_hours=0.5 + i * 0.4, depth=0
        ))
    return comments, 0.600


def generate_scenario_2_trial(seed: int) -> tuple:
    """Scenario 2: Late Organic Groundswell (N=60, Y* = -0.100)"""
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
    # Comment 34: Late critical critique (800 likes at t=24h, rank 35)
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
    """Scenario 3: Democratic Balanced Discussion (N=50, Y* = 0.000)"""
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


def run_full_reproduction(n_trials: int = 50):
    print(f"================================================================================")
    print(f"       REPRODUCING ALL EXPERIMENTS ACROSS {n_trials} TRIALS PER SCENARIO        ")
    print(f"================================================================================")

    trial_level_data = []
    ablation_trial_data = []

    scenarios = {
        "1. Early Viral Outlier": generate_scenario_1_trial,
        "2. Late Organic Groundswell": generate_scenario_2_trial,
        "3. Democratic Consensus": generate_scenario_3_trial
    }

    summary_rows = []
    summary_ablation_rows = []

    for sc_name, gen_fn in scenarios.items():
        print(f"\nProcessing Scenario: {sc_name}...")
        
        # Scenario accumulators for statistical tests
        mae_unw_list = []
        mae_heur_list = []
        mae_ips_list = []
        mae_pace_list = []
        
        w_max_heur_list = []
        w_max_pace_list = []
        rho_heur_list = []
        rho_pace_list = []
        ess_pace_list = []

        mae_notime_list = []
        mae_norank_list = []
        mae_nosmear_list = []

        for m in range(n_trials):
            seed = 42 + m
            comments, y_star = gen_fn(seed)
            suite = ConsensusEstimatorSuite(comments)

            m_unw = suite.estimate_unweighted()
            m_heur = suite.estimate_heuristic_log_likes()
            m_ips = suite.estimate_position_ips()
            m_pace = suite.estimate_pace()
            rob = suite._evaluate_robustness()

            mae_unw = abs(m_unw["consensus_score"] - y_star)
            mae_heur = abs(m_heur["consensus_score"] - y_star)
            mae_ips = abs(m_ips["consensus_score"] - y_star)
            mae_pace = abs(m_pace["consensus_score"] - y_star)

            w_max_h = float(np.max(m_heur["weights"]))
            w_max_p = float(np.max(m_pace["weights"]))
            rho_h = rob["heuristic_stability"]
            rho_p = rob["pace_stability"]
            ess_p = m_pace["effective_sample_size"]

            # Save individual trial observation
            trial_level_data.append({
                "trial_id": m + 1,
                "scenario": sc_name,
                "random_seed": seed,
                "ground_truth_y_star": y_star,
                "unweighted_estimate": round(m_unw["consensus_score"], 4),
                "unweighted_mae": round(mae_unw, 4),
                "heuristic_estimate": round(m_heur["consensus_score"], 4),
                "heuristic_mae": round(mae_heur, 4),
                "nc_ips_estimate": round(m_ips["consensus_score"], 4),
                "nc_ips_mae": round(mae_ips, 4),
                "pace_estimate": round(m_pace["consensus_score"], 4),
                "pace_mae": round(mae_pace, 4),
                "heuristic_w_max": round(w_max_h, 4),
                "pace_w_max": round(w_max_p, 4),
                "heuristic_stability_rho": round(rho_h, 4),
                "pace_stability_rho": round(rho_p, 4),
                "pace_ess": round(ess_p, 2)
            })

            mae_unw_list.append(mae_unw)
            mae_heur_list.append(mae_heur)
            mae_ips_list.append(mae_ips)
            mae_pace_list.append(mae_pace)
            w_max_heur_list.append(w_max_h)
            w_max_pace_list.append(w_max_p)
            rho_heur_list.append(rho_h)
            rho_pace_list.append(rho_p)
            ess_pace_list.append(ess_p)

            # Ablation refits on the same trial data.
            # Each ablation removes EXACTLY ONE component, keeping all others
            # (including Duan smearing) intact.
            log_likes = np.log1p(suite.likes)
            log_ranks = np.log(suite.ranks)
            log_arrivals = np.log1p(suite.arrivals)

            def _ablation_estimate(X_abl, sentiments, likes):
                """Fit OLS, apply Duan smearing, compute residual-based weights."""
                beta_abl, _, _, _ = np.linalg.lstsq(X_abl, log_likes, rcond=None)
                log_fitted_abl = X_abl @ beta_abl
                ols_res_abl = log_likes - log_fitted_abl
                smear_abl = float(np.mean(np.exp(ols_res_abl)))
                pred_abl = np.maximum(0.0, smear_abl * np.exp(log_fitted_abl) - 1.0)
                res_abl = likes - pred_abl
                med_abl = np.median(res_abl)
                mad_abl = np.median(np.abs(res_abl - med_abl)) + 1e-6
                z_abl = np.clip((res_abl - med_abl) / (1.4826 * mad_abl), -20.0, 20.0)
                w_abl = 1.0 / (1.0 + np.exp(-z_abl))
                w_abl = w_abl / np.sum(w_abl)
                return float(np.sum(w_abl * sentiments))

            # 1. w/o Temporal: remove ln(1 + arrival) from design matrix
            X_nt = np.column_stack([np.ones(suite.n), log_ranks])
            est_notime = _ablation_estimate(X_nt, suite.sentiments, suite.likes)
            mae_notime = abs(est_notime - y_star)

            # 2. w/o Rank: remove ln(rank) from design matrix
            X_nr = np.column_stack([np.ones(suite.n), log_arrivals])
            est_norank = _ablation_estimate(X_nr, suite.sentiments, suite.likes)
            mae_norank = abs(est_norank - y_star)

            # 3. w/o Smearing: fit full OLS but skip Duan smearing (naive exp)
            X_full = np.column_stack([np.ones(suite.n), log_ranks, log_arrivals])
            beta_ns, _, _, _ = np.linalg.lstsq(X_full, log_likes, rcond=None)
            log_fitted_ns = X_full @ beta_ns
            # NO smearing factor applied — this is the ablation target
            pred_ns = np.maximum(0.0, np.exp(log_fitted_ns) - 1.0)
            res_ns = suite.likes - pred_ns
            med_ns = np.median(res_ns)
            mad_ns = np.median(np.abs(res_ns - med_ns)) + 1e-6
            z_ns = np.clip((res_ns - med_ns) / (1.4826 * mad_ns), -20.0, 20.0)
            w_ns = 1.0 / (1.0 + np.exp(-z_ns))
            w_ns = w_ns / np.sum(w_ns)
            mae_nosmear = abs(float(np.sum(w_ns * suite.sentiments)) - y_star)

            ablation_trial_data.append({
                "trial_id": m + 1,
                "scenario": sc_name,
                "random_seed": seed,
                "ground_truth_y_star": y_star,
                "full_pace_mae": round(mae_pace, 4),
                "wo_temporal_mae": round(mae_notime, 4),
                "wo_rank_mae": round(mae_norank, 4),
                "wo_smearing_mae": round(mae_nosmear, 4)
            })

            mae_notime_list.append(mae_notime)
            mae_norank_list.append(mae_norank)
            mae_nosmear_list.append(mae_nosmear)

        # Paired Wilcoxon Signed-Rank Test across the 50 trials
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

        summary_ablation_rows.append({
            "Scenario": sc_name,
            "Full PACE MAE": f"{np.mean(mae_pace_list):.3f} +/- {np.std(mae_pace_list)/np.sqrt(n_trials):.3f}",
            "w/o Temporal MAE": f"{np.mean(mae_notime_list):.3f} +/- {np.std(mae_notime_list)/np.sqrt(n_trials):.3f}",
            "w/o Rank MAE": f"{np.mean(mae_norank_list):.3f} +/- {np.std(mae_norank_list)/np.sqrt(n_trials):.3f}",
            "w/o Smearing MAE": f"{np.mean(mae_nosmear_list):.3f} +/- {np.std(mae_nosmear_list)/np.sqrt(n_trials):.3f}"
        })

    # Evaluate Real-World YouTube Dataset (dQw4w9WgXcQ, N=100)
    print("\nEvaluating Real-World YouTube Dataset (data/youtube_corpus_100_comments.csv)...")
    csv_yt = ROOT_DIR / "data" / "youtube_corpus_100_comments.csv"
    if not csv_yt.exists():
        raise FileNotFoundError("Missing real YouTube dataset file.")

    df_yt = pd.read_csv(csv_yt)
    yt_comments = [
        CommentRecord(
            str(r["comment_id"]), str(r["text"]), float(r["sentiment_score"]),
            str(r["sentiment_label"]), int(r["likes"]), int(r["replies"]),
            int(r["rank"]), float(r["arrival_hours"]), 0
        )
        for _, r in df_yt.iterrows()
    ]

    yt_suite = ConsensusEstimatorSuite(yt_comments)
    yt_unw = yt_suite.estimate_unweighted()
    yt_heur = yt_suite.estimate_heuristic_log_likes()
    yt_ips = yt_suite.estimate_position_ips()
    yt_pace = yt_suite.estimate_pace()
    yt_rob = yt_suite._evaluate_robustness()

    # Full-pipeline refitting bootstrap CI on real YouTube data (500 resamples)
    pace_ci = yt_suite.compute_full_pipeline_bootstrap_ci(method="PACE_Proposed", n_bootstraps=500)
    heur_ci = yt_suite.compute_full_pipeline_bootstrap_ci(method="HeuristicLogLikes", n_bootstraps=500)

    summary_rows.append({
        "Scenario": "4. Real YouTube: Rick Astley",
        "N": 100,
        "Ground Truth Y*": "Observational",
        "Unweighted MAE": f"{yt_unw['consensus_score']:.3f} (mu)",
        "Heuristic MAE": f"{yt_heur['consensus_score']:.3f} [{heur_ci[0]:.2f}, {heur_ci[1]:.2f}]",
        "NC-IPS MAE": f"{yt_ips['consensus_score']:.3f} (mu)",
        "PACE MAE": f"{yt_pace['consensus_score']:.3f} [{pace_ci[0]:.2f}, {pace_ci[1]:.2f}]",
        "Paired Wilcoxon p-val": "N/A (Observational)",
        "Heuristic w_max*": f"{np.max(yt_heur['weights']):.3f}",
        "PACE w_max*": f"{np.max(yt_pace['weights']):.3f}",
        "Heuristic rho": f"{yt_rob['heuristic_stability']:.4f}",
        "PACE rho": f"{yt_rob['pace_stability']:.4f}",
        "PACE ESS": f"{yt_pace['effective_sample_size']:.1f}"
    })

    # Save trial-level raw data (150 rows)
    df_trials = pd.DataFrame(trial_level_data)
    trials_csv_path = ROOT_DIR / "data" / "monte_carlo_trial_level_results.csv"
    df_trials.to_csv(trials_csv_path, index=False)
    print(f"\n[OK] Saved 150 individual trial-level observations to: {trials_csv_path}")

    # Save trial-level ablation data (150 rows)
    df_ablation_trials = pd.DataFrame(ablation_trial_data)
    ablation_trials_csv_path = ROOT_DIR / "data" / "ablation_trial_level_results.csv"
    df_ablation_trials.to_csv(ablation_trials_csv_path, index=False)
    print(f"[OK] Saved 150 individual ablation trial observations to: {ablation_trials_csv_path}")

    # Save summary tables
    df_summary = pd.DataFrame(summary_rows)
    df_ablation = pd.DataFrame(summary_ablation_rows)
    summary_path = ROOT_DIR / "data" / "replicated_benchmark_results.csv"
    ablation_path = ROOT_DIR / "data" / "replicated_ablation_results.csv"
    df_summary.to_csv(summary_path, index=False)
    df_ablation.to_csv(ablation_path, index=False)
    print(f"[OK] Saved summary benchmark table to: {summary_path}")
    print(f"[OK] Saved summary ablation table to: {ablation_path}")

    print("\n================================================================================")
    print("       REPLICATED MONTE CARLO BENCHMARK (M = 50 RUNS, MEAN +/- SE)              ")
    print("================================================================================")
    print(df_summary.to_string(index=False))

    print("\n================================================================================")
    print("       REPLICATED MULTI-SCENARIO ABLATION (M = 50 RUNS, MAE +/- SE)             ")
    print("================================================================================")
    print(df_ablation.to_string(index=False))

    return df_trials, df_summary, df_ablation


if __name__ == "__main__":
    run_full_reproduction()
