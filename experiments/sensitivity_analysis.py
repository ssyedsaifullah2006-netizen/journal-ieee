"""
Comprehensive Sensitivity Analysis & Empirical Audit Suite for PACE Framework
Directly evaluates:
1. Ground-Truth Target (Y*) Sensitivity in Scenario 2 across continuous estimand definitions.
2. Exposure-Baseline Scale and Non-Linear Distortion Sensitivity on observational YouTube data.
3. Semantic Lexicon Diagnostic (VADER zero-inflation, slang failure, meme-discourse spread).
"""

import sys
import os
import math
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from analyzers.consensus_framework import ConsensusEstimatorSuite, CommentRecord
from experiments.reproduce_all_experiments import generate_scenario_2_trial


def run_ground_truth_sensitivity(n_trials: int = 50):
    """
    Evaluates MAE of all four estimators in Scenario 2 under varying definitions
    of the target ground truth Y*:
    - Y* = author constant (-0.100)
    - Y* = unweighted sample mean s_bar (~ -0.025)
    - Y* = neutral null (0.000)
    - Y* = late-groundswell cluster mean (~ -0.40)
    """
    print("\n================================================================================")
    print(" 1. GROUND-TRUTH (Y*) SENSITIVITY AUDIT IN SCENARIO 2 (LATE GROUNDSWELL)")
    print("================================================================================")

    y_star_candidates = [-0.40, -0.20, -0.10, -0.025, 0.00, 0.10]
    results = []

    # Precompute trial estimates
    trial_estimates = []
    for m in range(n_trials):
        seed = 42 + m
        comments, _ = generate_scenario_2_trial(seed)
        suite = ConsensusEstimatorSuite(comments)
        s_bar = float(np.mean([c.sentiment_score for c in comments]))
        trial_estimates.append({
            "unweighted": suite.estimate_unweighted()["consensus_score"],
            "heuristic": suite.estimate_heuristic_log_likes()["consensus_score"],
            "nc_ips": suite.estimate_position_ips()["consensus_score"],
            "pace": suite.estimate_pace()["consensus_score"],
            "sample_mean": s_bar
        })

    for y_val in y_star_candidates:
        mae_unw = [abs(t["unweighted"] - y_val) for t in trial_estimates]
        mae_heur = [abs(t["heuristic"] - y_val) for t in trial_estimates]
        mae_ips = [abs(t["nc_ips"] - y_val) for t in trial_estimates]
        mae_pace = [abs(t["pace"] - y_val) for t in trial_estimates]

        winner = "PACE"
        min_mae = np.mean(mae_pace)
        if np.mean(mae_unw) < min_mae:
            winner = "Unweighted"; min_mae = np.mean(mae_unw)
        if np.mean(mae_heur) < min_mae:
            winner = "Heuristic"; min_mae = np.mean(mae_heur)
        if np.mean(mae_ips) < min_mae:
            winner = "NC-IPS"; min_mae = np.mean(mae_ips)

        row = {
            "Assumed Ground Truth Y*": f"{y_val:+.3f}",
            "Unweighted MAE": f"{np.mean(mae_unw):.3f}",
            "Heuristic MAE": f"{np.mean(mae_heur):.3f}",
            "NC-IPS MAE": f"{np.mean(mae_ips):.3f}",
            "PACE MAE": f"{np.mean(mae_pace):.3f}",
            "Best Estimator": winner
        }
        results.append(row)
        print(f"  Y* = {y_val:+.3f} | Unw: {np.mean(mae_unw):.3f} | Heur: {np.mean(mae_heur):.3f} | IPS: {np.mean(mae_ips):.3f} | PACE: {np.mean(mae_pace):.3f} => Best: {winner}")

    # Trial-level adaptive ground truth: Y* = s_bar of each trial
    mae_unw_adapt = [abs(t["unweighted"] - t["sample_mean"]) for t in trial_estimates]
    mae_heur_adapt = [abs(t["heuristic"] - t["sample_mean"]) for t in trial_estimates]
    mae_ips_adapt = [abs(t["nc_ips"] - t["sample_mean"]) for t in trial_estimates]
    mae_pace_adapt = [abs(t["pace"] - t["sample_mean"]) for t in trial_estimates]

    print(f"\n  [Adaptive Y* = per-trial sample mean s_bar]")
    print(f"    Unweighted MAE: {np.mean(mae_unw_adapt):.3f} (identity)")
    print(f"    Heuristic MAE:  {np.mean(mae_heur_adapt):.3f}")
    print(f"    NC-IPS MAE:     {np.mean(mae_ips_adapt):.3f}")
    print(f"    PACE MAE:       {np.mean(mae_pace_adapt):.3f} (PACE loses if target is unweighted mean)")

    df_gt = pd.DataFrame(results)
    out_gt = ROOT_DIR / "data" / "sensitivity_ground_truth_results.csv"
    df_gt.to_csv(out_gt, index=False)
    print(f"  [OK] Saved Y* sensitivity table to: {out_gt}")
    return df_gt


def run_baseline_calibration_sensitivity():
    """
    Evaluates how sensitive PACE is to scaling or non-linear distortion
    of the expected engagement baseline E[Likes | r, arrival].
    """
    print("\n================================================================================")
    print(" 2. EXPOSURE-BASELINE SCALING & NON-LINEAR DISTORTION SENSITIVITY")
    print("================================================================================")

    csv_path = ROOT_DIR / "data" / "youtube_corpus_100_comments.csv"
    df = pd.read_csv(csv_path)
    comments = [
        CommentRecord(str(r["comment_id"]), str(r["text"]), float(r["sentiment_score"]),
                      str(r["sentiment_label"]), int(r["likes"]), int(r["replies"]),
                      int(r["rank"]), float(r["arrival_hours"]), 0)
        for _, r in df.iterrows()
    ]
    suite = ConsensusEstimatorSuite(comments)
    base_res = suite.estimate_pace()
    base_score = base_res["consensus_score"]
    expected_l = np.array(base_res["expected_likes"])
    likes = suite.likes
    sentiments = suite.sentiments

    def _eval_custom_baseline(custom_expected):
        residuals = likes - custom_expected
        med_res = np.median(residuals)
        mad = np.median(np.abs(residuals - med_res)) + 1e-6
        studentized = np.clip((residuals - med_res) / (1.4826 * mad), -20.0, 20.0)
        weights = 1.0 / (1.0 + np.exp(-studentized))
        norm_w = weights / np.sum(weights)
        return float(np.sum(norm_w * sentiments)), norm_w

    distortion_tests = [
        ("0.25x Baseline Scale", expected_l * 0.25),
        ("0.50x Baseline Scale", expected_l * 0.50),
        ("1.00x Nominal Baseline", expected_l * 1.00),
        ("1.50x Baseline Scale", expected_l * 1.50),
        ("2.00x Baseline Scale", expected_l * 2.00),
        ("Square Root Distortion (sqrt(E))", np.sqrt(np.maximum(0.0, expected_l))),
        ("Superlinear Distortion (E^1.2)", np.power(np.maximum(0.0, expected_l), 1.2)),
    ]

    results = []
    for name, distorted_exp in distortion_tests:
        score, weights = _eval_custom_baseline(distorted_exp)
        w_max = float(np.max(weights))
        row = {
            "Perturbation": name,
            "PACE Consensus Estimate": round(score, 4),
            "Delta from Nominal": round(score - base_score, 4),
            "Max Weight Share": round(w_max, 4)
        }
        results.append(row)
        print(f"  {name:<35} | Estimate: {score:+.4f} (Delta: {score - base_score:+.4f}) | w_max: {w_max:.4f}")

    df_dist = pd.DataFrame(results)
    out_dist = ROOT_DIR / "data" / "sensitivity_baseline_distortion_results.csv"
    df_dist.to_csv(out_dist, index=False)
    print(f"  [OK] Saved baseline distortion table to: {out_dist}")
    return df_dist


def run_semantic_lexicon_audit():
    """
    Audits VADER behavior, zero-inflation, and tone on the YouTube corpus.
    """
    print("\n================================================================================")
    print(" 3. SEMANTIC LEXICON & DISCOURSE AUDIT ON YOUTUBE CORPUS")
    print("================================================================================")

    csv_path = ROOT_DIR / "data" / "youtube_corpus_100_comments.csv"
    df = pd.read_csv(csv_path)

    total_comments = len(df)
    zero_scores = df[df["sentiment_score"] == 0.0]
    nonzero_scores = df[df["sentiment_score"] != 0.0]

    mean_all = float(df["sentiment_score"].mean())
    mean_nonzero = float(nonzero_scores["sentiment_score"].mean())

    print(f"  Total comments: {total_comments}")
    print(f"  Zero-polarity comments (score == 0.0): {len(zero_scores)} ({len(zero_scores)/total_comments*100:.1f}%)")
    print(f"  Non-zero polarity comments:            {len(nonzero_scores)} ({len(nonzero_scores)/total_comments*100:.1f}%)")
    print(f"  Mean sentiment (all comments):         {mean_all:.4f}")
    print(f"  Mean sentiment (non-zero comments):     {mean_nonzero:.4f}")

    # Top engagement comments and their VADER scores
    print("\n  Top 5 Most Liked Comments and VADER Polarity:")
    for _, row in df.sort_values(by="likes", ascending=False).head(5).iterrows():
        text_snippet = str(row["text"])[:60] + "..." if len(str(row["text"])) > 60 else str(row["text"])
        print(f"    Rank #{row['rank']:<3} | Likes: {row['likes']:<9} | VADER: {row['sentiment_score']:+.4f} | \"{text_snippet}\"")

    # Audit slang cases
    banging_match = df[df["text"].str.contains("banging", case=False, na=False)]
    sick_match = df[df["text"].str.contains("sick", case=False, na=False)]
    if len(banging_match) > 0:
        row = banging_match.iloc[0]
        print(f"\n  Slang Audit: \"{row['text']}\" => VADER Compound: {row['sentiment_score']}")
    if len(sick_match) > 0:
        row = sick_match.iloc[0]
        print(f"  Slang Audit: \"{row['text']}\" => VADER Compound: {row['sentiment_score']}")


if __name__ == "__main__":
    run_ground_truth_sensitivity(50)
    run_baseline_calibration_sensitivity()
    run_semantic_lexicon_audit()
