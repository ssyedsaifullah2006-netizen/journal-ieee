"""
Engagement-Calibrated Sentiment Diagnostics Framework.
Implements the 3 core diagnostic metrics:
1. Gini-Gated Reliability Criterion (R)
2. Sentiment Majority Flip Statistic (Phi) with Permutation Significance Test
3. Leave-Top-K-Out Robustness Curve (rho)

Completely deterministic and API-independent (uses numpy/scipy).
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from analyzers.lexicon_sentiment import CommentSentiment


def compute_gini(weights: np.ndarray) -> float:
    """
    Compute Gini coefficient of engagement weights.
    G = 0 means perfect equality (all comments have same weight).
    G -> 1 means extreme concentration (few viral comments dominate).
    """
    if len(weights) == 0:
        return 0.0
    weights = np.asarray(weights, dtype=np.float64)
    if np.all(weights == 0):
        return 0.0
    
    # Sort weights ascending
    sorted_weights = np.sort(weights)
    n = len(weights)
    index = np.arange(1, n + 1)
    
    # Gini formula: (2 * sum(i * w_i)) / (n * sum(w_i)) - (n + 1) / n
    total = np.sum(sorted_weights)
    if total == 0:
        return 0.0
    gini = (2.0 * np.sum(index * sorted_weights)) / (n * total) - (n + 1.0) / n
    return float(np.clip(gini, 0.0, 1.0))


class EngagementDiagnostics:
    """
    Diagnostic suite for validating whether engagement-weighted sentiment
    shifts are statistically sound and structurally reliable.
    """

    def __init__(self, n_permutations: int = 2000, random_seed: int = 42):
        self.n_permutations = n_permutations
        self.random_seed = random_seed

    def compute_weighted_distribution(self, comments: List[CommentSentiment]) -> Dict[str, Any]:
        """
        Compute normalized sentiment distribution scaled by engagement weights.
        """
        if not comments:
            return {"positive": 0.0, "negative": 0.0, "neutral": 0.0, "weights_sum": 0.0}

        w_pos = sum(c.engagement_weight for c in comments if c.label == "positive")
        w_neg = sum(c.engagement_weight for c in comments if c.label == "negative")
        w_neu = sum(c.engagement_weight for c in comments if c.label == "neutral")
        total_w = w_pos + w_neg + w_neu

        if total_w == 0:
            return {"positive": 0.333, "negative": 0.333, "neutral": 0.334, "weights_sum": 0.0}

        return {
            "positive": round(w_pos / total_w, 4),
            "negative": round(w_neg / total_w, 4),
            "neutral": round(w_neu / total_w, 4),
            "weights_sum": round(total_w, 2),
        }

    def compute_reliability_score(self, comments: List[CommentSentiment]) -> Dict[str, Any]:
        """
        Contribution 1: Gini-Gated Reliability Criterion (R)
        Combines inequality (Gini) with dispersion (Coefficient of Variation)
        Formula: R = (1 - Gini) * max(0, 1 - (std^2 / (mean^2 + eps)))
        """
        if not comments:
            return {"r_score": 0.0, "gini": 0.0, "reliability_level": "UNKNOWN"}

        weights = np.array([c.engagement_weight for c in comments], dtype=np.float64)
        gini = compute_gini(weights)

        mean_w = np.mean(weights)
        var_w = np.var(weights)
        eps = 1e-6

        # Dispersion penalty based on squared coefficient of variation
        cv_sq = var_w / (mean_w ** 2 + eps)
        dispersion_factor = max(0.0, 1.0 - min(cv_sq / 10.0, 1.0)) # scaled factor

        r_score = float((1.0 - gini) * dispersion_factor)
        r_score = round(np.clip(r_score, 0.0, 1.0), 4)

        if r_score >= 0.6:
            level = "HIGH"
            desc = "Engagement is democratically distributed. Weighted sentiment reflects broad audience consensus."
        elif r_score >= 0.3:
            level = "MODERATE"
            desc = "Moderate concentration. A small subset of comments holds amplified influence."
        else:
            level = "LOW"
            desc = "Extreme concentration. Sentiment outcome is heavily dictated by outlier viral comments."

        return {
            "r_score": r_score,
            "gini": round(gini, 4),
            "mean_weight": round(float(mean_w), 2),
            "variance_weight": round(float(var_w), 2),
            "reliability_level": level,
            "interpretation": desc
        }

    def compute_majority_flip_test(
        self, comments: List[CommentSentiment], raw_dist: Dict[str, Any], weighted_dist: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Contribution 2: Sentiment Majority Flip Statistic (Phi) with Permutation Significance Test.
        Tests H0: Engagement weights are distributed independently of comment polarity.
        """
        if not comments:
            return {"flip_detected": False, "p_value": 1.0, "is_significant": False}

        raw_majority = max(["positive", "negative", "neutral"], key=lambda k: raw_dist.get(f"{k}_pct", 0.0))
        weighted_majority = max(["positive", "negative", "neutral"], key=lambda k: weighted_dist.get(k, 0.0))

        flip_detected = (raw_majority != weighted_majority)

        # Observed maximum absolute divergence across classes
        labels = ["positive", "negative", "neutral"]
        observed_divergence = max(abs(weighted_dist[k] - raw_dist[f"{k}_pct"]) for k in labels)

        # Permutation Test
        weights = np.array([c.engagement_weight for c in comments], dtype=np.float64)
        comment_labels = np.array([c.label for c in comments])
        total_w = np.sum(weights)

        rng = np.random.default_rng(self.random_seed)
        greater_divergence_count = 0

        for _ in range(self.n_permutations):
            shuffled_weights = rng.permutation(weights)
            
            # Recompute weighted distribution with shuffled weights
            sim_pos = np.sum(shuffled_weights[comment_labels == "positive"]) / total_w
            sim_neg = np.sum(shuffled_weights[comment_labels == "negative"]) / total_w
            sim_neu = np.sum(shuffled_weights[comment_labels == "neutral"]) / total_w
            
            sim_div = max(
                abs(sim_pos - raw_dist["positive_pct"]),
                abs(sim_neg - raw_dist["negative_pct"]),
                abs(sim_neu - raw_dist["neutral_pct"])
            )

            if sim_div >= observed_divergence:
                greater_divergence_count += 1

        p_value = round((greater_divergence_count + 1) / (self.n_permutations + 1), 4)
        is_significant = bool(p_value < 0.05)

        return {
            "flip_detected": flip_detected,
            "raw_majority": raw_majority,
            "weighted_majority": weighted_majority,
            "observed_divergence": round(observed_divergence, 4),
            "p_value": p_value,
            "is_significant": is_significant,
            "interpretation": (
                "Statistically significant divergence: Audience endorsement structurally alters sentiment consensus."
                if is_significant else
                "Insignificant divergence: Weighted shift falls within random engagement fluctuation boundaries."
            )
        }

    def compute_leave_top_k_robustness(
        self, comments: List[CommentSentiment], raw_dist: Dict[str, Any], top_k_values: List[int] = [1, 2, 5, 10]
    ) -> Dict[str, Any]:
        """
        Contribution 3: Leave-Top-K-Out Robustness Curve.
        Progressively removes top-K comments by engagement weight and measures retention of divergence.
        """
        if len(comments) <= max(top_k_values, default=1):
            return {"robustness_score": 1.0, "curve": [], "verdict": "ROBUST"}

        # Sort comments descending by engagement weight
        sorted_comments = sorted(comments, key=lambda c: c.engagement_weight, reverse=True)
        labels = ["positive", "negative", "neutral"]

        # Base divergence at K=0
        base_weighted = self.compute_weighted_distribution(sorted_comments)
        d_0 = max(abs(base_weighted[k] - raw_dist[f"{k}_pct"]) for k in labels)

        curve = []
        d_at_10 = d_0

        for k in top_k_values:
            if k >= len(sorted_comments):
                continue
            truncated = sorted_comments[k:]
            trunc_dist = self.compute_weighted_distribution(truncated)
            trunc_div = max(abs(trunc_dist[k_label] - raw_dist[f"{k_label}_pct"]) for k_label in labels)
            
            curve.append({
                "k_removed": k,
                "divergence": round(trunc_div, 4),
                "majority": max(labels, key=lambda l: trunc_dist[l])
            })
            if k == 10:
                d_at_10 = trunc_div

        # Robustness score rho: 1 - relative drop in divergence after dropping top 10
        if d_0 > 1e-4:
            rho = round(float(np.clip(1.0 - abs(d_0 - d_at_10) / d_0, 0.0, 1.0)), 4)
        else:
            rho = 1.0

        if rho >= 0.75:
            verdict = "HIGHLY_ROBUST"
        elif rho >= 0.4:
            verdict = "MODERATELY_ROBUST"
        else:
            verdict = "FRAGILE_OUTLIER_DEPENDENT"

        return {
            "robustness_score": rho,
            "curve": curve,
            "verdict": verdict,
            "explanation": f"Robustness Score rho={rho}. Model retains {round(rho*100, 1)}% stability when viral comments are pruned."
        }

    def run_full_diagnostics(self, comments: List[CommentSentiment], raw_dist: Dict[str, Any]) -> Dict[str, Any]:
        """Run all 3 diagnostic dimensions and return cohesive report."""
        weighted_dist = self.compute_weighted_distribution(comments)
        reliability = self.compute_reliability_score(comments)
        flip_test = self.compute_majority_flip_test(comments, raw_dist, weighted_dist)
        robustness = self.compute_leave_top_k_robustness(comments, raw_dist)

        # Sentiment Amplification Index (SAI)
        sai = {
            "positive": round(weighted_dist["positive"] - raw_dist["positive_pct"], 4),
            "negative": round(weighted_dist["negative"] - raw_dist["negative_pct"], 4),
            "neutral": round(weighted_dist["neutral"] - raw_dist["neutral_pct"], 4),
        }

        return {
            "raw_distribution": raw_dist,
            "weighted_distribution": weighted_dist,
            "sentiment_amplification_index": sai,
            "reliability_metric": reliability,
            "majority_flip_test": flip_test,
            "robustness_analysis": robustness
        }
