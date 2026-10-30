"""
Exposure-proxy-adjusted semantic consensus framework for social media discussions.
Contains:
1. Baseline 1: Unweighted Mean (Standard Sample Average)
2. Baseline 2: Heuristic Log-Likes (Prior Art Chrome Extension / Social Mining)
3. Baseline 3: Position-Debiased Inverse Propensity Scoring (IPS - Joachims et al.)
4. Proposed Method: PACE (Preferential-Attachment Corrected Endorsement)
5. Comprehensive Evaluation & Robustness Metrics
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class CommentRecord:
    comment_id: str
    text: str
    sentiment_score: float  # Continuous sentiment in [-1.0, +1.0]
    sentiment_label: str    # 'positive', 'negative', 'neutral'
    likes: int
    replies: int
    rank: int               # Display order / visibility rank (1, 2, 3...)
    arrival_hours: float    # Elapsed hours between video publish and comment timestamp
    depth: int = 0          # 0 for root comment, 1 for direct reply, 2+ for nested


def exposure_surface_model(X, alpha, gamma, lambda_decay):
    """
    Parametric model for expected engagement given rank and temporal arrival:
    E[L | r, delta_t] = alpha * (r ^ -gamma) * ((1 + delta_t) ^ -lambda_decay)
    """
    rank, arrival = X
    # Avoid zero division and negative exponents blowing up
    safe_rank = np.maximum(rank, 1.0)
    safe_arrival = np.maximum(arrival, 0.0)
    return alpha * (safe_rank ** (-gamma)) * ((1.0 + safe_arrival) ** (-lambda_decay))


class ConsensusEstimatorSuite:
    """
    Comparative suite that runs all 4 methods side-by-side on the same comment stream.
    """

    def __init__(self, comments: List[CommentRecord], random_seed: int = 20261030):
        self.comments = comments
        self.n = len(comments)
        self.random_seed = random_seed
        
        # Extract numpy arrays for vector math
        self.sentiments = np.array([c.sentiment_score for c in comments], dtype=float)
        self.likes = np.array([max(0, c.likes) for c in comments], dtype=float)
        self.replies = np.array([max(0, c.replies) for c in comments], dtype=float)
        self.ranks = np.array([max(1, c.rank) for c in comments], dtype=float)
        self.arrivals = np.array([max(0.0, c.arrival_hours) for c in comments], dtype=float)
        self.depths = np.array([c.depth for c in comments], dtype=float)

    # -------------------------------------------------------------
    # Method 1: Unweighted Mean (Baseline 1)
    # -------------------------------------------------------------
    def estimate_unweighted(self) -> Dict[str, Any]:
        """Arithmetic mean: every comment counts as 1/N."""
        if self.n == 0:
            raise ValueError("At least one comment is required for estimation.")
        mean_sentiment = float(np.mean(self.sentiments))
        weights = np.ones(self.n) / max(1, self.n)
        return {
            "method": "UnweightedMean",
            "consensus_score": mean_sentiment,
            "weights": weights.tolist(),
            "majority_sentiment": self._majority_class(weights),
            "effective_sample_size": float(self.n)
        }

    # -------------------------------------------------------------
    # Method 2: Heuristic Log-Likes (Baseline 2 - Prior Art)
    # -------------------------------------------------------------
    def estimate_heuristic_log_likes(self) -> Dict[str, Any]:
        """w_i = 1 + ln(1 + likes) + 0.5 * ln(1 + replies)"""
        if self.n == 0:
            raise ValueError("At least one comment is required for estimation.")
        raw_w = 1.0 + np.log1p(self.likes) + 0.5 * np.log1p(self.replies)
        norm_w = raw_w / np.sum(raw_w)
        consensus_score = float(np.sum(norm_w * self.sentiments))
        ess = float((np.sum(norm_w) ** 2) / np.sum(norm_w ** 2))
        return {
            "method": "HeuristicLogLikes",
            "consensus_score": consensus_score,
            "weights": norm_w.tolist(),
            "majority_sentiment": self._majority_class(norm_w),
            "effective_sample_size": ess
        }

    # -------------------------------------------------------------
    # Method 3: Normalized Clipped IPS (NC-IPS Baseline - Joachims et al. 2017)
    # -------------------------------------------------------------
    def estimate_position_ips(self, gamma: float = 0.8, clip_percentile: float = 90.0) -> Dict[str, Any]:
        """
        Normalized Clipped Inverse Propensity Scoring (NC-IPS, Joachims et al. 2017):
        Propensity of visual examination: p_i = rank_i ^ (-gamma).
        Inverse propensity weights: w_i = (1 / p_i) * log1p(likes_i).
        Clipped at clip_percentile to prevent unbounded propensity variance.
        """
        if self.n == 0:
            raise ValueError("At least one comment is required for estimation.")
        propensity = self.ranks ** (-gamma)
        inv_propensity = 1.0 / np.maximum(propensity, 1e-4)
        
        # Clip inverse propensity at specified percentile
        clip_bound = np.percentile(inv_propensity, clip_percentile)
        clipped_inv_p = np.minimum(inv_propensity, clip_bound)
        
        raw_w = clipped_inv_p * np.log1p(self.likes) + 1e-5
        norm_w = raw_w / np.sum(raw_w)
        consensus_score = float(np.sum(norm_w * self.sentiments))
        ess = float((np.sum(norm_w) ** 2) / np.sum(norm_w ** 2))
        return {
            "method": "NormalizedClippedIPS",
            "consensus_score": consensus_score,
            "weights": norm_w.tolist(),
            "majority_sentiment": self._majority_class(norm_w),
            "effective_sample_size": ess
        }

    # -------------------------------------------------------------
    # Method 4: PACE (Exposure-Conditioned Engagement Adjustment)
    # -------------------------------------------------------------
    def estimate_pace(self) -> Dict[str, Any]:
        """
        Preferential-Attachment Corrected Endorsement (PACE).
        1. Closed-form log-linear baseline estimation:
           ln(Likes + 1) = beta_0 + beta_1 * ln(rank) + beta_2 * ln(1 + arrival)
        2. Duan's (1983) smearing retransformation factor to correct Jensen's inequality:
           E[Likes | r, arrival] = s_smear * exp(X * beta) - 1
        3. Standardized organic endorsement residual:
           delta = Likes - E[Likes | r, arrival]
        4. Bounded-influence M-estimation via robust studentized sigmoid gating.
        """
        if self.n == 0:
            raise ValueError("At least one comment is required for estimation.")
        if self.n < 5:
            return self.estimate_unweighted()

        # Design matrix X: [1, ln(rank), ln(1 + arrival)]
        log_likes = np.log1p(self.likes)
        log_ranks = np.log(self.ranks)
        log_arrivals = np.log1p(self.arrivals)
        X_mat = np.column_stack([np.ones(self.n), log_ranks, log_arrivals])

        try:
            # Closed-form Ordinary Least Squares
            beta, _, _, _ = np.linalg.lstsq(X_mat, log_likes, rcond=None)
            beta_0, beta_1, beta_2 = beta

            # Fitted log values and OLS residuals
            log_fitted = X_mat @ beta
            ols_residuals = log_likes - log_fitted

            # Duan's smearing factor: compensates for E[exp(eps)] >= 1 under Jensen's inequality
            duan_smearing_factor = float(np.mean(np.exp(ols_residuals)))

            # Exponentiated conditional mean with Duan's smearing correction
            expected_likes = np.maximum(0.0, duan_smearing_factor * np.exp(log_fitted) - 1.0)

            alpha_fit = float(np.exp(beta_0) * duan_smearing_factor)
            gamma_fit = float(max(0.0, -beta_1))   # Rank decay exponent
            lambda_fit = float(max(0.0, -beta_2))  # Temporal decay exponent
        except Exception:
            alpha_fit, gamma_fit, lambda_fit = (float(np.median(self.likes) + 1.0), 0.5, 0.2)
            expected_likes = alpha_fit * (self.ranks ** (-gamma_fit)) * ((1.0 + self.arrivals) ** (-lambda_fit))

        # Organic Endorsement Residual
        residuals = self.likes - expected_likes

        # Robust standardization using Median Absolute Deviation (MAD)
        median_res = np.median(residuals)
        mad = np.median(np.abs(residuals - median_res)) + 1e-6
        studentized_residuals = (residuals - median_res) / (1.4826 * mad)

        # Bounded-influence sigmoid activation clipped to [-20, 20]
        clipped_residuals = np.clip(studentized_residuals, -20.0, 20.0)
        pace_raw_weights = 1.0 / (1.0 + np.exp(-clipped_residuals))
        
        # Structural tree hierarchy damping factor: discounts nested sub-threads
        # to reflect their narrower conversational scope and lower visual reach
        depth_discount = 0.8 ** self.depths
        final_raw_w = pace_raw_weights * depth_discount
        
        norm_w = final_raw_w / np.sum(final_raw_w)
        consensus_score = float(np.sum(norm_w * self.sentiments))
        ess = float((np.sum(norm_w) ** 2) / np.sum(norm_w ** 2))

        observed_total = float(np.sum(self.likes))
        expected_total = float(np.sum(expected_likes))
        calibration_ratio = expected_total / observed_total if observed_total > 0 else None
        log_scale_r_squared = float(
            1.0 - np.sum(ols_residuals ** 2) / np.sum((log_likes - np.mean(log_likes)) ** 2)
        ) if 'ols_residuals' in locals() and np.sum((log_likes - np.mean(log_likes)) ** 2) > 0 else None

        return {
            "method": "PACE (Proposed)",
            "consensus_score": consensus_score,
            "weights": norm_w.tolist(),
            "expected_likes": expected_likes.tolist(),
            "residuals": residuals.tolist(),
            "majority_sentiment": self._majority_class(norm_w),
            "effective_sample_size": ess,
            "fitted_parameters": {
                "alpha": float(alpha_fit),
                "gamma_rank_decay": float(gamma_fit),
                "lambda_temporal_decay": float(lambda_fit),
                "duan_smearing_factor": float(duan_smearing_factor) if 'duan_smearing_factor' in locals() else 1.0
            },
            "mean_residual": float(np.mean(residuals)),
            "fit_diagnostics": {
                "observed_likes_total": observed_total,
                "expected_likes_total": expected_total,
                "expected_to_observed_ratio": calibration_ratio,
                "log_scale_r_squared": log_scale_r_squared,
                "warning": (
                    "Observed rank is a retrieval-time proxy, not historical exposure; "
                    "do not interpret this fit causally."
                ),
            },
        }

    # -------------------------------------------------------------
    # Comprehensive Comparative Evaluation
    # -------------------------------------------------------------
    def run_all_and_compare(self, random_seed: int | None = None) -> Dict[str, Any]:
        """Runs all 4 methods and performs statistical comparative analysis."""
        m1 = self.estimate_unweighted()
        m2 = self.estimate_heuristic_log_likes()
        m3 = self.estimate_position_ips()
        m4 = self.estimate_pace()

        # Compute 95% full-pipeline refitting bootstrap confidence intervals
        seed = self.random_seed if random_seed is None else random_seed
        for index, (m_key, m_val) in enumerate([("UnweightedMean", m1), ("HeuristicLogLikes", m2), ("PositionDebiasedIPS", m3), ("PACE_Proposed", m4)]):
            m_val["bootstrap_ci_95"] = self.compute_full_pipeline_bootstrap_ci(
                method=m_key, n_bootstraps=200, random_seed=seed + index
            )

        results = {
            "UnweightedMean": m1,
            "HeuristicLogLikes": m2,
            "PositionDebiasedIPS": m3,
            "PACE_Proposed": m4
        }

        # Compute robustness: Leave-Top-K-Out for top 10%
        robustness_scores = self._evaluate_robustness()

        # Compute Permutation Test: Is PACE significantly different from Unweighted / Heuristic?
        p_val_pace_vs_raw = self._permutation_divergence_test(np.array(m4["weights"]), np.array(m1["weights"]), random_seed=seed + 10)
        p_val_pace_vs_heur = self._permutation_divergence_test(np.array(m4["weights"]), np.array(m2["weights"]), random_seed=seed + 11)

        return {
            "num_comments": self.n,
            "methods": results,
            "robustness": robustness_scores,
            "significance_tests": {
                "pace_vs_unweighted_p_value": p_val_pace_vs_raw,
                "pace_vs_heuristic_p_value": p_val_pace_vs_heur,
            }
        }

    def _majority_class(self, weights: np.ndarray) -> str:
        """Determines weighted percentage for positive, negative, and neutral."""
        pos_mask = np.array([c.sentiment_label == "positive" for c in self.comments])
        neg_mask = np.array([c.sentiment_label == "negative" for c in self.comments])
        neu_mask = np.array([c.sentiment_label == "neutral" for c in self.comments])

        pos_w = np.sum(weights[pos_mask])
        neg_w = np.sum(weights[neg_mask])
        neu_w = np.sum(weights[neu_mask])

        if pos_w >= neg_w and pos_w >= neu_w:
            return f"positive ({pos_w*100:.1f}%)"
        elif neg_w >= pos_w and neg_w >= neu_w:
            return f"negative ({neg_w*100:.1f}%)"
        else:
            return f"neutral ({neu_w*100:.1f}%)"

    def _evaluate_robustness(self) -> Dict[str, float]:
        """
        Leave-Top-K-Out test: Drop the top 10% highest-engagement comments (using ceiling: ceil(0.10 * N))
        and measure how stable each method's consensus score remains.
        Stability rho = max(0.0, 1.0 - abs(score_full - score_pruned) / (std_dev + 1e-6))
        """
        k = max(1, int(math.ceil(self.n * 0.10)))
        top_indices = np.argsort(self.likes)[::-1][:k]
        keep_mask = np.ones(self.n, dtype=bool)
        keep_mask[top_indices] = False

        pruned_comments = [self.comments[i] for i in range(self.n) if keep_mask[i]]
        if len(pruned_comments) < 3:
            return {"unweighted_stability": 1.0, "heuristic_stability": 1.0, "pace_stability": 1.0}

        pruned_suite = ConsensusEstimatorSuite(pruned_comments, random_seed=self.random_seed)
        
        m_raw_orig = float(np.mean(self.sentiments))
        m_raw_prun = pruned_suite.estimate_unweighted()["consensus_score"]
        
        m_heur_orig = self.estimate_heuristic_log_likes()["consensus_score"]
        m_heur_prun = pruned_suite.estimate_heuristic_log_likes()["consensus_score"]

        m_pace_orig = self.estimate_pace()["consensus_score"]
        m_pace_prun = pruned_suite.estimate_pace()["consensus_score"]

        std_s = float(np.std(self.sentiments)) + 1e-6

        return {
            "unweighted_stability": round(max(0.0, 1.0 - abs(m_raw_orig - m_raw_prun) / std_s), 4),
            "heuristic_stability": round(max(0.0, 1.0 - abs(m_heur_orig - m_heur_prun) / std_s), 4),
            "pace_stability": round(max(0.0, 1.0 - abs(m_pace_orig - m_pace_prun) / std_s), 4)
        }

    def compute_full_pipeline_bootstrap_ci(self, method: str = "PACE_Proposed", n_bootstraps: int = 500,
                                           alpha: float = 0.05, random_seed: int | None = None) -> Tuple[float, float]:
        """
        Full-pipeline refitting bootstrap confidence interval (Davison & Hinkley 1997).
        On each bootstrap iteration:
        1. Resamples N comment records with replacement.
        2. Refits the entire estimation pipeline (OLS regression, Duan smearing, residuals, weights).
        3. Returns the 100*(1 - alpha)% percentile interval.
        """
        if self.n == 0:
            raise ValueError("At least one comment is required for bootstrapping.")
        if n_bootstraps < 1:
            raise ValueError("n_bootstraps must be positive.")
        rng = np.random.default_rng(self.random_seed if random_seed is None else random_seed)
        boot_estimates = np.zeros(n_bootstraps)
        indices = np.arange(self.n)

        for b in range(n_bootstraps):
            boot_idx = rng.choice(indices, size=self.n, replace=True)
            boot_comments = [self.comments[i] for i in boot_idx]
            boot_suite = ConsensusEstimatorSuite(boot_comments, random_seed=self.random_seed)

            if method == "PACE_Proposed":
                score = boot_suite.estimate_pace()["consensus_score"]
            elif method == "HeuristicLogLikes":
                score = boot_suite.estimate_heuristic_log_likes()["consensus_score"]
            elif method == "PositionDebiasedIPS":
                score = boot_suite.estimate_position_ips()["consensus_score"]
            else:
                score = boot_suite.estimate_unweighted()["consensus_score"]

            boot_estimates[b] = score

        ci_lower = float(np.percentile(boot_estimates, 100 * (alpha / 2.0)))
        ci_upper = float(np.percentile(boot_estimates, 100 * (1.0 - alpha / 2.0)))
        return round(ci_lower, 4), round(ci_upper, 4)

    def _permutation_divergence_test(self, weights_a: np.ndarray, weights_b: np.ndarray,
                                     n_permutations: int = 2000, random_seed: int | None = None) -> float:
        """
        Exact plus-one Monte Carlo permutation test (Ernst 2004) evaluating whether
        the difference in consensus estimates delta = |mu_A - mu_B| is statistically
        distinguishable under random exchangeability of comment sentiment valences.
        p_hat = (b + 1) / (B + 1)
        """
        est_a = float(np.sum(weights_a * self.sentiments))
        est_b = float(np.sum(weights_b * self.sentiments))
        observed_diff = abs(est_a - est_b)

        if n_permutations < 1:
            raise ValueError("n_permutations must be positive.")
        rng = np.random.default_rng(self.random_seed if random_seed is None else random_seed)
        b_count = 0
        s_copy = self.sentiments.copy()
        for _ in range(n_permutations):
            perm_s = rng.permutation(s_copy)
            perm_diff = abs(float(np.sum(weights_a * perm_s)) - float(np.sum(weights_b * perm_s)))
            if perm_diff >= observed_diff:
                b_count += 1

        return round((b_count + 1) / (n_permutations + 1), 5)
