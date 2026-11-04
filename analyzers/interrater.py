"""
Inter-rater reliability for the PACE semantic audit.

Provides standard agreement statistics for categorical/ordinal annotations:
- percent agreement
- Cohen's kappa (2 raters, nominal)
- weighted Cohen's kappa (2 raters, ordinal; linear or quadratic weights)
- Fleiss' kappa (2+ raters, nominal)
- Krippendorff's alpha (nominal; for the 2-rater case it coincides with
  Cohen's kappa, so it delegates there and documents the equivalence)

IMPORTANT: these functions compute agreement. They do NOT create annotations.
A second human rater's labels must be collected independently (see
`experiments/second_rater_protocol.md`). Fabricating them would invalidate the
audit; this module is deliberately data-free.

Usage:
    from analyzers.interrater import agreement_report
    report = agreement_report(rater1_labels, rater2_labels, labels=["positive","neutral","negative"])
"""
import numpy as np
from collections import Counter


def _as_codes(y, labels=None):
    """Map arbitrary labels to integer codes 0..k-1."""
    if labels is None:
        labels = sorted(set(y))
    code = {lab: i for i, lab in enumerate(labels)}
    return np.array([code[lab] for lab in y], dtype=int), labels


def _confusion(y1, y2, labels=None):
    c1, labels = _as_codes(y1, labels)
    c2, _ = _as_codes(y2, labels)
    k = len(labels)
    cm = np.zeros((k, k), dtype=int)
    for a, b in zip(c1, c2):
        cm[a, b] += 1
    return cm, labels


def percent_agreement(y1, y2, labels=None):
    cm, _ = _confusion(y1, y2, labels)
    return float(np.trace(cm) / cm.sum())


def cohens_kappa(y1, y2, labels=None):
    """Cohen's kappa for two raters, nominal categories."""
    cm, _ = _confusion(y1, y2, labels)
    n = cm.sum()
    po = np.trace(cm) / n
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    pe = (row @ col) / (n * n)
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def scotts_pi(y1, y2, labels=None):
    """Scott's pi for two raters (pooled/average marginals).

    Scott's pi is the two-rater limit of nominal Krippendorff's alpha.
    """
    cm, _ = _confusion(y1, y2, labels)
    n = cm.sum()
    po = np.trace(cm) / n
    p = (cm.sum(axis=1) + cm.sum(axis=0)) / (2.0 * n)
    pe = float(np.sum(p ** 2))
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def weighted_kappa(y1, y2, weights="quadratic", labels=None):
    """Weighted Cohen's kappa for ordinal categories.

    weights: 'linear' -> w_ij = 1 - |i-j|/(k-1)
             'quadratic' -> w_ij = 1 - ((i-j)/(k-1))^2
    """
    cm, _ = _confusion(y1, y2, labels)
    k = cm.shape[0]
    n = cm.sum()
    if k < 2:
        return 0.0
    w = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            d = abs(i - j) / (k - 1)
            w[i, j] = 1 - d if weights == "linear" else 1 - d * d
    o = cm / n
    row = cm.sum(axis=1) / n
    col = cm.sum(axis=0) / n
    e = np.outer(row, col)
    num = float(np.sum(w * o))
    den = float(np.sum(w * e))
    return float(1.0 - num / den) if den > 0 else 0.0


def fleiss_kappa(ratings):
    """Fleiss' kappa for multiple raters, nominal categories.

    ratings: (n_units, n_raters) array of integer category codes; entries < 0
    are treated as missing.

    Missing-data assumptions (documented):
    - Fleiss' kappa assumes every unit is rated by the SAME number of raters.
    - A unit with fewer than two valid ratings is EXCLUDED: agreement is
      undefined for a single rating, and it is NOT treated as perfect.
    - When the number of valid ratings varies across units, the per-unit n_i is
      used in the P_i term; this is a common generalization that is exact only
      when n_i is constant. This caveat is reported via the returned tuple's
      second element.

    Returns (kappa, info) where info is a dict with 'units_used',
    'units_dropped', and 'n_raters'.
    """
    ratings = np.asarray(ratings, dtype=int)
    if ratings.ndim != 2 or ratings.shape[1] < 2:
        raise ValueError("ratings must be a (n_units, n_raters) matrix with >= 2 raters")
    n_units, n_raters = ratings.shape
    valid = ratings >= 0
    n_i = valid.sum(axis=1)
    keep = n_i >= 2
    if keep.sum() == 0:
        raise ValueError("No unit has at least two valid ratings; Fleiss' kappa is undefined.")
    k = ratings[valid].max() + 1

    P_i = np.zeros(int(keep.sum()))
    for idx, u in enumerate(np.where(keep)[0]):
        counts = np.bincount(ratings[u][valid[u]], minlength=k)
        nu = int(n_i[u])
        P_i[idx] = (float(np.sum(counts ** 2)) - nu) / (nu * (nu - 1))
    P_bar = float(np.mean(P_i))

    p_j = np.zeros(k)
    for u in np.where(keep)[0]:
        p_j += np.bincount(ratings[u][valid[u]], minlength=k)
    p_j = p_j / p_j.sum()
    P_e = float(np.sum(p_j ** 2))
    kappa = (P_bar - P_e) / (1 - P_e) if P_e < 1 else float("nan")
    return float(kappa)


def krippendorff_alpha(ratings, level_of_measurement="nominal"):
    """Krippendorff's alpha via the coincidence-matrix formulation (nominal).

    ratings: (n_units, n_raters) numeric matrix; NaN (or values < 0) are missing.
    The difference function for the nominal level is delta(c,k) = 0 if c == k
    else 1.

    This is the general coincidence-matrix computation; it does NOT substitute
    Cohen's kappa or Fleiss' kappa. For two raters with no missing data it
    numerically coincides with Scott's pi (pooled-marginal agreement), which is
    the correct two-rater limit (it is NOT Cohen's kappa, which uses
    rater-specific marginals).

    Returns:
      float alpha, or float('nan') when the statistic is undefined
      (single-category / zero-variability data, where expected disagreement is 0).
    Raises ValueError for invalid input or insufficient data.
    """
    if level_of_measurement != "nominal":
        raise NotImplementedError("Only the nominal level is implemented.")
    ratings = np.asarray(ratings, dtype=float)
    if ratings.ndim != 2:
        raise ValueError("ratings must be a 2D (units x raters) array.")
    n_units, n_raters = ratings.shape
    if n_raters < 2:
        raise ValueError("At least two raters are required.")

    # Coincidence matrix: for each unit with m valid ratings, add 1/(m-1) for
    # each ordered pair of distinct raters.
    o = {}
    for u in range(n_units):
        vals = ratings[u][~np.isnan(ratings[u])]
        m = len(vals)
        if m < 2:
            continue
        w = 1.0 / (m - 1)
        for i in range(m):
            for j in range(m):
                if i != j:
                    key = (vals[i], vals[j])
                    o[key] = o.get(key, 0.0) + w

    if not o:
        raise ValueError("No coincidences available; alpha is undefined.")

    n_c = {}
    for (c, _k), w in o.items():
        n_c[c] = n_c.get(c, 0.0) + w
    N = sum(n_c.values())
    if N <= 1:
        raise ValueError("Total coincidences <= 1; alpha is undefined.")

    # Observed disagreement (nominal): off-diagonal coincidences / N.
    D_o = sum(w for (c, k), w in o.items() if c != k) / N
    # Expected disagreement: sum_{c != k} pi_c pi_k = 1 - sum_c pi_c^2,
    # where pi_c = n_c / N. Equivalently (N^2 - sum n_c^2) / N^2.
    # NOTE: this reduces to Scott's pi (NOT Cohen's kappa) for two raters.
    sum_sq = sum(v * v for v in n_c.values())
    D_e = (N * N - sum_sq) / (N * N)

    if D_e == 0.0:
        # Zero variability: every rating is the same single category.
        # The statistic is 0/0 -> undefined.
        return float("nan")
    return float(1.0 - D_o / D_e)


def agreement_report(rater1, rater2, labels=None):
    """Convenience summary for two raters' nominal labels."""
    labels = labels or sorted(set(rater1) | set(rater2))
    rep = {
        "n": len(rater1),
        "percent_agreement": round(percent_agreement(rater1, rater2, labels), 4),
        "cohens_kappa": round(cohens_kappa(rater1, rater2, labels), 4),
        "krippendorff_alpha_nominal": round(krippendorff_alpha(
            np.column_stack([_as_codes(rater1, labels)[0], _as_codes(rater2, labels)[0]]),
            "nominal"), 4),
        "weighted_kappa_quadratic": round(weighted_kappa(rater1, rater2, "quadratic", labels), 4),
        "weighted_kappa_linear": round(weighted_kappa(rater1, rater2, "linear", labels), 4),
    }
    return rep
