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

    ratings: (n_units, n_raters) array of integer category codes.
    Rows may be ragged (missing raters) via -1 entries; missing entries are
    skipped.
    """
    ratings = np.asarray(ratings, dtype=int)
    n_units, n_raters = ratings.shape
    k = ratings[ratings >= 0].max() + 1

    # per-unit: number of ratings and category counts
    n_i = (ratings >= 0).sum(axis=1)
    # P_i = (sum_j n_ij^2 - n_i) / (n_i*(n_i-1))
    P_i = np.zeros(n_units)
    for u in range(n_units):
        counts = np.bincount(ratings[u][ratings[u] >= 0], minlength=k)
        nu = n_i[u]
        if nu > 1:
            P_i[u] = (float(np.sum(counts ** 2)) - nu) / (nu * (nu - 1))
        else:
            P_i[u] = 1.0  # a single rating is trivially consistent
    P_bar = float(np.mean(P_i))
    # p_j = proportion of all assignments to category j
    p_j = np.zeros(k)
    for u in range(n_units):
        counts = np.bincount(ratings[u][ratings[u] >= 0], minlength=k)
        p_j += counts
    p_j = p_j / p_j.sum()
    P_e = float(np.sum(p_j ** 2))
    kappa = (P_bar - P_e) / (1 - P_e) if P_e < 1 else 0.0
    return float(kappa)


def krippendorff_alpha(ratings, level_of_measurement="nominal"):
    """Krippendorff's alpha.

    For the two-rater nominal case this is exactly Cohen's kappa, so it
    delegates and documents the equivalence. For >2 raters or ordinal/interval
    data, a full coincidence-matrix implementation is required and is
    intentionally not hand-rolled here (to avoid an unverified implementation).
    """
    ratings = np.asarray(ratings, dtype=int)
    n_units, n_raters = ratings.shape
    if level_of_measurement != "nominal":
        raise NotImplementedError(
            "Only the nominal level is provided. For ordinal/interval agreement "
            "use weighted_kappa (2 raters) or an external vetted implementation."
        )
    if n_raters == 2:
        # Krippendorff's alpha == Cohen's kappa for 2 raters + nominal data.
        a = ratings[:, 0]
        b = ratings[:, 1]
        mask = (a >= 0) & (b >= 0)
        return cohens_kappa(a[mask], b[mask])
    # >2 raters: Fleiss' kappa is the standard multi-rater nominal statistic.
    return fleiss_kappa(ratings)


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
