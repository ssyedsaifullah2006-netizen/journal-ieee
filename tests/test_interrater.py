"""Inter-rater reliability tests (correctness + edge cases).

Run:  python tests/test_interrater.py

Validates Krippendorff's alpha (coincidence-matrix, nominal), Fleiss' kappa,
Cohen's kappa, Scott's pi, and weighted kappa against analytic identities:
- nominal alpha == Scott's pi for two raters with no missing data,
- perfect agreement == 1, complete (binary) disagreement == -1,
- single-category (zero-variability) == NaN,
- missing ratings and 3+ raters handled.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from analyzers.interrater import (cohens_kappa, weighted_kappa, fleiss_kappa,
                                  krippendorff_alpha, percent_agreement, scotts_pi)

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


# 1. perfect agreement (2 raters)
a = np.column_stack([[0, 1, 2, 0, 1, 2] * 5, [0, 1, 2, 0, 1, 2] * 5]).astype(float)
check("perfect agreement (2 raters) == 1", krippendorff_alpha(a) == 1.0)

# 2. complete disagreement (binary)
c = np.column_stack([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]]).astype(float)
check("complete disagreement == -1", abs(krippendorff_alpha(c) + 1.0) < 1e-12)

# 3. partial agreement (2 raters) — finite, in [-1, 1]
p = np.column_stack([[0, 0, 1, 1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 1, 0, 0, 1, 1, 0, 1]]).astype(float)
v = krippendorff_alpha(p)
check("partial agreement in [-1,1]", -1.0 <= v <= 1.0 and not np.isnan(v))

# 4. missing ratings
m = np.array([[0.0, 0.0], [1.0, np.nan], [np.nan, 1.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
v = krippendorff_alpha(m)
check("missing ratings handled", -1.0 <= v <= 1.0 and not np.isnan(v))

# 5. two raters
check("two raters valid", abs(krippendorff_alpha(a) - 1.0) < 1e-12)

# 6. three raters
t = np.column_stack([[0, 1, 2, 0, 1, 2] * 5] * 3).astype(float)
check("three raters valid", krippendorff_alpha(t) == 1.0)

# 7. degenerate single category -> NaN
d = np.ones((10, 2))
check("single-category == NaN", np.isnan(krippendorff_alpha(d)))

# Identity: 2-rater nominal == Scott's pi (not Cohen's kappa)
rng = np.random.default_rng(0)
maxdiff = 0.0
for _ in range(100):
    n, k = int(rng.integers(5, 30)), int(rng.integers(2, 5))
    r1 = rng.integers(0, k, n); r2 = rng.integers(0, k, n)
    maxdiff = max(maxdiff, abs(krippendorff_alpha(np.column_stack([r1, r2]).astype(float))
                                - scotts_pi(list(r1), list(r2), list(range(k)))))
check("alpha == Scott's pi (2-rater, 100 datasets)", maxdiff < 1e-9)

# Cohen's kappa, Scott's pi, weighted kappa independently defined
# unbalanced rater marginals -> Cohen's kappa != Scott's pi (classic distinction)
x = ["a"] * 8 + ["b"] * 2
y = ["a"] * 2 + ["b"] * 8
check("cohens kappa != scotts pi (unbalanced marginals)", abs(cohens_kappa(x, y) - scotts_pi(x, y)) > 1e-6)
check("weighted kappa in [-1,1]", -1.0 <= weighted_kappa(x, y, "quadratic") <= 1.0)
check("percent agreement in [0,1]", 0.0 <= percent_agreement(x, y) <= 1.0)

# Fleiss' kappa: perfect -> 1, single-rating unit excluded
check("fleiss perfect == 1", fleiss_kappa(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 0, 0]])) == 1.0)
fk = fleiss_kappa(np.array([[0, 0, 0], [1, 1, 1], [2, -1, -1]]))
check("fleiss excludes single-rating units (no crash)", fk == 1.0)  # remaining 2 units agree perfectly

print(f"\nRESULT: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
