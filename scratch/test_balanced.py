import numpy as np
import sys
sys.path.insert(0, r"c:\VIT\Semester5\ai\prjt")

from analyzers.consensus_framework import ConsensusEstimatorSuite, CommentRecord

np.random.seed(42)
# 25 positive, 25 negative -> strictly zero ground truth Y* = 0.000
raw_s = np.array([0.5]*25 + [-0.5]*25)
perm_s = np.random.permutation(raw_s)

comments = []
for i in range(50):
    s = float(perm_s[i])
    lbl = "positive" if s > 0 else "negative"
    expected_l = int(1000 * ((i + 1) ** (-0.7)) * ((1 + i * 0.3) ** (-0.2)))
    noise_l = max(0, expected_l + int(np.random.randint(-15, 15)))
    comments.append(CommentRecord(
        comment_id=f"c_{i}",
        text=f"Comment {i}",
        sentiment_score=s,
        sentiment_label=lbl,
        likes=noise_l,
        replies=int(np.random.randint(0, 5)),
        rank=i + 1,
        arrival_hours=i * 0.3,
        depth=0
    ))

suite = ConsensusEstimatorSuite(comments)
res = suite.run_all_and_compare()

print("Unweighted:", res["methods"]["UnweightedMean"]["consensus_score"])
print("Heuristic:", res["methods"]["HeuristicLogLikes"]["consensus_score"])
print("NC-IPS:", res["methods"]["PositionDebiasedIPS"]["consensus_score"])
print("PACE:", res["methods"]["PACE_Proposed"]["consensus_score"])
print("p-value PACE vs Heuristic:", res["significance_tests"]["pace_vs_heuristic_p_value"])
print("MAE against Ground Truth 0.000:")
print("  Unweighted MAE:", abs(res["methods"]["UnweightedMean"]["consensus_score"] - 0.0))
print("  Heuristic MAE:", abs(res["methods"]["HeuristicLogLikes"]["consensus_score"] - 0.0))
print("  PACE MAE:", abs(res["methods"]["PACE_Proposed"]["consensus_score"] - 0.0))
