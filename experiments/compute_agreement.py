"""
Inter-annotator agreement and VADER accuracy against two independent raters.

Inputs
  data/youtube_corpus_100_comments_human_labeled.csv  (first annotator: column human_label,
                                                       VADER output: column vader_label)
  data/annotation_sheet_second_rater.csv              (second annotator: column label)

Outputs (stdout + data/inter_annotator_summary.json)
  raw agreement, Cohen's kappa (unweighted) with a bootstrap 95% CI, confusion matrix,
  list of disagreements, and VADER accuracy / macro-F1 against each rater, with Wilson 95% CIs.
"""
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["positive", "neutral", "negative"]


def wilson(k, n, z=1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(c - h, 3), round(c + h, 3)


def main():
    first = pd.read_csv(ROOT / "data/youtube_corpus_100_comments_human_labeled.csv")
    second = pd.read_csv(ROOT / "data/annotation_sheet_second_rater.csv")
    if second["label"].isna().any() or (second["label"].astype(str).str.strip() == "").any():
        sys.exit("The second rater's 'label' column is not complete yet.")
    second["label"] = second["label"].astype(str).str.strip().str.lower()
    bad = set(second["label"]) - set(LABELS)
    if bad:
        sys.exit(f"Invalid labels in second rater sheet: {bad}")
    df = first.merge(second[["comment_id", "label"]].rename(columns={"label": "rater2"}), on="comment_id", how="inner")
    if len(df) != len(first):
        sys.exit("comment_id mismatch between the two files.")

    r1, r2, v = df["human_label"], df["rater2"], df["vader_label"]
    n = len(df)
    kappa = cohen_kappa_score(r1, r2, labels=LABELS)
    rng = np.random.default_rng(20261030)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        a, b = r1.iloc[idx], r2.iloc[idx]
        if a.nunique() == 1 and b.nunique() == 1:
            continue
        boots.append(cohen_kappa_score(a, b, labels=LABELS))
    out = {
        "n": n,
        "raw_agreement": round(float((r1 == r2).mean()), 4),
        "cohen_kappa": round(float(kappa), 4),
        "cohen_kappa_bootstrap_95ci": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
        "confusion_rater1_rows_rater2_cols": confusion_matrix(r1, r2, labels=LABELS).tolist(),
        "vader_vs_rater1": {"accuracy": round(accuracy_score(r1, v), 4), "macro_f1": round(f1_score(r1, v, labels=LABELS, average="macro", zero_division=0), 4),
                            "wilson95": wilson(int((r1 == v).sum()), n)},
        "vader_vs_rater2": {"accuracy": round(accuracy_score(r2, v), 4), "macro_f1": round(f1_score(r2, v, labels=LABELS, average="macro", zero_division=0), 4),
                            "wilson95": wilson(int((r2 == v).sum()), n)},
        "label_counts_rater1": r1.value_counts().to_dict(),
        "label_counts_rater2": r2.value_counts().to_dict(),
    }
    print(json.dumps(out, indent=2))
    dis = df[r1 != r2][["comment_id", "human_label", "rater2", "text"]]
    print(f"\n{len(dis)} disagreements:")
    print(dis.to_string(index=False, max_colwidth=90))
    (ROOT / "data/inter_annotator_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
