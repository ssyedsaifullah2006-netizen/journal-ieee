"""Generate a BLIND second-rater labeling sheet (comment text only, no labels)."""
import csv
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
src = ROOT / "data" / "youtube_corpus_100_comments.csv"
out = ROOT / "data" / "second_rater_template.csv"

df = pd.read_csv(src)
rows = [{"comment_id": r["comment_id"], "text": r["text"], "rater2_label": ""}
        for _, r in df.iterrows()]
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["comment_id", "text", "rater2_label"])
    w.writeheader()
    w.writerows(rows)
print(f"Wrote blind template ({len(rows)} rows) -> {out}")
