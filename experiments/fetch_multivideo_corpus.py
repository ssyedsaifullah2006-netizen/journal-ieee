"""
Multi-Video Observational Corpus: fetch real YouTube comments from four diverse
videos, score them with VADER, and apply the four consensus estimators.

This extends the single-video (Rick Astley) case study to a four-domain
observational comparison. Each run produces dated, SHA-256-hashed provenance so
the snapshot is reviewable. Re-running re-fetches live comments (like counts and
relevance order may drift), which is expected for a dated observational corpus.

Usage:  python experiments/fetch_multivideo_corpus.py
"""
import os
import sys
import json
import csv
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.parse import urlencode

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

# (video_id, short_label)
VIDEOS = [
    ("dQw4w9WgXcQ", "rick_astley_meme"),
    ("Jf_Sz27I458", "smartphone_30k"),
    ("OMZ_HLTJZRQ", "iphone_review"),
    ("LZYfSYXY-go", "thin_device_reveal"),
]
MAX_COMMENTS = 100


def api(path, params):
    params = dict(params)
    params["key"] = API_KEY
    url = f"https://www.googleapis.com/youtube/v3/{path}?" + urlencode(params)
    with urlopen(Request(url), timeout=30) as r:
        return json.loads(r.read().decode())


def get_meta(vid):
    d = api("videos", {"part": "snippet,statistics", "id": vid})
    it = d["items"][0]
    return it["snippet"]["publishedAt"], it["snippet"]["title"], it["statistics"].get("commentCount", "?")


def fetch_comments(vid, n=MAX_COMMENTS):
    out, tok = [], None
    while len(out) < n:
        p = {"part": "snippet", "videoId": vid, "order": "relevance",
             "maxResults": min(100, n - len(out)), "textFormat": "plainText"}
        if tok:
            p["pageToken"] = tok
        d = api("commentThreads", p)
        for it in d.get("items", []):
            s = it["snippet"]["topLevelComment"]["snippet"]
            out.append({
                "comment_id": it["snippet"]["topLevelComment"]["id"],
                "author": s["authorDisplayName"],
                "timestamp": s["publishedAt"],
                "likes": s["likeCount"],
                "replies": it["snippet"]["totalReplyCount"],
                "text": s["textDisplay"].replace("\n", " ").strip(),
            })
        tok = d.get("nextPageToken")
        if not tok:
            break
    return out[:n]


def score_vader(comments):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    for c in comments:
        sc = sia.polarity_scores(c["text"])["compound"]
        c["sentiment_score"] = round(sc, 4)
        c["sentiment_label"] = "positive" if sc >= 0.05 else ("negative" if sc <= -0.05 else "neutral")
    return comments


def main():
    if not API_KEY:
        print("ERROR: YOUTUBE_API_KEY not found in .env")
        sys.exit(1)

    from analyzers.consensus_framework import ConsensusEstimatorSuite, CommentRecord

    retrieved_at = datetime.now(timezone.utc).isoformat()
    summary_rows = []
    manifest_videos = []

    import pandas as _pd
    for vid, label in VIDEOS:
        if label == "rick_astley_meme":
            # Reuse the dated 2026-10-30 corpus (human-labeled, with provenance)
            # rather than re-fetching, to stay consistent with Tables I/III.
            _df = _pd.read_csv(ROOT / "data" / "youtube_corpus_100_comments.csv")
            comments = _df.to_dict("records")
            title = "Rick Astley - Never Gonna Give You Up"
            ccount = len(comments)
            published = "2009-10-25T06:57:33+00:00"  # from the 2026-10-30 provenance
            records = [CommentRecord(str(c["comment_id"]), str(c["text"]),
                                     float(c["sentiment_score"]), str(c["sentiment_label"]),
                                     max(0, int(c["likes"])), int(c["replies"]),
                                     int(c["rank"]), float(c["arrival_hours"]), 0)
                       for c in comments]
        else:
            print(f"\n[{label}] fetching {MAX_COMMENTS} comments from {vid} ...")
            published, title, ccount = get_meta(vid)
            pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            comments = fetch_comments(vid, MAX_COMMENTS)
            comments = score_vader(comments)
            records = []
            for i, c in enumerate(comments):
                cd = datetime.fromisoformat(c["timestamp"].replace("Z", "+00:00"))
                arr = max(0.0, (cd - pub_dt).total_seconds() / 3600.0)
                c["rank"] = i + 1
                c["arrival_hours"] = round(arr, 2)
                records.append(CommentRecord(
                    c["comment_id"], c["text"], c["sentiment_score"], c["sentiment_label"],
                    max(0, int(c["likes"])), int(c["replies"]), i + 1, arr, 0))

        suite = ConsensusEstimatorSuite(records)
        u = suite.estimate_unweighted()
        h = suite.estimate_heuristic_log_likes()
        ip = suite.estimate_position_ips()       # IPE
        nips = suite.estimate_normalized_ips()   # genuine NIPS
        p = suite.estimate_pace()

        dist = {"positive": 0, "neutral": 0, "negative": 0}
        for r in records:
            dist[r.sentiment_label] += 1
        zero = sum(1 for r in records if r.sentiment_score == 0.0)

        row = {
            "video_label": label, "video_id": vid, "video_title": title,
            "n_comments": len(records), "video_comment_count": ccount,
            "vader_pos": dist["positive"], "vader_neu": dist["neutral"], "vader_neg": dist["negative"],
            "zero_inflation_count": zero, "zero_inflation_pct": round(zero / len(records), 4),
            "mu_unweighted": round(u["consensus_score"], 4),
            "mu_heuristic": round(h["consensus_score"], 4),
            "mu_ipe": round(ip["consensus_score"], 4),
            "mu_nips": round(nips["consensus_score"], 4),
            "mu_pace": round(p["consensus_score"], 4),
            "wmax_heuristic": round(float(max(h["weights"])), 4),
            "wmax_pace": round(float(max(p["weights"])), 4),
            "ess_pace": round(p["effective_sample_size"], 1),
        }
        summary_rows.append(row)

        # per-video comments CSV
        csv_path = ROOT / "data" / f"multivideo_{label}.csv"
        fields = ["comment_id", "author", "timestamp", "rank", "arrival_hours",
                  "likes", "replies", "sentiment_score", "sentiment_label", "text"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for c in comments:
                w.writerow(c)
        sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        manifest_videos.append({"label": label, "video_id": vid, "video_title": title,
                                "published_at_utc": published, "csv_file": csv_path.name,
                                "csv_sha256": sha, "n_comments": len(records)})

        print(f"  fetched={len(records)}  dist={dist}  zero={zero}  "
              f"mu_unw={u['consensus_score']:+.3f} mu_heur={h['consensus_score']:+.3f} "
              f"mu_ips={ip['consensus_score']:+.3f} mu_pace={p['consensus_score']:+.3f}")

    # summary CSV
    import pandas as pd
    summary_path = ROOT / "data" / "multivideo_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    manifest = {
        "schema_version": 1,
        "source_platform": "YouTube Data API v3",
        "retrieved_at_utc": retrieved_at,
        "collection_method": "youtube.commentThreads.list(order=relevance, top-level comments only)",
        "ranking_interpretation": "retrieval-time relevance order; not historical display rank",
        "sentiment_method": "NLTK VADER compound score; thresholds +/-0.05",
        "videos": manifest_videos,
    }
    manifest_path = ROOT / "data" / "multivideo_provenance.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\nSaved summary -> {summary_path}")
    print(f"Saved provenance -> {manifest_path}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
