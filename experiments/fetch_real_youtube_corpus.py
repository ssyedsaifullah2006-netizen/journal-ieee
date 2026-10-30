"""
Fetch REAL YouTube comments from dQw4w9WgXcQ using YouTube Data API v3,
score them with VADER, and save as the empirical corpus CSV.
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

# Load API key
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

VIDEO_ID = "dQw4w9WgXcQ"
MAX_COMMENTS = 100
COLLECTION_METHOD = "youtube.commentThreads.list(order=relevance, top-level comments only)"

# --- Video publish time (known: Oct 25, 2009) ---
VIDEO_PUBLISH = datetime(2009, 10, 25, 6, 57, 33, tzinfo=timezone.utc)


def fetch_comments(video_id, api_key, max_results=100):
    """Fetch top-level comments sorted by relevance using YouTube Data API v3."""
    comments = []
    page_token = None
    per_page = min(max_results, 100)

    while len(comments) < max_results:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "order": "relevance",
            "maxResults": per_page,
            "textFormat": "plainText",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token

        url = "https://www.googleapis.com/youtube/v3/commentThreads?" + urlencode(params)
        req = Request(url)
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "author": snippet["authorDisplayName"],
                "timestamp": snippet["publishedAt"],
                "likes": snippet["likeCount"],
                "replies": item["snippet"]["totalReplyCount"],
                "text": snippet["textDisplay"].replace("\n", " ").strip(),
            })
            if len(comments) >= max_results:
                break

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return comments[:max_results]


def score_with_vader(comments):
    """Add VADER sentiment scores to each comment dict."""
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    for c in comments:
        scores = sia.polarity_scores(c["text"])
        c["sentiment_score"] = round(scores["compound"], 4)
        if scores["compound"] >= 0.05:
            c["sentiment_label"] = "positive"
        elif scores["compound"] <= -0.05:
            c["sentiment_label"] = "negative"
        else:
            c["sentiment_label"] = "neutral"
    return comments


def compute_arrival_hours(comments, video_publish_dt):
    """Compute hours between video publish and comment timestamp."""
    for c in comments:
        ts_str = c["timestamp"]
        # Parse ISO 8601
        comment_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        delta = comment_dt - video_publish_dt
        c["arrival_hours"] = round(max(0.0, delta.total_seconds() / 3600.0), 2)
    return comments


def main():
    if not API_KEY:
        print("ERROR: YOUTUBE_API_KEY not found in .env")
        sys.exit(1)

    print(f"Fetching {MAX_COMMENTS} comments from video {VIDEO_ID} ...")
    comments = fetch_comments(VIDEO_ID, API_KEY, MAX_COMMENTS)
    print(f"  Fetched {len(comments)} comments.")

    if len(comments) == 0:
        print("ERROR: No comments returned. Check API key and quota.")
        sys.exit(1)

    # Assign display rank (1-indexed, order returned by API = relevance rank)
    for i, c in enumerate(comments):
        c["rank"] = i + 1

    # Compute arrival hours from video publish
    comments = compute_arrival_hours(comments, VIDEO_PUBLISH)

    # Score with VADER
    print("  Running VADER sentiment analysis ...")
    comments = score_with_vader(comments)

    # Determine reply depth (all top-level = 0)
    for c in comments:
        c["reply_depth"] = 0

    # Save the derived analysis table. This is a retrieval-time snapshot: the API
    # does not expose historical display impressions or historical display rank.
    out_path = ROOT / "data" / "youtube_corpus_100_comments.csv"
    fieldnames = ["comment_id", "author", "timestamp", "rank", "arrival_hours",
                  "likes", "replies", "sentiment_score", "sentiment_label", "text"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for c in comments:
            writer.writerow(c)

    # A provenance manifest makes a future corpus reviewable without pretending
    # that a CSV alone proves where its contents originated.
    csv_sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "source_platform": "YouTube Data API v3",
        "video_id": VIDEO_ID,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection_method": COLLECTION_METHOD,
        "ranking_interpretation": (
            "retrieval-time relevance order returned by the API; it is not a "
            "historical display-rank or impression measurement"
        ),
        "comment_scope": "top-level comments only",
        "requested_comment_count": MAX_COMMENTS,
        "returned_comment_count": len(comments),
        "video_publish_time_utc": VIDEO_PUBLISH.isoformat(),
        "sentiment_method": "NLTK VADER compound score; thresholds +/-0.05",
        "csv_file": out_path.name,
        "csv_sha256": csv_sha256,
    }
    manifest_path = out_path.with_suffix(".provenance.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"  Saved to: {out_path}")
    print(f"  Saved provenance manifest to: {manifest_path}")

    # Print summary stats
    likes = [c["likes"] for c in comments]
    scores = [c["sentiment_score"] for c in comments]
    arrivals = [c["arrival_hours"] for c in comments]
    unique_authors = len(set(c["author"] for c in comments))

    print(f"\n--- Corpus Summary ---")
    print(f"  N = {len(comments)}")
    print(f"  Unique authors: {unique_authors}")
    print(f"  Likes range: {min(likes)} - {max(likes)}")
    print(f"  Sentiment range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"  Unique sentiment values: {len(set(scores))}")
    print(f"  Arrival hours range: {min(arrivals):.2f} - {max(arrivals):.2f}h")
    print(f"  Zero-like comments: {sum(1 for l in likes if l == 0)}")
    print(f"  Sample authors: {[c['author'] for c in comments[:5]]}")
    print(f"  Likes monotonically decreasing: {all(likes[i] >= likes[i+1] for i in range(len(likes)-1))}")
    print(f"  DONE.")


if __name__ == "__main__":
    main()
