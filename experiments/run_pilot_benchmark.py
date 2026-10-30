"""
Pilot Benchmark Script for Engagement-Calibrated Sentiment Diagnostics.
Run this script across multiple YouTube videos to collect empirical data for the journal paper.
Generates a markdown and CSV table of:
- Video Title / Category
- Total Comments Fetched
- Raw Majority vs. Weighted Majority
- Flip Occurrence (Phi)
- Permutation Test p-value (Statistical Significance)
- Reliability Score (R) & Gini Index
- Robustness Score (rho)
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path

# Ensure root directory is on path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fetchers.youtube_fetcher import YouTubeFetcher
from analyzers.lexicon_sentiment import LexiconSentimentAnalyzer
from analyzers.engagement_diagnostics import EngagementDiagnostics


def run_benchmark(video_urls: list, max_comments_per_video: int = 250):
    fetcher = YouTubeFetcher()
    analyzer = LexiconSentimentAnalyzer()
    diagnostics = EngagementDiagnostics(n_permutations=1000)

    results = []

    print(f"Starting pilot benchmark across {len(video_urls)} videos...")

    for i, url in enumerate(video_urls, 1):
        print(f"\n[{i}/{len(video_urls)}] Processing: {url}")
        try:
            metadata = fetcher.get_metadata(url)
            title = metadata.get("title", f"Video {i}")
            print(f"  Title: {title}")

            comments = fetcher.fetch_comments(url, max_comments=max_comments_per_video)
            print(f"  Fetched {len(comments)} comments.")

            if not comments:
                print("  No comments found, skipping.")
                continue

            comments_data = [vars(c) for c in comments]
            classified = analyzer.analyze(comments_data)
            raw_dist = analyzer.get_raw_distribution(classified)

            report = diagnostics.run_full_diagnostics(classified, raw_dist)

            raw = report["raw_distribution"]
            weighted = report["weighted_distribution"]
            flip = report["majority_flip_test"]
            rel = report["reliability_metric"]
            rob = report["robustness_analysis"]

            row = {
                "Title": title[:40],
                "Count": len(comments),
                "Raw Majority": flip.get("raw_majority", "").capitalize(),
                "Raw Pos %": round(raw.get("positive_pct", 0) * 100, 1),
                "Raw Neg %": round(raw.get("negative_pct", 0) * 100, 1),
                "Weighted Majority": flip.get("weighted_majority", "").capitalize(),
                "Weighted Pos %": round(weighted.get("positive", 0) * 100, 1),
                "Weighted Neg %": round(weighted.get("negative", 0) * 100, 1),
                "Majority Flip": "YES" if flip.get("flip_detected") else "NO",
                "p-value": flip.get("p_value", 1.0),
                "Significant": "YES" if flip.get("is_significant") else "NO",
                "Reliability (R)": rel.get("r_score", 0.0),
                "Gini": rel.get("gini", 0.0),
                "Robustness (rho)": rob.get("robustness_score", 1.0)
            }
            results.append(row)
            print(f"  -> Flip: {row['Majority Flip']} (p={row['p-value']}), R={row['Reliability (R)']}, rho={row['Robustness (rho)']}")

        except Exception as e:
            print(f"  Error processing {url}: {e}")

    if results:
        df = pd.DataFrame(results)
        output_dir = ROOT_DIR / "data"
        output_dir.mkdir(exist_ok=True)
        
        csv_path = output_dir / "pilot_benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[SUCCESS] Benchmark completed! Saved results to {csv_path}")

        print("\n=== SUMMARY TABLE FOR JOURNAL PAPER ===")
        print(df.to_markdown(index=False))
        return df
    else:
        print("\nNo results gathered.")
        return None


if __name__ == "__main__":
    # Add any real YouTube video URLs here to run your pilot
    # (Prefer videos with varied audience reception: product reviews, updates, debates)
    SAMPLE_URLS = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Classic music video
    ]
    run_benchmark(SAMPLE_URLS, max_comments_per_video=100)
