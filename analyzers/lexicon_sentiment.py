"""
Lexicon-based sentiment analyzer using VADER.
Classifies each comment's sentiment polarity WITHOUT any AI/cloud API dependency.
Part of the Engagement-Calibrated Sentiment Diagnostic Framework.
"""

import math
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


@dataclass
class CommentSentiment:
    """Sentiment classification result for a single comment."""
    text: str
    label: str              # "positive", "negative", "neutral"
    compound: float         # VADER compound score [-1, +1]
    pos: float              # Positive proportion
    neg: float              # Negative proportion
    neu: float              # Neutral proportion
    likes: int = 0
    replies_count: int = 0
    author: str = ""
    timestamp: str = ""
    comment_id: str = ""
    engagement_weight: float = 1.0


# Threshold for VADER compound → label mapping
# (Standard recommendation from Hutto & Gilbert 2014)
COMPOUND_POSITIVE_THRESHOLD = 0.05
COMPOUND_NEGATIVE_THRESHOLD = -0.05


class LexiconSentimentAnalyzer:
    """
    VADER-based sentiment analyzer that classifies comments locally.
    
    This is the classification layer — it uses VADER as-is without modification.
    Our novel contribution starts AFTER classification (in engagement_analyzer.py).
    """

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def classify_comment(self, text: str) -> Dict[str, Any]:
        """
        Classify a single comment's sentiment using VADER.
        
        Returns:
            dict with keys: label, compound, pos, neg, neu
        """
        scores = self.sia.polarity_scores(text)
        compound = scores['compound']

        if compound >= COMPOUND_POSITIVE_THRESHOLD:
            label = "positive"
        elif compound <= COMPOUND_NEGATIVE_THRESHOLD:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "compound": compound,
            "pos": scores['pos'],
            "neg": scores['neg'],
            "neu": scores['neu'],
        }

    def analyze(self, comments: List[Dict[str, Any]]) -> List[CommentSentiment]:
        """
        Classify all comments.
        
        Args:
            comments: List of dicts, each with at least 'text'.
                      Optional: 'likes', 'replies_count', 'author', 'timestamp', 'comment_id'.
        
        Returns:
            List of CommentSentiment objects with classification + metadata.
        """
        results = []

        for c in comments:
            text = c.get("text", "")
            if not text or not text.strip():
                continue

            classification = self.classify_comment(text)

            # Reddit scores can be negative. Engagement is an exposure signal,
            # so negative scores must not be passed to log(1 + x).
            likes = max(0, c.get("likes", 0) or 0)
            replies_count = max(0, c.get("replies_count", 0) or 0)

            # Compute engagement weight: w = 1 + ln(1 + likes) + β * ln(1 + replies)
            beta = 0.5
            engagement_weight = (
                1.0
                + math.log(1 + likes)
                + beta * math.log(1 + replies_count)
            )

            results.append(CommentSentiment(
                text=text,
                label=classification["label"],
                compound=classification["compound"],
                pos=classification["pos"],
                neg=classification["neg"],
                neu=classification["neu"],
                likes=likes,
                replies_count=replies_count,
                author=c.get("author", ""),
                timestamp=c.get("timestamp", ""),
                comment_id=c.get("comment_id", ""),
                engagement_weight=engagement_weight,
            ))

        return results

    def get_raw_distribution(self, results: List[CommentSentiment]) -> Dict[str, Any]:
        """
        Compute the standard (unweighted) sentiment distribution.
        1 comment = 1 vote.
        """
        n = len(results)
        if n == 0:
            return {"positive": 0, "negative": 0, "neutral": 0,
                    "total": 0, "positive_pct": 0, "negative_pct": 0, "neutral_pct": 0}

        counts = {"positive": 0, "negative": 0, "neutral": 0}
        for r in results:
            counts[r.label] += 1

        return {
            "positive": counts["positive"],
            "negative": counts["negative"],
            "neutral": counts["neutral"],
            "total": n,
            "positive_pct": round(counts["positive"] / n, 4),
            "negative_pct": round(counts["negative"] / n, 4),
            "neutral_pct": round(counts["neutral"] / n, 4),
        }
