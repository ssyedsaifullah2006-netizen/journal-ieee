"""
Analyzers package for social discourse and consensus estimation.
Core estimation modules (consensus_framework) operate with standard scientific Python (NumPy, SciPy).
Web platform NLP modules (sentiment, topics, summarizer) load lazily or gracefully fall back.
"""

from .consensus_framework import (
    ConsensusEstimatorSuite,
    CommentRecord,
    exposure_surface_model
)

# Optional dependencies for web platform RAG / summarization pipeline
try:
    from .sentiment import SentimentAnalyzer
except ImportError:
    SentimentAnalyzer = None

try:
    from .topics import TopicModeler
except ImportError:
    TopicModeler = None

try:
    from .summarizer import CommentSummarizer
except ImportError:
    CommentSummarizer = None

__all__ = [
    'ConsensusEstimatorSuite',
    'CommentRecord',
    'exposure_surface_model',
    'SentimentAnalyzer',
    'TopicModeler',
    'CommentSummarizer'
]
