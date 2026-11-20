"""Reputation Analyzer: measure how an entity is talked about online.

Collects news articles, web pages or review exports that mention an entity,
scores every mention with a transformer sentiment model (or VADER), and turns
the results into a 0-100 reputation score with an HTML report.
"""

from .models import AnalysisResult, Document, Mention, SentimentScore
from .pipeline import AnalysisConfig, analyze

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "Document",
    "Mention",
    "SentimentScore",
    "analyze",
]

__version__ = "2.0.0"
