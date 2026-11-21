from datetime import datetime, timedelta, timezone

import pytest

from reputation_analyzer.models import Document, Mention
from reputation_analyzer.scoring import (
    band_for,
    reputation_score,
    score_documents,
    summarize,
)
from reputation_analyzer.sentiment import scores_from_probs

NOW = datetime(2026, 11, 20, tzinfo=timezone.utc)


def doc(*probs, source="a.com", days_ago=None):
    d = Document(
        url=f"https://{source}/{id(probs)}",
        source=source,
        published=NOW - timedelta(days=days_ago) if days_ago is not None else None,
    )
    d.mentions = [Mention(text="x", sentiment=scores_from_probs(*p)) for p in probs]
    return d


POS = (0.05, 0.15, 0.80)
NEG = (0.80, 0.15, 0.05)
NEU = (0.10, 0.80, 0.10)


def test_score_reflects_the_data_regression_for_old_minmax_bug():
    # The old notebook min-max scaled the class averages against each other, so
    # mostly-positive and mostly-negative coverage could look the same.
    positive = [doc(POS) for _ in range(20)]
    negative = [doc(NEG) for _ in range(20)]
    score_documents(positive + negative)
    assert reputation_score(positive) > 75
    assert reputation_score(negative) < 25


def test_neutral_documents_score_fifty():
    docs = [doc(NEU) for _ in range(10)]
    score_documents(docs)
    assert reputation_score(docs) == pytest.approx(50.0)


def test_prior_shrinks_small_samples():
    one = [doc(POS)]
    score_documents(one)
    many = [doc(POS) for _ in range(50)]
    score_documents(many)
    assert reputation_score(one) == pytest.approx(50 + 50 * 0.75 / 3)
    assert reputation_score(one) < reputation_score(many)
    assert reputation_score(one, prior_weight=0) == pytest.approx(87.5)


def test_document_is_mean_of_mentions_so_long_articles_count_once():
    long_neg = doc(*([NEG] * 20))
    short_pos = doc(POS)
    score_documents([long_neg, short_pos])
    assert reputation_score([long_neg, short_pos], prior_weight=0) == pytest.approx(50.0)


def test_half_life_weights_recent_documents_more():
    old_neg = doc(NEG, days_ago=365)
    new_pos = doc(POS, days_ago=1)
    docs = [old_neg, new_pos]
    score_documents(docs)
    flat = reputation_score(docs, prior_weight=0, now=NOW)
    recent = reputation_score(docs, prior_weight=0, half_life_days=30, now=NOW)
    assert flat == pytest.approx(50.0)
    assert recent > 80


def test_documents_without_mentions_are_not_scored():
    d = Document(url="https://x.com/1")
    score_documents([d])
    assert d.polarity is None and d.label is None
    assert reputation_score([d]) is None


def test_summarize_shapes():
    docs = [
        doc(POS, source="a.com", days_ago=3),
        doc(NEG, source="b.com", days_ago=10),
        doc(POS, source="a.com", days_ago=10),
        Document(url="https://c.com/none"),
    ]
    score_documents(docs)
    s = summarize(docs, now=NOW)
    assert s["documents_total"] == 4
    assert s["documents_scored"] == 3
    assert s["documents_without_mentions"] == 1
    assert s["label_counts"] == {"positive": 2, "neutral": 0, "negative": 1}
    assert s["by_source"][0] == {"source": "a.com", "documents": 2, "score": 87.5}
    assert len(s["timeline"]) == 2
    assert s["most_negative"][0]["source"] == "b.com"
    assert s["most_positive"][0]["polarity"] > 0
    assert s["confidence"] == "low"
    assert s["band"] == band_for(s["reputation_score"])


@pytest.mark.parametrize(
    "score,band", [(80, "Strong"), (60, "Positive"), (50, "Mixed"), (40, "At risk"), (10, "Poor")]
)
def test_bands(score, band):
    assert band_for(score) == band


def test_top_passages_are_deduplicated():
    docs = [doc(NEG, source=f"s{i}.com") for i in range(4)]
    score_documents(docs)
    s = summarize(docs, now=NOW)
    assert len(s["most_negative"]) == 1  # same syndicated sentence on four sites
    assert s["most_positive"] == []
