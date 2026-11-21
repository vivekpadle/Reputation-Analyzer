"""Turn mention-level sentiment into document scores and a reputation score.

How the score works
-------------------
1. Every mention gets a polarity in [-1, 1] (positive prob - negative prob).
2. A document's polarity is the mean of its mentions, so a long article with
   20 mentions counts once, the same as a short review.
3. The reputation score is a weighted mean of document polarities, mapped to
   0-100 (50 = neutral), and shrunk toward 50 when there is little data::

       score = 50 + 50 * sum(w_i * p_i) / (sum(w_i) + prior_weight)

   With ``prior_weight=2`` one glowing article gives ~67 instead of 100;
   with 50 documents the shrinkage is negligible.
4. Optional recency weighting: ``w_i = 0.5 ** (age_days / half_life_days)``.

The old notebook min-max scaled the three average class scores against each
other, which always put the largest one at 10 and the smallest at 1 no matter
what the data said. This module replaces that.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from statistics import fmean

from .models import Document

NEUTRAL_BAND = 0.15  # |polarity| below this counts as neutral at document level

BANDS = (
    (65.0, "Strong"),
    (55.0, "Positive"),
    (45.0, "Mixed"),
    (35.0, "At risk"),
    (0.0, "Poor"),
)


def label_for(polarity: float) -> str:
    if polarity >= NEUTRAL_BAND:
        return "positive"
    if polarity <= -NEUTRAL_BAND:
        return "negative"
    return "neutral"


def band_for(score: float) -> str:
    for threshold, name in BANDS:
        if score >= threshold:
            return name
    return BANDS[-1][1]


def score_documents(docs: list[Document]) -> None:
    for doc in docs:
        pols = [m.sentiment.polarity for m in doc.mentions if m.sentiment is not None]
        if pols:
            doc.polarity = fmean(pols)
            doc.label = label_for(doc.polarity)
        else:
            doc.polarity = None
            doc.label = None


def _weight(doc: Document, now: datetime, half_life_days: float | None) -> float:
    if not half_life_days or doc.published is None:
        return 1.0
    age = max(0.0, (now - doc.published).total_seconds() / 86400)
    return 0.5 ** (age / half_life_days)


def reputation_score(
    docs: list[Document],
    prior_weight: float = 2.0,
    half_life_days: float | None = None,
    now: datetime | None = None,
) -> float | None:
    scored = [d for d in docs if d.polarity is not None]
    if not scored:
        return None
    now = now or datetime.now(timezone.utc)
    weights = [_weight(d, now, half_life_days) for d in scored]
    num = sum(w * d.polarity for w, d in zip(weights, scored, strict=True))
    return 50.0 + 50.0 * num / (sum(weights) + prior_weight)


def confidence_for(n: int) -> str:
    if n >= 30:
        return "high"
    if n >= 10:
        return "medium"
    return "low"


def _week_start(dt: datetime) -> str:
    d = dt.astimezone(timezone.utc).date()
    return (d - timedelta(days=d.weekday())).isoformat()


def summarize(
    docs: list[Document],
    prior_weight: float = 2.0,
    half_life_days: float | None = None,
    now: datetime | None = None,
) -> dict:
    scored = [d for d in docs if d.polarity is not None]
    n = len(scored)
    score = reputation_score(scored, prior_weight, half_life_days, now)
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for d in scored:
        counts[d.label] += 1

    by_source: dict[str, list[float]] = defaultdict(list)
    for d in scored:
        by_source[d.source or "unknown"].append(d.polarity)
    sources = sorted(
        (
            {"source": s, "documents": len(p), "score": round(50 + 50 * fmean(p), 1)}
            for s, p in by_source.items()
        ),
        key=lambda r: (-r["documents"], r["source"]),
    )

    by_week: dict[str, list[float]] = defaultdict(list)
    for d in scored:
        if d.published:
            by_week[_week_start(d.published)].append(d.polarity)
    timeline = [
        {"week": w, "documents": len(p), "score": round(50 + 50 * fmean(p), 1)}
        for w, p in sorted(by_week.items())
    ]

    mentions = [(m, d) for d in scored for m in d.mentions if m.sentiment is not None]
    ranked = sorted(mentions, key=lambda md: md[0].sentiment.polarity)

    def _mention_row(md):
        m, d = md
        return {
            "text": m.text,
            "polarity": round(m.sentiment.polarity, 3),
            "source": d.source,
            "url": d.url,
            "title": d.title,
            "published": d.published.date().isoformat() if d.published else "",
        }

    def _top(items, keep) -> list[dict]:
        # syndicated stories repeat the same sentence on many sites; show each once
        seen, out = set(), []
        for md in items:
            key = md[0].text.strip().lower()
            if key in seen or not keep(md[0].sentiment.polarity):
                continue
            seen.add(key)
            out.append(_mention_row(md))
            if len(out) == 5:
                break
        return out

    ratings = [d.rating for d in scored if d.rating is not None]

    return {
        "reputation_score": round(score, 1) if score is not None else None,
        "band": band_for(score) if score is not None else None,
        "confidence": confidence_for(n),
        "documents_scored": n,
        "documents_total": len(docs),
        "documents_without_mentions": sum(1 for d in docs if d.polarity is None),
        "mentions_scored": len(mentions),
        "mean_polarity": round(fmean(d.polarity for d in scored), 4) if scored else None,
        "label_counts": counts,
        "label_share": {k: round(v / n, 4) if n else 0.0 for k, v in counts.items()},
        "average_rating": round(fmean(ratings), 2) if ratings else None,
        "by_source": sources,
        "timeline": timeline,
        "most_negative": _top(ranked, lambda p: p < 0),
        "most_positive": _top(reversed(ranked), lambda p: p > 0),
        "method": {
            "prior_weight": prior_weight,
            "half_life_days": half_life_days,
            "neutral_band": NEUTRAL_BAND,
        },
    }
