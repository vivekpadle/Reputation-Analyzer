"""Plain data containers shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class SentimentScore:
    """Sentiment of one piece of text.

    ``negative``/``neutral``/``positive`` are class probabilities (they sum to 1
    for transformer models). ``polarity`` is a single number in [-1, 1]:
    positive minus negative for transformer models, VADER's compound score for
    VADER.
    """

    negative: float
    neutral: float
    positive: float
    polarity: float
    label: str  # "negative" | "neutral" | "positive"


@dataclass
class Mention:
    """A passage of a document that talks about the entity."""

    text: str
    sentiment: SentimentScore | None = None


@dataclass
class Document:
    """One source item: a news article, web page or review."""

    url: str = ""
    title: str = ""
    source: str = ""  # domain or platform name, e.g. "thehindu.com", "Google reviews"
    published: datetime | None = None
    snippet: str = ""  # search-result summary, used when the page can't be fetched
    text: str = ""  # full extracted text
    origin: str = ""  # which collector produced it: gdelt, ddg, serpapi, brave, urls, csv
    about_entity: bool = False  # True for reviews: every sentence is about the entity
    rating: float | None = None  # star rating from a review export, if any
    fetch_error: str | None = None
    text_source: str = ""  # "page", "snippet", "provided"
    mentions: list[Mention] = field(default_factory=list)
    polarity: float | None = None  # document-level score in [-1, 1]
    label: str | None = None

    @property
    def doc_id(self) -> str:
        return self.url or f"{self.origin}:{self.title[:60]}"

    def to_row(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "published": self.published.isoformat() if self.published else "",
            "origin": self.origin,
            "text_source": self.text_source,
            "rating": self.rating if self.rating is not None else "",
            "mentions": len(self.mentions),
            "polarity": round(self.polarity, 4) if self.polarity is not None else "",
            "label": self.label or "",
            "fetch_error": self.fetch_error or "",
        }


@dataclass
class AnalysisResult:
    entity: str
    aliases: list[str]
    generated_at: datetime
    engine: str
    documents: list[Document]
    summary: dict[str, Any]
    themes: str | None = None

    @property
    def scored_documents(self) -> list[Document]:
        return [d for d in self.documents if d.polarity is not None]

    def mention_rows(self) -> list[dict[str, Any]]:
        rows = []
        for doc in self.documents:
            for m in doc.mentions:
                if m.sentiment is None:
                    continue
                rows.append(
                    {
                        "url": doc.url,
                        "source": doc.source,
                        "published": doc.published.isoformat() if doc.published else "",
                        "text": m.text,
                        **{
                            k: round(v, 4) if isinstance(v, float) else v
                            for k, v in asdict(m.sentiment).items()
                        },
                    }
                )
        return rows
