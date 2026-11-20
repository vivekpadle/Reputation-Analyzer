"""End-to-end analysis: collect -> fetch -> find mentions -> score -> summarise."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .fetch import fetch_all, use_snippets_only
from .http import make_session
from .models import AnalysisResult, Document, Mention
from .scoring import score_documents, summarize
from .sentiment import SentimentEngine, make_engine
from .sources import SEARCH_SOURCES, SourceError, dedupe, load_reviews_csv, load_urls
from .text import build_pattern, chunk_text, clean_text, extract_mentions

log = logging.getLogger(__name__)

Progress = Callable[[str, float], None]


@dataclass
class AnalysisConfig:
    entity: str
    aliases: list[str] = field(default_factory=list)
    query: str | None = None  # search query; defaults to the entity name
    sources: list[str] = field(default_factory=lambda: ["gdelt"])
    max_results: int = 50
    timespan: str = "3months"  # GDELT window, e.g. 1w, 1month, 3months
    language: str = "english"  # GDELT source language filter ("any" for all)
    urls_file: str | Path | None = None
    reviews_csv: str | Path | None = None
    fetch_pages: bool = True
    workers: int = 8
    engine: str = "transformer"  # transformer | multilingual | vader
    model: str | None = None
    batch_size: int = 32
    window: int = 1  # neighbouring sentences kept around each mention
    prior_weight: float = 2.0
    half_life_days: float | None = None
    themes: bool = False  # summarise themes with Claude (needs ANTHROPIC_API_KEY)
    themes_model: str | None = None


def collect_documents(cfg: AnalysisConfig, errors: list[str] | None = None) -> list[Document]:
    errors = errors if errors is not None else []
    query = cfg.query or cfg.entity
    session = make_session()
    docs: list[Document] = []
    for name in cfg.sources:
        fn = SEARCH_SOURCES.get(name)
        if fn is None:
            raise ValueError(f"Unknown source {name!r}; choose from {', '.join(SEARCH_SOURCES)}")
        try:
            if name == "gdelt":
                found = fn(
                    query, cfg.max_results, timespan=cfg.timespan, language=cfg.language, session=session
                )
            elif name == "ddg":
                found = fn(query, cfg.max_results)
            else:
                found = fn(query, cfg.max_results, session=session)
            log.info("%s: %d results", name, len(found))
            docs.extend(found)
        except (SourceError, OSError, ValueError) as exc:
            msg = f"{name}: {exc}"
            log.warning("Source failed - %s", msg)
            errors.append(msg)
    if cfg.urls_file:
        docs.extend(load_urls(cfg.urls_file))
    if cfg.reviews_csv:
        docs.extend(load_reviews_csv(cfg.reviews_csv))
    return dedupe(docs)


def find_mentions(docs: list[Document], entity: str, aliases: list[str], window: int = 1) -> None:
    pattern = build_pattern(entity, aliases)
    for doc in docs:
        if not doc.text:
            doc.mentions = []
            continue
        if doc.about_entity:
            passages = chunk_text(doc.text)
        else:
            passages = extract_mentions(doc.text, pattern, window=window)
            if not passages and doc.text_source == "snippet" and pattern.search(doc.title or ""):
                passages = [clean_text(doc.text)]
        doc.mentions = [Mention(text=p) for p in passages]


def score_mentions(docs: list[Document], engine: SentimentEngine) -> None:
    mentions = [m for d in docs for m in d.mentions]
    if not mentions:
        return
    scores = engine.score([m.text for m in mentions])
    for mention, s in zip(mentions, scores, strict=True):
        mention.sentiment = s


def analyze(
    cfg: AnalysisConfig,
    engine: SentimentEngine | None = None,
    documents: list[Document] | None = None,
    progress: Progress | None = None,
) -> AnalysisResult:
    """Run the full pipeline.

    Pass ``documents`` to skip collection (e.g. data you already loaded), and
    ``engine`` to reuse a loaded model between runs.
    """

    def report(stage: str, frac: float) -> None:
        if progress:
            progress(stage, frac)

    errors: list[str] = []
    # load the model first so a missing dependency fails fast, before any downloads
    engine = engine or make_engine(cfg.engine, cfg.model, cfg.batch_size)
    report("Collecting sources", 0.0)
    docs = documents if documents is not None else collect_documents(cfg, errors)
    if not docs:
        raise SourceError(
            "No documents found. "
            + (
                "Errors: " + "; ".join(errors)
                if errors
                else "Try another source, a longer --timespan or an alias."
            )
        )

    report("Reading pages", 0.15)
    if cfg.fetch_pages:
        fetch_all(
            docs,
            workers=cfg.workers,
            progress=lambda d, t: report("Reading pages", 0.15 + 0.45 * d / max(t, 1)),
        )
    else:
        use_snippets_only(docs)

    report("Finding mentions", 0.6)
    find_mentions(docs, cfg.entity, cfg.aliases, cfg.window)

    report("Scoring sentiment", 0.65)
    score_mentions(docs, engine)
    score_documents(docs)

    report("Summarising", 0.9)
    summary = summarize(docs, cfg.prior_weight, cfg.half_life_days)
    summary["query"] = cfg.query or cfg.entity
    summary["sources_used"] = sorted({d.origin for d in docs})
    summary["source_errors"] = errors
    summary["pages_read"] = sum(1 for d in docs if d.text_source == "page")
    summary["snippet_only"] = sum(1 for d in docs if d.text_source == "snippet")

    result = AnalysisResult(
        entity=cfg.entity,
        aliases=list(cfg.aliases),
        generated_at=datetime.now(timezone.utc),
        engine=getattr(engine, "name", type(engine).__name__),
        documents=docs,
        summary=summary,
    )

    if cfg.themes:
        report("Summarising themes with Claude", 0.95)
        from .llm import summarize_themes

        try:
            result.themes = summarize_themes(result, model=cfg.themes_model)
        except Exception as exc:  # optional feature: never fail the whole run
            log.warning("Theme summary skipped: %s", exc)
            summary["themes_error"] = str(exc)

    report("Done", 1.0)
    return result
