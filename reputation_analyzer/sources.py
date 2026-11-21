"""Collectors that find documents mentioning the entity.

Every collector returns a list of :class:`Document`. None of them scrape
Google's HTML (which breaks whenever Google changes its markup and is
frequently blocked); they use public or official APIs instead:

* ``gdelt``   - GDELT DOC 2.0 news API. Free, no key, ~3 months of global news.
* ``ddg``     - DuckDuckGo / metasearch via the ``ddgs`` package. Free, no key.
* ``serpapi`` - Google results through SerpAPI. Needs ``SERPAPI_API_KEY``.
* ``brave``   - Brave Search API. Needs ``BRAVE_API_KEY``.
* ``urls``    - a text file of URLs you already have.
* ``csv``     - a review export (Google reviews, Practo, Justdial, app stores...).
"""

from __future__ import annotations

import csv
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

from .http import DEFAULT_TIMEOUT, make_session
from .models import Document

log = logging.getLogger(__name__)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
SERPAPI_URL = "https://serpapi.com/search.json"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


class SourceError(RuntimeError):
    """A collector could not run (missing key, missing package, bad response)."""


def domain_of(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    formats = (
        "%Y%m%dT%H%M%SZ",  # GDELT seendate
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%b %d, %Y",
        "%d %b %Y",
    )
    candidate = value.replace("Z", "+00:00") if "T" in value and value.endswith("Z") else value
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        dt = None
        for fmt in formats:
            try:
                dt = datetime.strptime(value, fmt)
                break
            except ValueError:
                continue
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _phrase(query: str) -> str:
    q = query.strip()
    if " " in q and not (q.startswith('"') and q.endswith('"')):
        return f'"{q}"'
    return q


# --------------------------------------------------------------------------- news


def search_gdelt(
    query: str,
    max_results: int = 50,
    timespan: str = "3months",
    language: str | None = "english",
    session: requests.Session | None = None,
) -> list[Document]:
    """News articles from the GDELT DOC 2.0 API (no key needed, max 250)."""
    session = session or make_session()
    q = _phrase(query)
    if language and language.lower() != "any":
        q += f" sourcelang:{language.lower()}"
    params = {
        "query": q,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max(1, min(int(max_results), 250)),
        "sort": "DateDesc",
        "timespan": timespan,
    }
    resp = session.get(GDELT_URL, params=params, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    if not resp.text.strip():
        return []  # GDELT answers an empty body when nothing matches
    try:
        data = resp.json()
    except ValueError as exc:  # GDELT reports query errors as plain text
        raise SourceError(f"GDELT rejected the query: {resp.text.strip()[:200]}") from exc
    docs = []
    for art in data.get("articles", []) or []:
        url = art.get("url")
        if not url:
            continue
        docs.append(
            Document(
                url=url,
                title=(art.get("title") or "").strip(),
                source=art.get("domain") or domain_of(url),
                published=_parse_date(art.get("seendate")),
                origin="gdelt",
            )
        )
    return docs


def search_duckduckgo(query: str, max_results: int = 30, news: bool = True) -> list[Document]:
    """News (default) or web results through the ``ddgs`` metasearch package."""
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise SourceError("The 'ddg' source needs the ddgs package: pip install ddgs") from exc
    with DDGS() as client:
        rows = (
            client.news(query, max_results=max_results)
            if news
            else client.text(query, max_results=max_results)
        )
    docs = []
    for row in rows or []:
        url = row.get("url") or row.get("href")
        if not url:
            continue
        docs.append(
            Document(
                url=url,
                title=row.get("title", ""),
                snippet=row.get("body", ""),
                source=row.get("source") or domain_of(url),
                published=_parse_date(row.get("date")),
                origin="ddg",
            )
        )
    return docs


def search_serpapi(
    query: str,
    max_results: int = 30,
    api_key: str | None = None,
    session: requests.Session | None = None,
    gl: str = "in",
    hl: str = "en",
) -> list[Document]:
    """Google organic results through SerpAPI (needs SERPAPI_API_KEY)."""
    api_key = api_key or os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        raise SourceError("The 'serpapi' source needs SERPAPI_API_KEY in the environment")
    session = session or make_session()
    docs: list[Document] = []
    start = 0
    while len(docs) < max_results:
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": 10,
            "start": start,
            "gl": gl,
            "hl": hl,
        }
        resp = session.get(SERPAPI_URL, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            if docs:
                break
            raise SourceError(f"SerpAPI: {data['error']}")
        page = data.get("organic_results") or []
        if not page:
            break
        for row in page:
            url = row.get("link")
            if url:
                docs.append(
                    Document(
                        url=url,
                        title=row.get("title", ""),
                        snippet=row.get("snippet", ""),
                        source=domain_of(url),
                        published=_parse_date(row.get("date")),
                        origin="serpapi",
                    )
                )
        start += 10
        if start >= 100:
            break
    return docs[:max_results]


def search_brave(
    query: str,
    max_results: int = 30,
    api_key: str | None = None,
    session: requests.Session | None = None,
    country: str = "IN",
) -> list[Document]:
    """Web results from the Brave Search API (needs BRAVE_API_KEY)."""
    api_key = api_key or os.environ.get("BRAVE_API_KEY")
    if not api_key:
        raise SourceError("The 'brave' source needs BRAVE_API_KEY in the environment")
    session = session or make_session()
    headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}
    docs: list[Document] = []
    offset = 0
    while len(docs) < max_results and offset <= 9:
        params = {"q": query, "count": 20, "offset": offset, "country": country}
        resp = session.get(BRAVE_URL, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        results = (resp.json().get("web") or {}).get("results") or []
        if not results:
            break
        for row in results:
            url = row.get("url")
            if url:
                docs.append(
                    Document(
                        url=url,
                        title=row.get("title", ""),
                        snippet=row.get("description", ""),
                        source=domain_of(url),
                        published=_parse_date(row.get("page_age")),
                        origin="brave",
                    )
                )
        offset += 1
    return docs[:max_results]


# --------------------------------------------------------------------------- local files


def load_urls(path: str | Path) -> list[Document]:
    """One URL per line; blank lines and lines starting with # are ignored."""
    docs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        if not url.startswith(("http://", "https://")):
            log.warning("Skipping non-HTTP line in %s: %s", path, url)
            continue
        docs.append(Document(url=url, source=domain_of(url), origin="urls"))
    return docs


_TEXT_COLUMNS = ("text", "review", "review_text", "content", "comment", "body", "message")
_DATE_COLUMNS = ("date", "published", "time", "timestamp", "created_at", "review_date")
_SOURCE_COLUMNS = ("source", "platform", "site")
_RATING_COLUMNS = ("rating", "stars", "score")


def _pick(row: dict[str, str], names: tuple[str, ...]) -> str:
    lowered = {k.strip().lower(): v for k, v in row.items() if k}
    for name in names:
        value = lowered.get(name)
        if value not in (None, ""):
            return value.strip()
    return ""


def load_reviews_csv(path: str | Path, default_source: str = "reviews") -> list[Document]:
    """Reviews or comments you exported yourself.

    Needs a text column (``text``, ``review``, ``content``, ``comment``...).
    Optional columns: ``date``, ``source``, ``rating``, ``url``, ``title``.
    """
    docs = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        headers = {h.strip().lower() for h in (reader.fieldnames or []) if h}
        if not headers.intersection(_TEXT_COLUMNS):
            raise SourceError(f"{path} has no text column; expected one of: {', '.join(_TEXT_COLUMNS)}")
        for i, row in enumerate(reader):
            text = _pick(row, _TEXT_COLUMNS)
            if not text:
                continue
            rating_raw = _pick(row, _RATING_COLUMNS)
            try:
                rating = float(rating_raw) if rating_raw else None
            except ValueError:
                rating = None
            url = _pick(row, ("url", "link"))
            docs.append(
                Document(
                    url=url or f"{Path(path).name}#row{i + 1}",
                    title=_pick(row, ("title",)) or text[:80],
                    source=_pick(row, _SOURCE_COLUMNS) or default_source,
                    published=_parse_date(_pick(row, _DATE_COLUMNS)),
                    text=text,
                    text_source="provided",
                    origin="csv",
                    about_entity=True,
                    rating=rating,
                )
            )
    return docs


SEARCH_SOURCES: dict[str, Callable[..., list[Document]]] = {
    "gdelt": search_gdelt,
    "ddg": search_duckduckgo,
    "serpapi": search_serpapi,
    "brave": search_brave,
}


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    return f"{parsed.netloc.lower().removeprefix('www.')}{path}?{parsed.query}".rstrip("?")


def dedupe(docs: list[Document]) -> list[Document]:
    seen: set[str] = set()
    out = []
    for doc in docs:
        key = normalize_url(doc.url) if doc.url.startswith("http") else doc.doc_id
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out
