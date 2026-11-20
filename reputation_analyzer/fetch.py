"""Download pages and pull out the readable article text."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

from .http import DEFAULT_TIMEOUT, make_session
from .models import Document

log = logging.getLogger(__name__)

MAX_BYTES = 3_000_000
SKIP_EXTENSIONS = (
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".mp4",
    ".mp3",
    ".zip",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
)
_NOISE_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "svg",
    "iframe",
    "button",
)


def extract_text(html: str, url: str = "") -> str:
    """Main text of an HTML page.

    Uses trafilatura when installed (much better at dropping menus, cookie
    banners and related-article lists); otherwise a BeautifulSoup fallback.
    """
    try:
        import trafilatura
    except ImportError:
        trafilatura = None
    if trafilatura is not None:
        text = trafilatura.extract(
            html,
            url=url or None,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        if text and len(text) > 200:
            return text.strip()

    soup = BeautifulSoup(html, "lxml") if _has_lxml() else BeautifulSoup(html, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    blocks = []
    for el in root.find_all(["h1", "h2", "h3", "p", "li", "blockquote"]):
        t = " ".join(el.get_text(" ", strip=True).split())
        # list items and short fragments are mostly menus; keep real sentences
        if el.name == "li" and len(t) < 60:
            continue
        if len(t) >= 25 or el.name in ("h1", "h2", "h3"):
            blocks.append(t)
    return "\n".join(_dedupe_lines(blocks)).strip()


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen = set()
    out = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def _has_lxml() -> bool:
    try:
        import lxml  # noqa: F401
    except ImportError:
        return False
    return True


def fetch_html(url: str, session: requests.Session) -> str:
    if url.lower().split("?")[0].endswith(SKIP_EXTENSIONS):
        raise ValueError("not an HTML page")
    with session.get(url, timeout=DEFAULT_TIMEOUT, stream=True) as resp:
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if ctype and "html" not in ctype and "xml" not in ctype:
            raise ValueError(f"unsupported content type {ctype.split(';')[0]}")
        chunks, size = [], 0
        for chunk in resp.iter_content(64 * 1024):
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_BYTES:
                break
        raw = b"".join(chunks)
        encoding = resp.encoding if resp.encoding and resp.encoding.lower() != "iso-8859-1" else None
        if encoding is None:
            encoding = resp.apparent_encoding or "utf-8"
    return raw.decode(encoding, errors="replace")


def fetch_document(doc: Document, session: requests.Session) -> Document:
    """Fill ``doc.text``; fall back to title + snippet if the page can't be read."""
    if doc.text:
        doc.text_source = doc.text_source or "provided"
        return doc
    try:
        html = fetch_html(doc.url, session)
        text = extract_text(html, doc.url)
        if not doc.title:
            soup = BeautifulSoup(html[:200_000], "html.parser")
            if soup.title and soup.title.string:
                doc.title = soup.title.string.strip()
        if len(text) < 200:
            raise ValueError("page has too little readable text (paywall or JavaScript-only?)")
        doc.text, doc.text_source = text, "page"
    except Exception as exc:  # network errors, HTTP errors, parsing errors
        doc.fetch_error = f"{type(exc).__name__}: {exc}"[:300]
        fallback = " ".join(p for p in (doc.title, doc.snippet) if p).strip()
        if fallback:
            doc.text, doc.text_source = fallback, "snippet"
        log.debug("fetch failed for %s: %s", doc.url, doc.fetch_error)
    return doc


def fetch_all(
    docs: list[Document],
    workers: int = 8,
    session: requests.Session | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[Document]:
    session = session or make_session(pool_size=max(workers, 4))
    total = len(docs)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(fetch_document, d, session) for d in docs]
        for _ in as_completed(futures):
            done += 1
            if progress:
                progress(done, total)
    return docs


def use_snippets_only(docs: list[Document]) -> list[Document]:
    """Skip downloading; analyse the title and search snippet only."""
    for doc in docs:
        if not doc.text:
            doc.text = " ".join(p for p in (doc.title, doc.snippet) if p).strip()
            doc.text_source = "snippet"
        elif not doc.text_source:
            doc.text_source = "provided"
    return docs
