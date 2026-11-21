"""Text cleaning and finding the passages that mention the entity."""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HANDLE_RE = re.compile(r"(?<!\w)@\w+")
_WS_RE = re.compile(r"[ \t ]+")
# sentence ends: . ! ? and the Devanagari danda, followed by whitespace
_SENT_RE = re.compile(r"(?<=[.!?।])\s+(?=[\"'“(\[]?[A-Z0-9ऀ-ॿ])")

MAX_MENTION_CHARS = 700


def clean_text(text: str) -> str:
    """Normalise whitespace and replace URLs / @handles the way the
    CardiffNLP Twitter models were trained (``http`` and ``@user``)."""
    text = _URL_RE.sub("http", text)
    text = _HANDLE_RE.sub("@user", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for para in re.split(r"\n+", text):
        para = para.strip()
        if not para:
            continue
        sentences.extend(s.strip() for s in _SENT_RE.split(para) if s.strip())
    return sentences


def build_pattern(entity: str, aliases: list[str] | tuple[str, ...] = ()) -> re.Pattern[str]:
    names = sorted({n.strip() for n in (entity, *aliases) if n and n.strip()}, key=len, reverse=True)
    if not names:
        raise ValueError("entity name is empty")
    parts = []
    for name in names:
        # allow flexible whitespace/hyphens inside multi-word names
        escaped = r"[\s\-]+".join(re.escape(w) for w in name.split())
        parts.append(escaped)
    return re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)", re.IGNORECASE)


def _truncate(text: str, limit: int = MAX_MENTION_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit * 0.6 else cut) + " …"


def extract_mentions(
    text: str,
    pattern: re.Pattern[str],
    window: int = 1,
    max_mentions: int = 25,
) -> list[str]:
    """Passages around each sentence that names the entity.

    ``window`` is how many neighbouring sentences to keep on each side (so a
    sentence like "It was a nightmare." right after the entity's name still
    counts). Overlapping windows are merged.
    """
    sentences = split_sentences(text)
    hits = [i for i, s in enumerate(sentences) if pattern.search(s)]
    if not hits:
        return []
    spans: list[list[int]] = []
    for i in hits:
        lo, hi = max(0, i - window), min(len(sentences) - 1, i + window)
        if spans and lo <= spans[-1][1] + 1:
            spans[-1][1] = max(spans[-1][1], hi)
        else:
            spans.append([lo, hi])
    passages = [clean_text(" ".join(sentences[lo : hi + 1])) for lo, hi in spans]
    return [_truncate(p) for p in passages if p][:max_mentions]


def chunk_text(text: str, max_chars: int = MAX_MENTION_CHARS, max_chunks: int = 10) -> list[str]:
    """Split text that is entirely about the entity (e.g. a review) into
    model-sized passages on sentence boundaries."""
    chunks: list[str] = []
    current = ""
    for sent in split_sentences(clean_text(text)):
        if current and len(current) + 1 + len(sent) > max_chars:
            chunks.append(current)
            current = sent
        else:
            current = f"{current} {sent}".strip()
        if len(chunks) >= max_chunks:
            break
    if current and len(chunks) < max_chunks:
        chunks.append(current)
    return [_truncate(c, max_chars) for c in chunks]
