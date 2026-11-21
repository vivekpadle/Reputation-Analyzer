from __future__ import annotations

import json as jsonlib

import pytest

from reputation_analyzer.sentiment import scores_from_probs

POSITIVE_WORDS = {"excellent", "great", "caring", "best", "clean", "helpful", "praised", "good"}
NEGATIVE_WORDS = {"terrible", "rude", "dirty", "worst", "overcharged", "negligence", "bad", "delay"}


class FakeEngine:
    """Deterministic keyword scorer so tests never download a model."""

    name = "fake-lexicon"

    def __init__(self):
        self.calls = 0

    def score(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            words = {w.strip(".,!?").lower() for w in t.split()}
            pos = len(words & POSITIVE_WORDS)
            neg = len(words & NEGATIVE_WORDS)
            if pos > neg:
                out.append(scores_from_probs(0.05, 0.15, 0.80))
            elif neg > pos:
                out.append(scores_from_probs(0.80, 0.15, 0.05))
            else:
                out.append(scores_from_probs(0.10, 0.80, 0.10))
        return out


@pytest.fixture
def fake_engine():
    return FakeEngine()


class FakeResponse:
    def __init__(self, body="", status=200, headers=None, json=None, encoding="utf-8"):
        if json is not None:
            body = jsonlib.dumps(json)
        self.text = body
        self.content = body.encode("utf-8")
        self.status_code = status
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.encoding = encoding
        self.apparent_encoding = "utf-8"

    def json(self):
        return jsonlib.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code} error")

    def iter_content(self, size):
        for i in range(0, len(self.content), size):
            yield self.content[i : i + size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Returns queued responses (or a function of the URL) and records calls."""

    def __init__(self, responses=None, by_url=None):
        self.responses = list(responses or [])
        self.by_url = by_url or {}
        self.calls = []

    def get(self, url, params=None, **kwargs):
        self.calls.append({"url": url, "params": params, **kwargs})
        if url in self.by_url:
            r = self.by_url[url]
            if isinstance(r, Exception):
                raise r
            return r
        return self.responses.pop(0)


@pytest.fixture
def fake_session_cls():
    return FakeSession


@pytest.fixture
def fake_response_cls():
    return FakeResponse
