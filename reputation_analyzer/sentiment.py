"""Sentiment engines.

``TransformerEngine`` runs any Hugging Face text-classification model that
predicts negative/neutral/positive (or 1-5 stars). The default is
``cardiffnlp/twitter-roberta-base-sentiment-latest``, the updated version of
the model the original notebook used. For Hindi / Hinglish / other languages
use ``cardiffnlp/twitter-xlm-roberta-base-sentiment``.

``VaderEngine`` is a tiny rule-based fallback that needs no model download.
"""

from __future__ import annotations

import re
from typing import Protocol

from .models import SentimentScore

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MULTILINGUAL_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

_LABEL_ALIASES = {
    "negative": "negative",
    "neg": "negative",
    "label_0": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "label_1": "neutral",
    "positive": "positive",
    "pos": "positive",
    "label_2": "positive",
}
_STAR_RE = re.compile(r"^([1-5])\s*stars?$")


class SentimentEngine(Protocol):
    name: str

    def score(self, texts: list[str]) -> list[SentimentScore]: ...


def scores_from_probs(neg: float, neu: float, pos: float) -> SentimentScore:
    total = neg + neu + pos
    if total > 0:
        neg, neu, pos = neg / total, neu / total, pos / total
    label = max((("negative", neg), ("neutral", neu), ("positive", pos)), key=lambda kv: kv[1])[0]
    return SentimentScore(negative=neg, neutral=neu, positive=pos, polarity=pos - neg, label=label)


def normalize_prediction(pred: list[dict]) -> SentimentScore:
    """Turn a pipeline's ``[{'label': ..., 'score': ...}, ...]`` into a SentimentScore.

    Handles 3-class models (negative/neutral/positive, LABEL_0..2) and
    5-star review models (1 star .. 5 stars: 1-2 negative, 3 neutral, 4-5 positive).
    """
    probs = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    for item in pred:
        raw = str(item["label"]).strip().lower()
        star = _STAR_RE.match(raw)
        if star:
            n = int(star.group(1))
            key = "negative" if n <= 2 else "neutral" if n == 3 else "positive"
        else:
            key = _LABEL_ALIASES.get(raw)
        if key is None:
            raise ValueError(f"Unrecognised sentiment label {item['label']!r}")
        probs[key] += float(item["score"])
    return scores_from_probs(probs["negative"], probs["neutral"], probs["positive"])


def _pick_device():
    import torch

    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return -1


class TransformerEngine:
    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32, device=None):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "The transformer engine needs transformers and torch: "
                "pip install 'reputation-analyzer[ml]'  (or use --engine vader)"
            ) from exc
        self.name = model_name
        self.batch_size = batch_size
        self._pipe = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=_pick_device() if device is None else device,
        )

    def score(self, texts: list[str]) -> list[SentimentScore]:
        if not texts:
            return []
        outputs = self._pipe(texts, batch_size=self.batch_size, top_k=None, truncation=True, max_length=512)
        return [normalize_prediction(o if isinstance(o, list) else [o]) for o in outputs]


class VaderEngine:
    """Rule-based English sentiment. Fast and offline, less accurate than a transformer."""

    name = "vader"

    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as exc:
            raise RuntimeError("The VADER engine needs: pip install vaderSentiment") from exc
        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, texts: list[str]) -> list[SentimentScore]:
        out = []
        for text in texts:
            s = self._analyzer.polarity_scores(text)
            compound = s["compound"]
            # VADER's standard thresholds
            label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
            out.append(
                SentimentScore(
                    negative=s["neg"],
                    neutral=s["neu"],
                    positive=s["pos"],
                    polarity=compound,
                    label=label,
                )
            )
        return out


def make_engine(kind: str = "transformer", model: str | None = None, batch_size: int = 32):
    kind = kind.lower()
    if kind in ("transformer", "roberta", "hf"):
        return TransformerEngine(model or DEFAULT_MODEL, batch_size=batch_size)
    if kind == "multilingual":
        return TransformerEngine(model or MULTILINGUAL_MODEL, batch_size=batch_size)
    if kind == "vader":
        return VaderEngine()
    raise ValueError(f"Unknown engine {kind!r}; use transformer, multilingual or vader")
