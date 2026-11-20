import pytest

from reputation_analyzer.sentiment import make_engine, normalize_prediction, scores_from_probs


def test_normalize_named_labels():
    s = normalize_prediction(
        [
            {"label": "positive", "score": 0.7},
            {"label": "neutral", "score": 0.2},
            {"label": "negative", "score": 0.1},
        ]
    )
    assert s.label == "positive"
    assert s.polarity == pytest.approx(0.6)


def test_normalize_legacy_label_ids():
    # cardiffnlp/twitter-roberta-base-sentiment (the original model) returns LABEL_0..2
    s = normalize_prediction(
        [
            {"label": "LABEL_0", "score": 0.8},
            {"label": "LABEL_1", "score": 0.15},
            {"label": "LABEL_2", "score": 0.05},
        ]
    )
    assert s.label == "negative"
    assert s.polarity == pytest.approx(-0.75)


def test_normalize_star_ratings():
    pred = [
        {"label": f"{i} star" + ("s" if i > 1 else ""), "score": p}
        for i, p in zip(range(1, 6), (0.05, 0.05, 0.1, 0.3, 0.5), strict=True)
    ]
    s = normalize_prediction(pred)
    assert s.label == "positive"
    assert s.positive == pytest.approx(0.8)
    assert s.negative == pytest.approx(0.1)


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        normalize_prediction([{"label": "joy", "score": 1.0}])


def test_scores_from_probs_normalises():
    s = scores_from_probs(2, 1, 1)
    assert s.negative == pytest.approx(0.5)
    assert s.label == "negative"


def test_unknown_engine():
    with pytest.raises(ValueError):
        make_engine("nope")
