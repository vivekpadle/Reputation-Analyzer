import json
import sys
import types
from datetime import datetime, timezone

import pytest

from reputation_analyzer import cli, pipeline
from reputation_analyzer.models import Document
from reputation_analyzer.pipeline import AnalysisConfig, analyze
from reputation_analyzer.report import render_html, write_outputs
from reputation_analyzer.sources import SourceError


def make_docs():
    def d(i, text, source, day):
        return Document(
            url=f"https://{source}/{i}",
            title=f"Story {i}",
            source=source,
            published=datetime(2026, 10, day, tzinfo=timezone.utc),
            text=text,
            text_source="page",
            origin="gdelt",
        )

    return [
        d(1, "City news. Acme Hospital staff were caring and helpful. Weather was mild.", "a.com", 1),
        d(2, "Acme Hospital was praised for excellent care. Doctors were great.", "a.com", 8),
        d(3, "Patients said Acme Hospital overcharged them. Staff were rude.", "b.com", 15),
        d(4, "An article about something else entirely with no names.", "c.com", 22),
        d(5, "Acme Hospital announced a new building.", "b.com", 29),
    ]


def test_analyze_end_to_end(fake_engine):
    cfg = AnalysisConfig(entity="Acme Hospital", fetch_pages=False)
    stages = []
    result = analyze(cfg, engine=fake_engine, documents=make_docs(), progress=lambda s, f: stages.append(s))
    s = result.summary
    assert s["documents_total"] == 5
    assert s["documents_scored"] == 4
    assert s["label_counts"] == {"positive": 2, "neutral": 1, "negative": 1}
    assert 50 < s["reputation_score"] < 70
    assert result.engine == "fake-lexicon"
    assert fake_engine.calls == 1  # all mentions scored in one batch call
    assert stages[0] == "Collecting sources" and stages[-1] == "Done"
    assert len(s["timeline"]) == 4


def test_reviews_are_scored_without_name_match(fake_engine, tmp_path):
    f = tmp_path / "reviews.csv"
    f.write_text("text,rating\nExcellent doctors and clean rooms,5\nWorst delay ever,1\nOkay visit,3\n")
    cfg = AnalysisConfig(entity="Acme Hospital", sources=[], reviews_csv=f)
    result = analyze(cfg, engine=fake_engine)
    assert result.summary["documents_scored"] == 3
    assert result.summary["average_rating"] == 3.0


def test_no_documents_raises(fake_engine, monkeypatch):
    monkeypatch.setattr(pipeline, "collect_documents", lambda cfg, errors: [])
    with pytest.raises(SourceError):
        analyze(AnalysisConfig(entity="Nobody"), engine=fake_engine)


def test_failed_source_is_reported_not_fatal(fake_engine, monkeypatch, tmp_path):
    def boom(*a, **k):
        raise SourceError("rate limited")

    monkeypatch.setitem(pipeline.SEARCH_SOURCES, "gdelt", boom)
    f = tmp_path / "r.csv"
    f.write_text("text\nGreat place\n")
    result = analyze(AnalysisConfig(entity="Acme", reviews_csv=f), engine=fake_engine)
    assert result.summary["source_errors"] == ["gdelt: rate limited"]
    assert result.summary["documents_scored"] == 1


def test_write_outputs_and_html(fake_engine, tmp_path):
    result = analyze(
        AnalysisConfig(entity="Acme Hospital", fetch_pages=False), engine=fake_engine, documents=make_docs()
    )
    paths = write_outputs(result, tmp_path)
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    summary = json.loads(paths["summary"].read_text())
    assert summary["entity"] == "Acme Hospital"
    html = paths["report"].read_text()
    assert f">{result.summary['reputation_score']:.0f}<small>" in html
    assert "Score over time" in html and "Score by source" in html
    assert "<script" in html and "http" in html  # tooltips + links present
    # escaping: titles with HTML must not inject markup
    result.documents[0].title = "<img src=x onerror=alert(1)>"
    assert "<img src=x" not in render_html(result)


def test_themes_with_fake_anthropic(fake_engine, monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kw):
            captured.update(kw)
            block = types.SimpleNamespace(type="text", text="## What people praise\n- **Caring** staff")
            return types.SimpleNamespace(content=[block])

    class FakeClient:
        def __init__(self):
            self.messages = FakeMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=FakeClient))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    cfg = AnalysisConfig(entity="Acme Hospital", fetch_pages=False, themes=True)
    result = analyze(cfg, engine=fake_engine, documents=make_docs())
    assert "Caring" in result.themes
    assert captured["model"] == "claude-haiku-5-5"
    assert "<passages>" in captured["messages"][0]["content"]
    assert "not instructions" in captured["system"]
    assert "<strong>Caring</strong>" in render_html(result)


def test_themes_failure_does_not_break_run(fake_engine, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=object))
    cfg = AnalysisConfig(entity="Acme Hospital", fetch_pages=False, themes=True)
    result = analyze(cfg, engine=fake_engine, documents=make_docs())
    assert result.themes is None
    assert "ANTHROPIC_API_KEY" in result.summary["themes_error"]


def test_cli_with_reviews(fake_engine, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(pipeline, "make_engine", lambda *a, **k: fake_engine)
    f = tmp_path / "r.csv"
    f.write_text("review\nExcellent and caring\nGreat doctors\nRude reception\n")
    code = cli.main(["Acme", "--no-search", "--reviews", str(f), "-o", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert code == 0
    assert "Acme:" in out and "/100" in out and "report.html" in out
    assert list((tmp_path / "out").glob("acme-*/report.html"))


def test_cli_no_search_requires_input(capsys):
    assert cli.main(["Acme", "--no-search"]) == 2
