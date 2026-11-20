import pytest

from reputation_analyzer import sources
from reputation_analyzer.models import Document
from reputation_analyzer.sources import (
    SourceError,
    dedupe,
    load_reviews_csv,
    load_urls,
    search_brave,
    search_gdelt,
    search_serpapi,
)


def test_gdelt_parses_articles_and_quotes_phrases(fake_session_cls, fake_response_cls):
    session = fake_session_cls(
        [
            fake_response_cls(
                json={
                    "articles": [
                        {
                            "url": "https://www.thehindu.com/a",
                            "title": " Hospital expands ",
                            "domain": "thehindu.com",
                            "seendate": "20261101T101500Z",
                        },
                        {"title": "no url"},
                    ]
                }
            )
        ]
    )
    docs = search_gdelt("Ruby Hall Clinic", max_results=500, session=session)
    assert len(docs) == 1
    d = docs[0]
    assert d.title == "Hospital expands" and d.source == "thehindu.com" and d.origin == "gdelt"
    assert d.published.year == 2026 and d.published.month == 11
    params = session.calls[0]["params"]
    assert params["query"] == '"Ruby Hall Clinic" sourcelang:english'
    assert params["maxrecords"] == 250


def test_gdelt_plain_text_error_and_empty_body(fake_session_cls, fake_response_cls):
    s = fake_session_cls([fake_response_cls("The specified phrase is too short.")])
    with pytest.raises(SourceError, match="too short"):
        search_gdelt("ab", session=s)
    s = fake_session_cls([fake_response_cls("  ")])
    assert search_gdelt("acme", session=s, language="any") == []
    assert "sourcelang" not in s.calls[0]["params"]["query"]


def test_serpapi_needs_key(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    with pytest.raises(SourceError):
        search_serpapi("acme")


def test_serpapi_paginates(fake_session_cls, fake_response_cls):
    def page(n):
        results = [
            {"link": f"https://site{n}-{i}.com/x", "title": f"t{i}", "snippet": "s"} for i in range(10)
        ]
        return fake_response_cls(json={"organic_results": results})

    s = fake_session_cls([page(1), page(2), page(3)])
    docs = search_serpapi("acme", max_results=25, api_key="k", session=s)
    assert len(docs) == 25
    assert [c["params"]["start"] for c in s.calls] == [0, 10, 20]
    # stops early when a page comes back empty
    s = fake_session_cls([page(1), fake_response_cls(json={"organic_results": []})])
    assert len(search_serpapi("acme", max_results=25, api_key="k", session=s)) == 10


def test_brave(fake_session_cls, fake_response_cls):
    s = fake_session_cls(
        [
            fake_response_cls(
                json={"web": {"results": [{"url": "https://b.com/1", "title": "B", "description": "d"}]}}
            ),
            fake_response_cls(json={"web": {"results": []}}),
        ]
    )
    docs = search_brave("acme", api_key="k", session=s)
    assert docs[0].snippet == "d" and docs[0].origin == "brave"
    assert s.calls[0]["headers"]["X-Subscription-Token"] == "k"


def test_ddg_missing_package_gives_clear_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "ddgs":
            raise ImportError("no ddgs")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SourceError, match="pip install ddgs"):
        sources.search_duckduckgo("acme")


def test_load_urls(tmp_path):
    f = tmp_path / "u.txt"
    f.write_text("# comment\nhttps://www.a.com/x\n\nnot-a-url\nhttp://b.com\n")
    docs = load_urls(f)
    assert [d.source for d in docs] == ["a.com", "b.com"]


def test_load_reviews_csv_flexible_columns(tmp_path):
    f = tmp_path / "r.csv"
    f.write_text(
        "Review,Stars,Date,Platform\nGreat doctors,5,2026-10-01,Google\n,3,,\n"
        "Rude staff,1,05/10/2026,Practo\n",
        encoding="utf-8-sig",
    )
    docs = load_reviews_csv(f)
    assert len(docs) == 2
    assert docs[0].text == "Great doctors" and docs[0].rating == 5.0 and docs[0].source == "Google"
    assert docs[0].about_entity and docs[0].text_source == "provided"
    assert docs[1].published.day == 5 and docs[1].published.month == 10


def test_load_reviews_csv_without_text_column(tmp_path):
    f = tmp_path / "r.csv"
    f.write_text("name,stars\nx,5\n")
    with pytest.raises(SourceError, match="no text column"):
        load_reviews_csv(f)


def test_dedupe_normalises_urls():
    docs = [
        Document(url="https://www.a.com/x/"),
        Document(url="https://a.com/x"),
        Document(url="https://a.com/x?id=2"),
    ]
    assert len(dedupe(docs)) == 2
