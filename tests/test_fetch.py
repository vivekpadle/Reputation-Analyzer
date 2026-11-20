import requests

from reputation_analyzer.fetch import extract_text, fetch_all, fetch_document, use_snippets_only
from reputation_analyzer.models import Document

ARTICLE = """
<html><head><title>Clinic news</title><script>var x = 'ignore me';</script></head>
<body>
<nav><ul><li>Home</li><li>About</li></ul></nav>
<article>
<h1>Acme Hospital opens new wing</h1>
<p>Acme Hospital opened a new cardiac wing on Monday, adding forty beds to the city's capacity.</p>
<p>Patients praised the caring staff and said the new rooms were clean and well equipped.</p>
<p>Some visitors complained about parking, which remains limited during peak hours.</p>
</article>
<footer><p>Copyright 2026 Example News. All rights reserved worldwide.</p></footer>
</body></html>
"""


def test_extract_text_keeps_article_and_drops_chrome():
    text = extract_text(ARTICLE)
    assert "new cardiac wing" in text
    assert "parking" in text
    assert "ignore me" not in text
    assert "Copyright" not in text
    assert "Home" not in text


def test_fetch_document_reads_page(fake_session_cls, fake_response_cls):
    s = fake_session_cls(by_url={"https://n.com/a": fake_response_cls(ARTICLE)})
    d = fetch_document(Document(url="https://n.com/a"), s)
    assert d.text_source == "page"
    assert d.title == "Clinic news"
    assert d.fetch_error is None


def test_fetch_document_falls_back_to_snippet(fake_session_cls):
    s = fake_session_cls(by_url={"https://n.com/a": requests.ConnectionError("boom")})
    d = fetch_document(Document(url="https://n.com/a", title="Acme fined", snippet="Regulator fines Acme"), s)
    assert d.text_source == "snippet"
    assert d.text == "Acme fined Regulator fines Acme"
    assert "ConnectionError" in d.fetch_error


def test_fetch_rejects_non_html(fake_session_cls, fake_response_cls):
    resp = fake_response_cls("%PDF", headers={"Content-Type": "application/pdf"})
    s = fake_session_cls(by_url={"https://n.com/a": resp})
    d = fetch_document(Document(url="https://n.com/a", title="t"), s)
    assert "unsupported content type" in d.fetch_error
    d = fetch_document(Document(url="https://n.com/report.pdf", title="t"), s)
    assert "not an HTML page" in d.fetch_error


def test_short_pages_count_as_failures(fake_session_cls, fake_response_cls):
    s = fake_session_cls(by_url={"https://n.com/a": fake_response_cls("<p>Subscribe to read.</p>")})
    d = fetch_document(Document(url="https://n.com/a", title="Acme"), s)
    assert d.text_source == "snippet"


def test_fetch_all_skips_documents_with_text(fake_session_cls):
    s = fake_session_cls()
    docs = [Document(url="r1", text="Already here", text_source="provided")]
    seen = []
    fetch_all(docs, workers=2, session=s, progress=lambda d, t: seen.append((d, t)))
    assert s.calls == [] and seen == [(1, 1)]


def test_use_snippets_only():
    docs = use_snippets_only([Document(url="u", title="T", snippet="S")])
    assert docs[0].text == "T S" and docs[0].text_source == "snippet"
