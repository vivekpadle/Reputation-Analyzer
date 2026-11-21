"""Streamlit dashboard: ``streamlit run app.py``"""

from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from reputation_analyzer.pipeline import AnalysisConfig, analyze
from reputation_analyzer.report import render_html, slugify
from reputation_analyzer.sentiment import make_engine
from reputation_analyzer.sources import SourceError

st.set_page_config(page_title="Reputation Analyzer", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Loading sentiment model (first run downloads ~500 MB)…")
def get_engine(kind: str, model: str | None):
    return make_engine(kind, model or None)


def to_csv(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


with st.sidebar:
    st.header("What to analyse")
    entity = st.text_input("Name", placeholder="e.g. Ruby Hall Clinic")
    aliases_raw = st.text_input("Other names (comma separated)", placeholder="Ruby Hall, RHC")
    query = st.text_input("Search query (optional)", placeholder="defaults to the name")

    st.header("Sources")
    sources = st.multiselect(
        "Search",
        ["gdelt", "ddg", "serpapi", "brave"],
        default=["gdelt"],
        help="gdelt and ddg are free. serpapi and brave need SERPAPI_API_KEY / BRAVE_API_KEY.",
    )
    max_results = st.slider("Results per source", 10, 250, 50, step=10)
    timespan = st.selectbox("News window (GDELT)", ["1w", "2w", "1month", "3months"], index=3)
    reviews = st.file_uploader(
        "Reviews CSV (optional)",
        type=["csv"],
        help="Needs a 'text' or 'review' column; date, rating, source optional.",
    )
    urls = st.text_area("Extra URLs (optional, one per line)")

    st.header("Analysis")
    engine_kind = st.selectbox(
        "Sentiment engine",
        ["transformer", "multilingual", "vader"],
        help="multilingual handles Hindi/Hinglish; vader is fast but rough.",
    )
    model = st.text_input("Custom Hugging Face model (optional)")
    fetch_pages = st.toggle(
        "Read full pages", value=True, help="Off = score titles/snippets only (much faster)."
    )
    half_life = st.number_input("Recency half-life in days (0 = off)", 0, 365, 0)
    themes = st.checkbox("Explain the score with Claude", help="Needs ANTHROPIC_API_KEY.")
    run = st.button("Analyse", type="primary", use_container_width=True, disabled=not entity.strip())

st.title("Reputation Analyzer")

if not run:
    st.info("Enter a name in the sidebar and press **Analyse**.")
    st.stop()

tmp = Path(tempfile.mkdtemp())
reviews_path = urls_path = None
if reviews is not None:
    reviews_path = tmp / "reviews.csv"
    reviews_path.write_bytes(reviews.getvalue())
if urls.strip():
    urls_path = tmp / "urls.txt"
    urls_path.write_text(urls, encoding="utf-8")

cfg = AnalysisConfig(
    entity=entity.strip(),
    aliases=[a.strip() for a in aliases_raw.split(",") if a.strip()],
    query=query.strip() or None,
    sources=sources,
    max_results=max_results,
    timespan=timespan,
    urls_file=urls_path,
    reviews_csv=reviews_path,
    fetch_pages=fetch_pages,
    engine=engine_kind,
    model=model.strip() or None,
    half_life_days=float(half_life) or None,
    themes=themes,
)

bar = st.progress(0.0, text="Starting…")
try:
    engine = get_engine(engine_kind, cfg.model)
    result = analyze(cfg, engine=engine, progress=lambda stage, f: bar.progress(min(f, 1.0), text=stage))
except (SourceError, RuntimeError, ValueError) as exc:
    bar.empty()
    st.error(str(exc))
    st.stop()
bar.empty()

s = result.summary
for err in s.get("source_errors", []):
    st.warning(err)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Reputation score",
    "–" if s["reputation_score"] is None else f"{s['reputation_score']:.0f} / 100",
    s.get("band") or "",
)
c2.metric("Documents scored", s["documents_scored"], f"of {s['documents_total']} found", delta_color="off")
c3.metric("Positive", f"{s['label_share']['positive']:.0%}")
c4.metric("Negative", f"{s['label_share']['negative']:.0%}")

html = render_html(result)
components.html(html, height=1600, scrolling=True)

name = f"{slugify(result.entity)}-{result.generated_at:%Y%m%d-%H%M}"
d1, d2, d3, d4 = st.columns(4)
d1.download_button("Report (HTML)", html, f"{name}.html", "text/html")
d2.download_button(
    "Documents (CSV)", to_csv([d.to_row() for d in result.documents]), f"{name}-documents.csv", "text/csv"
)
d3.download_button("Mentions (CSV)", to_csv(result.mention_rows()), f"{name}-mentions.csv", "text/csv")
d4.download_button(
    "Summary (JSON)",
    json.dumps({**s, "themes": result.themes}, indent=2, default=str),
    f"{name}-summary.json",
    "application/json",
)
