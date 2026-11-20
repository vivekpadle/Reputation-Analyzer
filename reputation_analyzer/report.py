"""Write CSV / JSON outputs and a self-contained HTML report.

The HTML report has no external dependencies (no CDN, no JS libraries): charts
are inline SVG, so the file can be emailed to a client or opened offline.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from html import escape
from pathlib import Path

from .models import AnalysisResult

# --------------------------------------------------------------------------- files


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "entity"


def write_outputs(result: AnalysisResult, out_dir: str | Path, subfolder: bool = True) -> dict[str, Path]:
    out = Path(out_dir)
    if subfolder:
        out = out / f"{slugify(result.entity)}-{result.generated_at:%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "documents": out / "documents.csv",
        "mentions": out / "mentions.csv",
        "summary": out / "summary.json",
        "report": out / "report.html",
    }
    doc_rows = [d.to_row() for d in result.documents]
    _write_csv(paths["documents"], doc_rows)
    _write_csv(paths["mentions"], result.mention_rows())

    payload = {
        "entity": result.entity,
        "aliases": result.aliases,
        "generated_at": result.generated_at.isoformat(),
        "engine": result.engine,
        **result.summary,
        "themes": result.themes,
    }
    paths["summary"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["report"].write_text(render_html(result), encoding="utf-8")
    return paths


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        if not rows:
            fh.write("")
            return
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- charts

_W = 720  # SVG viewBox width


def _fmt(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _bar_path(x0: float, x1: float, y: float, h: float, r: float = 4.0) -> str:
    """Horizontal bar from x0 (baseline, square) to x1 (data end, rounded)."""
    length = abs(x1 - x0)
    if length < 0.5:
        return ""
    r = min(r, length, h / 2)
    if x1 >= x0:
        return (
            f"M{_fmt(x0)},{_fmt(y)} H{_fmt(x1 - r)} Q{_fmt(x1)},{_fmt(y)} {_fmt(x1)},{_fmt(y + r)} "
            f"V{_fmt(y + h - r)} Q{_fmt(x1)},{_fmt(y + h)} {_fmt(x1 - r)},{_fmt(y + h)} "
            f"H{_fmt(x0)} Z"
        )
    return (
        f"M{_fmt(x0)},{_fmt(y)} H{_fmt(x1 + r)} Q{_fmt(x1)},{_fmt(y)} {_fmt(x1)},{_fmt(y + r)} "
        f"V{_fmt(y + h - r)} Q{_fmt(x1)},{_fmt(y + h)} {_fmt(x1 + r)},{_fmt(y + h)} "
        f"H{_fmt(x0)} Z"
    )


def mix_bar_svg(counts: dict[str, int]) -> str:
    total = sum(counts.values())
    if not total:
        return ""
    order = [("negative", "Negative"), ("neutral", "Neutral"), ("positive", "Positive")]
    h, gap, y = 36, 2, 4
    x = 0.0
    parts = []
    nonzero = [k for k, _ in order if counts.get(k)]
    for key, name in order:
        n = counts.get(key, 0)
        if not n:
            continue
        w = _W * n / total
        # 2px surface gap between segments
        seg_w = w - (gap if key != nonzero[-1] else 0)
        share = n / total
        tip = f"{name}: {n} of {total} documents ({share:.0%})"
        rx = 4 if key in (nonzero[0], nonzero[-1]) else 0
        parts.append(
            f'<rect class="mark fill-{key}" x="{_fmt(x)}" y="{y}" width="{_fmt(max(seg_w, 0.5))}" '
            f'height="{h}" rx="{rx}" data-tip="{escape(tip)}" tabindex="0"><title>{escape(tip)}</title></rect>'
        )
        if seg_w > 56:
            parts.append(
                f'<text class="seg-label on-{key}" x="{_fmt(x + 10)}" y="{y + h / 2 + 5}">{share:.0%}</text>'
            )
        x += w
    return (
        f'<svg class="chart" viewBox="0 0 {_W} {h + 8}" role="img" '
        f'aria-label="Share of documents by sentiment">{"".join(parts)}</svg>'
    )


def sources_svg(rows: list[dict], limit: int = 12) -> str:
    rows = rows[:limit]
    if not rows:
        return ""
    label_w, right_pad = 210, 48
    plot_l, plot_r = label_w, _W - right_pad
    mid = (plot_l + plot_r) / 2
    half = (plot_r - plot_l) / 2
    row_h, bar_h, top = 30, 14, 22
    height = top + row_h * len(rows) + 8
    parts = [
        f'<text class="axis-label" x="{_fmt(plot_l)}" y="12">0</text>',
        f'<text class="axis-label" x="{_fmt(mid)}" y="12" text-anchor="middle">50 neutral</text>',
        f'<text class="axis-label" x="{_fmt(plot_r)}" y="12" text-anchor="end">100</text>',
        f'<line class="baseline" x1="{_fmt(mid)}" x2="{_fmt(mid)}" y1="{top - 4}" y2="{height - 4}"/>',
    ]
    for i, r in enumerate(rows):
        y = top + i * row_h
        score = float(r["score"])
        x1 = mid + (score - 50) / 50 * half
        key = "positive" if score >= 50 else "negative"
        name = r["source"] if len(r["source"]) <= 26 else r["source"][:25] + "…"
        n = r["documents"]
        tip = f"{r['source']}: score {score:.0f} from {n} document{'s' if n != 1 else ''}"
        parts.append(
            f'<text class="row-label" x="{label_w - 12}" y="{y + bar_h / 2 + 5}" text-anchor="end">'
            f'{escape(name)} <tspan class="muted">· {n}</tspan></text>'
        )
        path = _bar_path(mid, x1, y, bar_h)
        if path:
            parts.append(f'<path class="mark fill-{key}" d="{path}"/>')
        # value label just past the data end
        vx = x1 + 6 if score >= 50 else x1 - 6
        anchor = "start" if score >= 50 else "end"
        parts.append(
            f'<text class="value-label" x="{_fmt(vx)}" y="{y + bar_h / 2 + 4.5}" text-anchor="{anchor}">'
            f"{score:.0f}</text>"
        )
        # generous invisible hit target for the tooltip
        parts.append(
            f'<rect class="hit" x="0" y="{y - 7}" width="{_W}" height="{row_h}" '
            f'data-tip="{escape(tip)}" tabindex="0"><title>{escape(tip)}</title></rect>'
        )
    return (
        f'<svg class="chart" viewBox="0 0 {_W} {height}" role="img" '
        f'aria-label="Reputation score by source">{"".join(parts)}</svg>'
    )


def timeline_svg(points: list[dict]) -> str:
    if len(points) < 2:
        return ""
    ml, mr, mt, mb = 40, 20, 14, 30
    h = 240
    pw, ph = _W - ml - mr, h - mt - mb
    n = len(points)

    def px(i: int) -> float:
        return ml + pw * i / (n - 1)

    def py(v: float) -> float:
        return mt + ph * (1 - v / 100)

    parts = []
    for v in (0, 25, 50, 75, 100):
        cls = "baseline" if v == 50 else "grid"
        parts.append(f'<line class="{cls}" x1="{ml}" x2="{_W - mr}" y1="{_fmt(py(v))}" y2="{_fmt(py(v))}"/>')
        parts.append(
            f'<text class="axis-label" x="{ml - 8}" y="{_fmt(py(v) + 4)}" text-anchor="end">{v}</text>'
        )
    for i in sorted({0, n // 2, n - 1}):
        anchor = "start" if i == 0 else "end" if i == n - 1 else "middle"
        label = datetime.fromisoformat(points[i]["week"]).strftime("%d %b")
        parts.append(
            f'<text class="axis-label" x="{_fmt(px(i))}" y="{h - 8}" text-anchor="{anchor}">{label}</text>'
        )
    d = " ".join(
        f"{'M' if i == 0 else 'L'}{_fmt(px(i))},{_fmt(py(p['score']))}" for i, p in enumerate(points)
    )
    parts.append(f'<path class="line" d="{d}"/>')
    for i, p in enumerate(points):
        tip = (
            f"Week of {datetime.fromisoformat(p['week']).strftime('%d %b %Y')}: "
            f"score {p['score']:.0f} from {p['documents']} document{'s' if p['documents'] != 1 else ''}"
        )
        cx, cy = _fmt(px(i)), _fmt(py(p["score"]))
        parts.append(f'<circle class="dot" cx="{cx}" cy="{cy}" r="4"/>')
        parts.append(
            f'<circle class="hit" cx="{cx}" cy="{cy}" r="14" data-tip="{escape(tip)}" '
            f'tabindex="0"><title>{escape(tip)}</title></circle>'
        )
    last = points[-1]
    parts.append(
        f'<text class="value-label" x="{_fmt(px(n - 1) - 8)}" y="{_fmt(py(last["score"]) - 10)}" '
        f'text-anchor="end">{last["score"]:.0f}</text>'
    )
    return (
        f'<svg class="chart" viewBox="0 0 {_W} {h}" role="img" '
        f'aria-label="Weekly reputation score">{"".join(parts)}</svg>'
    )


# --------------------------------------------------------------------------- html


def _markdown(md: str) -> str:
    """Tiny Markdown subset for the Claude theme brief (headings, bullets, bold)."""
    out, in_list = [], False
    for raw in md.splitlines():
        line = escape(raw.rstrip())
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        if re.match(r"^\s*[-*]\s+", line):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append("<li>" + re.sub(r"^\s*[-*]\s+", "", line) + "</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            out.append(f"<h3>{m.group(2)}</h3>")
        elif re.match(r"^\s*\d+\.\s+", line):
            out.append(f"<p>{line}</p>")
        elif line.strip():
            out.append(f"<p>{line}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _score100(polarity: float | None) -> str:
    return "" if polarity is None else f"{50 + 50 * polarity:.0f}"


def _quote_list(rows: list[dict], kind: str) -> str:
    if not rows:
        return '<p class="muted">None found.</p>'
    items = []
    for r in rows:
        meta = " · ".join(x for x in (escape(r["source"]), r.get("published", "")) if x)
        link = (
            f'<a href="{escape(r["url"])}" target="_blank" rel="noopener">{meta}</a>'
            if r["url"].startswith("http")
            else meta
        )
        items.append(
            f'<li><span class="pill pill-{kind}">{_score100(r["polarity"])}</span>'
            f'<div><p>{escape(r["text"])}</p><p class="muted small">{link}</p></div></li>'
        )
    return f'<ul class="quotes">{"".join(items)}</ul>'


_CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--page:#f9f9f7;--text-primary:#0b0b0b;
--text-secondary:#52514e;--muted:#898781;--grid:#e1e0d9;--baseline:#c3c2b7;
--border:rgba(11,11,11,.10);--positive:#2a78d6;--negative:#e34948;--neutral:#b5b4ad;
--on-positive:#fff;--on-negative:#fff;--on-neutral:#0b0b0b}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#f0efec;--text-secondary:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--baseline:#383835;--border:rgba(255,255,255,.10);--positive:#3987e5;--negative:#e66767;
--neutral:#5f5e5a;--on-neutral:#f0efec}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;
--text-primary:#f0efec;--text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--baseline:#383835;
--border:rgba(255,255,255,.10);--positive:#3987e5;--negative:#e66767;--neutral:#5f5e5a;--on-neutral:#f0efec}
*{box-sizing:border-box}
body{margin:0}
.viz-root{background:var(--page);color:var(--text-primary);min-height:100vh;
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;padding:32px 20px 64px}
.wrap{max-width:880px;margin:0 auto}
header h1{font-size:28px;line-height:1.2;margin:0 0 4px;font-weight:650}
header p{margin:0;color:var(--text-secondary)}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:20px 22px;margin-top:16px}
.card h2{font-size:15px;margin:0 0 2px;font-weight:600}
.card .sub{margin:0 0 14px;color:var(--text-secondary);font-size:13px}
.hero{display:grid;grid-template-columns:auto 1fr;gap:8px 28px;align-items:end}
.hero .num{font-size:64px;line-height:1;font-weight:650;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.hero .num small{font-size:20px;color:var(--text-secondary);font-weight:500;margin-left:4px}
.hero .band{font-size:18px;font-weight:600}
.hero .facts{color:var(--text-secondary);font-size:13px}
.stats{display:flex;gap:28px;flex-wrap:wrap;margin-top:16px;padding-top:14px;border-top:1px solid var(--border)}
.stats div{font-size:13px;color:var(--text-secondary)}
.stats b{display:block;font-size:20px;color:var(--text-primary);font-variant-numeric:tabular-nums}
.chart{width:100%;height:auto;display:block;overflow:visible}
.chart text{font:12px system-ui,-apple-system,"Segoe UI",sans-serif}
.fill-positive{fill:var(--positive)}.fill-negative{fill:var(--negative)}.fill-neutral{fill:var(--neutral)}
.seg-label{font-weight:600;font-size:13px!important}
.on-positive{fill:var(--on-positive)}.on-negative{fill:var(--on-negative)}.on-neutral{fill:var(--on-neutral)}
.row-label{fill:var(--text-primary)}.row-label .muted,.axis-label{fill:var(--muted)}
.value-label{fill:var(--text-secondary);font-variant-numeric:tabular-nums}
.grid{stroke:var(--grid);stroke-width:1}.baseline{stroke:var(--baseline);stroke-width:1}
.line{fill:none;stroke:var(--positive);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.dot{fill:var(--positive);stroke:var(--surface-1);stroke-width:2}
.hit{fill:transparent;cursor:default;outline:none}
.hit:focus-visible{stroke:var(--text-secondary);stroke-width:1}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:10px;font-size:13px;color:var(--text-secondary)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{width:10px;height:10px;border-radius:3px;display:inline-block}
.quotes{list-style:none;margin:0;padding:0}
.quotes li{display:flex;gap:12px;padding:10px 0;border-top:1px solid var(--border)}
.quotes li:first-child{border-top:0}
.quotes p{margin:0}
.pill{flex:none;min-width:38px;height:24px;border-radius:12px;font-size:12px;font-weight:600;
display:inline-flex;align-items:center;justify-content:center;font-variant-numeric:tabular-nums}
.pill-negative{background:var(--negative);color:var(--on-negative)}
.pill-positive{background:var(--positive);color:var(--on-positive)}
.muted{color:var(--muted)}.small{font-size:12px}
a{color:inherit;text-underline-offset:2px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.cols .card{margin-top:0}
@media (max-width:720px){.cols{grid-template-columns:1fr}.hero{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 8px;border-top:1px solid var(--border);vertical-align:top}
th{color:var(--text-secondary);font-weight:600;border-top:0;cursor:pointer;user-select:none;white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td.date{white-space:nowrap}
.table-wrap{max-height:520px;overflow:auto}
.themes h3{font-size:14px;margin:14px 0 4px}.themes ul{margin:0;padding-left:20px}
.note{font-size:12px;color:var(--muted);margin-top:24px}
#tip{position:fixed;pointer-events:none;background:var(--text-primary);color:var(--surface-1);
font-size:12px;padding:6px 9px;border-radius:6px;max-width:280px;opacity:0;transition:opacity .08s;z-index:10}
"""

_JS = """
(function(){
  var tip=document.getElementById('tip');
  function show(e,t){tip.textContent=t;tip.style.opacity=1;move(e)}
  function move(e){var x=e.clientX+14,y=e.clientY+14;
    if(x+tip.offsetWidth>innerWidth-8)x=e.clientX-tip.offsetWidth-14;
    if(y+tip.offsetHeight>innerHeight-8)y=e.clientY-tip.offsetHeight-14;
    tip.style.left=x+'px';tip.style.top=y+'px'}
  document.querySelectorAll('[data-tip]').forEach(function(el){
    var t=el.getAttribute('data-tip');var title=el.querySelector('title');if(title)title.remove();
    el.addEventListener('mouseenter',function(e){show(e,t)});
    el.addEventListener('mousemove',move);
    el.addEventListener('mouseleave',function(){tip.style.opacity=0});
    el.addEventListener('focus',function(){var r=el.getBoundingClientRect();
      show({clientX:r.left+r.width/2,clientY:r.top},t)});
    el.addEventListener('blur',function(){tip.style.opacity=0});
  });
  document.querySelectorAll('table.sortable th').forEach(function(th,i){
    th.addEventListener('click',function(){
      var tb=th.closest('table').tBodies[0],rows=[].slice.call(tb.rows);
      var asc=th.dataset.dir!=='asc';th.dataset.dir=asc?'asc':'desc';
      rows.sort(function(a,b){var x=a.cells[i].dataset.v||a.cells[i].textContent,
        y=b.cells[i].dataset.v||b.cells[i].textContent,nx=parseFloat(x),ny=parseFloat(y);
        var c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return asc?c:-c});
      rows.forEach(function(r){tb.appendChild(r)});
    });
  });
})();
"""


def render_html(result: AnalysisResult) -> str:
    s = result.summary
    score = s.get("reputation_score")
    counts = s.get("label_counts", {})
    share = s.get("label_share", {})
    title = f"Reputation report: {result.entity}"

    hero_num = "–" if score is None else f"{score:.0f}"
    facts = (
        f"Based on {s.get('documents_scored', 0)} documents and {s.get('mentions_scored', 0)} "
        f"mentions · confidence: {s.get('confidence')}"
    )
    rating = s.get("average_rating")
    stats = [
        ("Documents found", s.get("documents_total", 0)),
        ("Pages read in full", s.get("pages_read", 0)),
        ("Snippet only", s.get("snippet_only", 0)),
        ("No mention of entity", s.get("documents_without_mentions", 0)),
    ]
    if rating is not None:
        stats.append(("Average rating", f"{rating:.2f} ★"))
    stats_html = "".join(f"<div><b>{escape(str(v))}</b>{escape(k)}</div>" for k, v in stats)

    legend = "".join(
        f'<span><i class="fill-{k}" style="background:var(--{k})"></i>{k.title()} '
        f"{counts.get(k, 0)} ({share.get(k, 0):.0%})</span>"
        for k in ("negative", "neutral", "positive")
    )

    sections = [
        f"""<section class="card hero-card">
  <div class="hero">
    <div class="num">{hero_num}<small>/ 100</small></div>
    <div><div class="band">{escape(s.get("band") or "Not enough data")}</div>
    <div class="facts">{escape(facts)}</div></div>
  </div>
  <div class="stats">{stats_html}</div>
</section>"""
    ]

    if sum(counts.values()):
        sections.append(f"""<section class="card">
  <h2>Sentiment mix</h2><p class="sub">Share of documents that are mainly negative, neutral or positive about {escape(result.entity)}</p>
  {mix_bar_svg(counts)}
  <div class="legend">{legend}</div>
</section>""")

    tl = timeline_svg(s.get("timeline", []))
    if tl:
        sections.append(f"""<section class="card">
  <h2>Score over time</h2><p class="sub">Weekly average, 0–100 (50 = neutral)</p>{tl}
</section>""")

    src = sources_svg(s.get("by_source", []))
    if src:
        sections.append(f"""<section class="card">
  <h2>Score by source</h2><p class="sub">Top sources by number of documents (number after the name). Bars run left of 50 when coverage is negative.</p>{src}
</section>""")

    if result.themes:
        sections.append(f"""<section class="card themes">
  <h2>What's driving the score</h2><p class="sub">Summary written by Claude from the most positive and most negative passages</p>
  {_markdown(result.themes)}
</section>""")

    sections.append(f"""<div class="cols" style="margin-top:16px">
<section class="card"><h2>Most negative passages</h2><p class="sub">Score 0–100 per passage</p>{_quote_list(s.get("most_negative", []), "negative")}</section>
<section class="card"><h2>Most positive passages</h2><p class="sub">Score 0–100 per passage</p>{_quote_list(s.get("most_positive", []), "positive")}</section>
</div>""")

    rows = []
    for d in sorted(result.documents, key=lambda d: (d.polarity is None, d.polarity or 0)):
        title_txt = escape(d.title or d.url)
        link = (
            f'<a href="{escape(d.url)}" target="_blank" rel="noopener">{title_txt}</a>'
            if d.url.startswith("http")
            else title_txt
        )
        date = d.published.date().isoformat() if d.published else ""
        sc = _score100(d.polarity)
        rows.append(
            f"<tr><td>{link}</td><td>{escape(d.source)}</td><td class='date' data-v='{date}'>{date}</td>"
            f"<td class='num' data-v='{sc or -1}'>{sc or '–'}</td><td>{escape(d.label or 'no mention')}</td>"
            f"<td class='num'>{len(d.mentions)}</td><td>{escape(d.text_source)}</td></tr>"
        )
    sections.append(f"""<section class="card">
  <h2>All documents</h2><p class="sub">Click a column to sort</p>
  <div class="table-wrap"><table class="sortable"><thead><tr><th>Title</th><th>Source</th><th>Date</th>
  <th class="num">Score</th><th>Label</th><th class="num">Mentions</th><th>Text from</th></tr></thead>
  <tbody>{"".join(rows)}</tbody></table></div>
</section>""")

    errors = s.get("source_errors") or []
    err_html = f"<p>Sources that failed: {escape('; '.join(errors))}</p>" if errors else ""
    method = s.get("method", {})
    half_life = method.get("half_life_days")
    sections.append(f"""<div class="note">
  <p>Method: each passage that names {escape(result.entity)} is scored by <code>{escape(result.engine)}</code>
  (positive − negative probability, −1 to 1). A document's score is the mean of its passages; the overall score is
  the mean across documents mapped to 0–100 and pulled toward 50 by a prior of {method.get("prior_weight")} documents
  so a handful of results can't give an extreme score.{f" Recent documents weigh more (half-life {half_life:g} days)." if half_life else ""}
  Query: “{escape(str(s.get("query", "")))}” · sources: {escape(", ".join(s.get("sources_used", [])))} ·
  generated {result.generated_at:%d %b %Y %H:%M} UTC.</p>{err_html}
</div>""")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>{_CSS}</style></head>
<body><div class="viz-root"><div class="wrap">
<header><h1>{escape(result.entity)}</h1>
<p>Reputation report{(" · also matched: " + escape(", ".join(result.aliases))) if result.aliases else ""}</p></header>
{"".join(sections)}
</div><div id="tip" role="tooltip"></div></div>
<script>{_JS}</script></body></html>"""
