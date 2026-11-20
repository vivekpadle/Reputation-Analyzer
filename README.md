# Reputation Analyzer

Measure how a company, hospital, institute, product or person is talked about online.

It finds news articles, web pages or reviews that mention the name, keeps only the passages that are actually about it, scores each passage with a transformer sentiment model, and turns the result into a **0–100 reputation score** with a shareable HTML report.

```
$ reputation-analyzer "Acme Hospital" --alias "Acme" --source gdelt --source ddg
[  0%] Collecting sources…
[ 15%] Reading pages…
[ 60%] Finding mentions…
[ 65%] Scoring sentiment…
[ 90%] Summarising…

Acme Hospital: 62/100 (Positive, high confidence)
  41 documents, 97 mentions · positive 49% · neutral 34% · negative 17%

Report: reports/acme-hospital-20261120-191500/report.html
```

*(Illustrative output for a made-up name.)*

## What you get

- **Reputation score (0–100)**: 50 is neutral. See [How the score works](#how-the-score-works).
- **Sentiment mix**: the share of documents that are mainly positive, neutral or negative.
- **Score over time**: weekly trend, so you can catch a bad news cycle.
- **Score by source**: which sites or platforms are kind or harsh.
- **Most negative / most positive passages**, with links, to show what drives the score.
- **"What's driving the score"** (optional): Claude reads the passages and writes up the themes plus recommended actions.
- `documents.csv`, `mentions.csv` and `summary.json` for your own analysis or a CRM.

The report is a single HTML file with inline SVG charts and no external scripts. You can email it to a client or open it offline. It supports light and dark mode.

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/vivekpadle/Reputation-Analyzer.git
cd Reputation-Analyzer
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -e ".[ml,extract,search]"   # transformer model, better text extraction, DuckDuckGo
# or everything (adds VADER, Claude themes, Streamlit dashboard):
pip install -e ".[all]"
```

The first run downloads the sentiment model from Hugging Face (~500 MB). After that, runs work offline apart from fetching pages.

### Google Colab

```python
!pip install -q "reputation-analyzer[ml,extract,search] @ git+https://github.com/vivekpadle/Reputation-Analyzer.git"
!reputation-analyzer "Infosys" --source gdelt -n 50
```

Turn on the GPU runtime to make scoring much faster.

## Usage

```bash
# News from the last 3 months (free, no API key)
reputation-analyzer "Ruby Hall Clinic" --alias "Ruby Hall"

# Several sources, more results, newer news counts more
reputation-analyzer "MIT-WPU" --alias "MIT World Peace University" \
    --source gdelt --source ddg -n 100 --half-life 30

# A client's own review export (Google reviews, Practo, Justdial, app stores…)
reputation-analyzer "Sample Clinic" --no-search --reviews examples/sample_reviews.csv

# Pages you already have
reputation-analyzer "Acme Realty" --no-search --urls my_links.txt

# Fast scan: score titles and snippets without downloading pages
reputation-analyzer "Acme Realty" --source ddg --no-fetch

# Hindi / Hinglish / mixed-language content
reputation-analyzer "Acme Realty" --engine multilingual --language any

# Add the Claude-written summary of themes and actions
export ANTHROPIC_API_KEY=...
reputation-analyzer "Ruby Hall Clinic" --themes
```

Run `reputation-analyzer --help` for every option. `python -m reputation_analyzer` also works.

### Dashboard

```bash
pip install -e ".[all]"
streamlit run app.py
```

The dashboard has the same options as the CLI in a sidebar. It also takes a review CSV upload and shows the report in the page, with buttons to download the HTML, CSV and JSON files.

### From Python

```python
from reputation_analyzer import AnalysisConfig, analyze
from reputation_analyzer.report import write_outputs

result = analyze(AnalysisConfig(entity="Ruby Hall Clinic", aliases=["Ruby Hall"], sources=["gdelt"]))
print(result.summary["reputation_score"], result.summary["band"])
write_outputs(result, "reports")
```

## Sources

| Source | Flag | Key needed | Notes |
|---|---|---|---|
| GDELT DOC 2.0 news API | `--source gdelt` (default) | none | Global news, about 3 months back, up to 250 articles. Filter by language with `--language`. |
| DuckDuckGo metasearch | `--source ddg` | none | News results through the [`ddgs`](https://pypi.org/project/ddgs/) package. Can be rate-limited. |
| SerpAPI (Google results) | `--source serpapi` | `SERPAPI_API_KEY` | Real Google results through an official API. |
| Brave Search API | `--source brave` | `BRAVE_API_KEY` | Independent web index. |
| Your URLs | `--urls file.txt` | none | One URL per line. |
| Your reviews | `--reviews file.csv` | none | Needs a `text`/`review`/`content`/`comment` column. Optional columns: `date`, `rating`, `source`, `url`, `title`. |

Put API keys in your environment. See `.env.example`.

Pages are downloaded in parallel, with retries and timeouts. The text is extracted with [trafilatura](https://trafilatura.readthedocs.io/) when it's installed, or with BeautifulSoup otherwise. If a page can't be read (paywall, block, PDF), its title and snippet are used instead, and the report shows how many documents that applied to.

Respect each site's terms of use. This tool reads public pages at a modest rate and does not get around paywalls or CAPTCHAs.

## Sentiment engines

| `--engine` | Model | When to use |
|---|---|---|
| `transformer` (default) | [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | English news, reviews and social posts |
| `multilingual` | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | Hindi, Marathi, Hinglish and other languages |
| `vader` | VADER lexicon | No model download. Fast, but less accurate. |

`--model` accepts any Hugging Face text-classification model that predicts negative/neutral/positive, or 1–5 stars (for example `nlptown/bert-base-multilingual-uncased-sentiment`). Scoring runs in batches on the GPU (CUDA or Apple MPS) when one is available.

## How the score works

1. **Find mentions.** Only sentences that name the entity (or an `--alias`) are scored, plus one neighbouring sentence on each side (`--window`). Text about other things on the same page is ignored. Review CSVs are scored in full, because every review is about the entity.
2. **Score each passage.** Polarity = P(positive) − P(negative), from −1 to 1.
3. **Score each document.** A document's score is the mean of its passages, so a long article with 20 mentions counts once, the same as a short review.
4. **Overall score.**
   `score = 50 + 50 × Σ(wᵢ·pᵢ) / (Σwᵢ + prior)`
   The `--prior` (default 2) pulls small samples toward 50: one glowing article scores about 67, not 100. With `--half-life N`, a document that is N days old counts half as much as one from today.
5. **Bands:** ≥65 Strong · 55–65 Positive · 45–55 Mixed · 35–45 At risk · <35 Poor. Confidence is *low* below 10 scored documents, *medium* below 30, and *high* from 30 up.

## Project layout

```
reputation_analyzer/
  sources.py    search providers + URL / review CSV loaders
  fetch.py      parallel page download and text extraction
  text.py       cleaning, sentence splitting, finding mentions
  sentiment.py  transformer / VADER engines
  scoring.py    document scores, reputation score, summary
  pipeline.py   AnalysisConfig + analyze()
  report.py     HTML report, CSV and JSON outputs
  llm.py        optional Claude theme summary
  cli.py        command line
app.py          Streamlit dashboard
tests/          pytest suite (no network or model download needed)
legacy/         the original Colab script, for reference
```

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check . && ruff format --check .
```

The tests use a fake sentiment engine and fake HTTP sessions, so they run in under a second and need no network access. GitHub Actions runs lint and tests on every push and pull request.

## What changed from v1

v1 was a Colab notebook export. Its main problems:

- **It couldn't run as a script.** It contained `!pip install` lines and hard-coded `/content/` paths.
- **The 1–10 score was meaningless.** It min-max scaled the three average class scores against each other, so the largest class always became 10 and the smallest 1, whatever the data said.
- **Google HTML scraping** relied on one CSS class, had no page limit and no delays, and Google regularly blocks it. v2 replaces it with official or free APIs.
- **Noise was scored as sentiment**: `Title:`/`URL:` lines, "Failed to fetch" messages, and every sentence on a page whether or not it mentioned the entity.
- **Text handling.** Long articles were cut off at the model's 512-token limit, or crashed the unbatched path. The model was also reloaded for every line.
- **The VADER chart** plotted a single number. The correlation heatmap of three probabilities that always sum to 1 carried no information.
- **README errors.** The README named the wrong script, used a placeholder clone URL, and listed incomplete dependencies.

The original script is kept in `legacy/` for reference.

## Acknowledgments

[Cardiff NLP TweetEval models](https://huggingface.co/cardiffnlp) · [GDELT Project](https://www.gdeltproject.org/) · [Hugging Face Transformers](https://huggingface.co/docs/transformers) · [trafilatura](https://trafilatura.readthedocs.io/) · [ddgs](https://pypi.org/project/ddgs/) · [VADER](https://github.com/cjhutto/vaderSentiment) · [Streamlit](https://streamlit.io/) · [Anthropic Claude API](https://docs.claude.com/)
