"""Command line: ``reputation-analyzer "Ruby Hall Clinic" --alias "Ruby Hall"``."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .pipeline import AnalysisConfig, analyze
from .report import write_outputs
from .sources import SEARCH_SOURCES, SourceError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reputation-analyzer",
        description="Measure the online reputation of a company, product or person.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("entity", help='Name to analyse, e.g. "Ruby Hall Clinic"')
    p.add_argument(
        "-a",
        "--alias",
        action="append",
        default=[],
        dest="aliases",
        help="Other names that count as a mention (repeatable)",
    )
    p.add_argument("-q", "--query", help="Search query (defaults to the entity name)")
    p.add_argument(
        "-s",
        "--source",
        action="append",
        dest="sources",
        choices=sorted(SEARCH_SOURCES),
        help="Where to search (repeatable). Default: gdelt",
    )
    p.add_argument(
        "--no-search", action="store_true", help="Don't search the web; analyse only --urls / --reviews"
    )
    p.add_argument("-n", "--max-results", type=int, default=50, help="Results per source")
    p.add_argument("--timespan", default="3months", help="GDELT window: 1w, 1month, 3months...")
    p.add_argument("--language", default="english", help="GDELT source language filter, or 'any'")
    p.add_argument("--urls", dest="urls_file", help="Text file with one URL per line")
    p.add_argument(
        "--reviews", dest="reviews_csv", help="CSV of reviews/comments (needs a 'text' or 'review' column)"
    )
    p.add_argument(
        "--no-fetch", action="store_true", help="Don't download pages; score titles and snippets only (fast)"
    )
    p.add_argument("--workers", type=int, default=8, help="Parallel page downloads")
    p.add_argument("-e", "--engine", default="transformer", choices=["transformer", "multilingual", "vader"])
    p.add_argument("-m", "--model", help="Hugging Face model id (overrides the engine default)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--window", type=int, default=1, help="Sentences kept on each side of a mention")
    p.add_argument(
        "--half-life",
        type=float,
        dest="half_life_days",
        help="Weight recent documents more; half-life in days",
    )
    p.add_argument(
        "--prior",
        type=float,
        default=2.0,
        dest="prior_weight",
        help="Shrink the score toward 50 by this many 'neutral documents'",
    )
    p.add_argument(
        "--themes", action="store_true", help="Explain the score with Claude (needs ANTHROPIC_API_KEY)"
    )
    p.add_argument("--themes-model", help="Claude model id for --themes")
    p.add_argument("-o", "--out", default="reports", help="Output folder")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.no_search:
        sources = []
        if not (args.urls_file or args.reviews_csv):
            print("error: --no-search needs --urls or --reviews", file=sys.stderr)
            return 2
    else:
        sources = args.sources or ["gdelt"]

    cfg = AnalysisConfig(
        entity=args.entity,
        aliases=args.aliases,
        query=args.query,
        sources=sources,
        max_results=args.max_results,
        timespan=args.timespan,
        language=args.language,
        urls_file=args.urls_file,
        reviews_csv=args.reviews_csv,
        fetch_pages=not args.no_fetch,
        workers=args.workers,
        engine=args.engine,
        model=args.model,
        batch_size=args.batch_size,
        window=args.window,
        prior_weight=args.prior_weight,
        half_life_days=args.half_life_days,
        themes=args.themes,
        themes_model=args.themes_model,
    )

    last = {"stage": None}

    def progress(stage: str, frac: float) -> None:
        if stage != last["stage"]:
            print(f"[{frac:4.0%}] {stage}…", file=sys.stderr)
            last["stage"] = stage

    try:
        result = analyze(cfg, progress=progress)
    except (SourceError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    paths = write_outputs(result, args.out)
    s = result.summary
    score = s["reputation_score"]
    print()
    if score is None:
        print(f"No passages mentioning {cfg.entity!r} were found in {s['documents_total']} documents.")
        print("Try --alias with other spellings, or --window 2.")
    else:
        share = s["label_share"]
        print(f"{cfg.entity}: {score:.0f}/100 ({s['band']}, {s['confidence']} confidence)")
        print(
            f"  {s['documents_scored']} documents, {s['mentions_scored']} mentions · "
            f"positive {share['positive']:.0%} · neutral {share['neutral']:.0%} · "
            f"negative {share['negative']:.0%}"
        )
    for err in s.get("source_errors", []):
        print(f"  warning: {err}")
    print(f"\nReport: {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
