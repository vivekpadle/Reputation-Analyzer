from reputation_analyzer.text import (
    build_pattern,
    chunk_text,
    clean_text,
    extract_mentions,
    split_sentences,
)


def test_pattern_matches_case_aliases_and_flexible_spacing():
    p = build_pattern("Ruby Hall Clinic", ["RHC"])
    assert p.search("I visited ruby hall clinic yesterday")
    assert p.search("Ruby-Hall  Clinic is busy")
    assert p.search("RHC staff were kind")
    assert not p.search("Ruby Halls Clinicians")
    assert not p.search("RHCX")


def test_split_sentences_handles_newlines_and_danda():
    text = "First one. Second one!\nThird line? चौथा वाक्य। Fifth."
    assert split_sentences(text) == [
        "First one.",
        "Second one!",
        "Third line?",
        "चौथा वाक्य।",
        "Fifth.",
    ]


def test_extract_mentions_keeps_window_and_merges_overlaps():
    text = (
        "Intro sentence. Acme opened a store. Customers loved it. Unrelated news here. "
        "Weather was fine. Acme also hired staff. Then lunch."
    )
    p = build_pattern("Acme")
    got = extract_mentions(text, p, window=1)
    assert got == [
        "Intro sentence. Acme opened a store. Customers loved it.",
        "Weather was fine. Acme also hired staff. Then lunch.",
    ]
    assert extract_mentions(text, p, window=0) == ["Acme opened a store.", "Acme also hired staff."]
    assert extract_mentions("No mention at all.", p) == []


def test_extract_mentions_merges_adjacent_windows():
    text = "Acme is here. Middle. Acme again."
    assert extract_mentions(text, build_pattern("Acme"), window=1) == [text]


def test_clean_text_replaces_urls_and_handles():
    assert clean_text("Hi @someone see https://x.com/a  now") == "Hi @user see http now"


def test_chunk_text_respects_size():
    text = " ".join(f"Sentence number {i} is here." for i in range(100))
    chunks = chunk_text(text, max_chars=120, max_chunks=5)
    assert len(chunks) == 5
    assert all(len(c) <= 122 for c in chunks)


def test_long_mention_is_truncated():
    text = "Acme " + "word " * 400 + "."
    (m,) = extract_mentions(text, build_pattern("Acme"))
    assert len(m) <= 702 and m.endswith("…")
