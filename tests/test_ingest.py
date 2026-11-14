"""Ingestion: boilerplate removal, chunking invariants, overlap-safe gold labels."""
from src.build_demo_corpus import find_chunks_for
from src.ingest import split_pages, strip_boilerplate

TOPICS = ["turbines", "flanges", "valves", "bearings", "gaskets", "couplings"]
HEADER = "Aurora Operations Handbook - internal circulation"


def _page(i: int, topic: str, extra_middle: str = "") -> dict:
    """A realistic page: 2 header lines, body, 2 footer lines."""
    body = [
        "Line one of the body about {t} intervals and tolerances.".format(t=topic),
        extra_middle or "Line two of the body describing the inspection procedure in detail.",
        "Line three of the body with a reported figure of 412 units.",
        "Line four of the body closing the section with a note on records.",
    ]
    lines = [HEADER, f"Page {i + 1}", *body, HEADER]
    return {"text": "\n".join(ln for ln in lines if ln), "page": i + 1, "source": "doc.pdf"}


def _pages() -> list[dict]:
    return [_page(i, topic) for i, topic in enumerate(TOPICS)]


def test_footer_removed_unique_body_kept():
    cleaned = strip_boilerplate(_pages())
    for page, topic in zip(cleaned, TOPICS):
        assert HEADER not in page["text"]
        assert f"about {topic}" in page["text"]
        assert "inspection procedure" in page["text"]
    assert "Page 1" not in cleaned[0]["text"]


def test_repeated_mid_page_line_is_content():
    pages = [_page(i, topic, extra_middle="Templated body line that repeats everywhere.")
             for i, topic in enumerate(TOPICS)]
    cleaned = strip_boilerplate(pages)
    assert all("Templated body line" in p["text"] for p in cleaned)


def test_short_pages_are_left_alone():
    pages = [{"text": f"Page {i}\n{HEADER}", "page": i, "source": "d.pdf"} for i in range(6)]
    cleaned = strip_boilerplate(pages)
    assert all(HEADER in p["text"] for p in cleaned)


def test_one_off_text_survives():
    pages = _pages()
    pages[0]["text"] = pages[0]["text"].replace(
        "Line four of the body", "One-off legal notice. Line four of the body"
    )
    assert "One-off legal notice" in strip_boilerplate(pages)[0]["text"]


def test_chunks_respect_size_and_are_unique():
    text = " ".join(f"Sentence number {i} reports a value of {i * 7} units." for i in range(80))
    chunks = split_pages([{"text": text, "page": 1, "source": "d.pdf"}], 300, 60)
    assert chunks
    assert all(len(c["text"]) <= 300 for c in chunks)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    assert all(c["page"] == 1 for c in chunks)


def test_chunking_does_not_cut_words():
    text = " ".join(f"token{i:03d}" for i in range(200))
    chunks = split_pages([{"text": text, "page": 1, "source": "d.pdf"}], 200, 40)
    joined = " ".join(c["text"] for c in chunks)
    for token in ("token000", "token099", "token199"):
        assert token in joined


def test_chunks_overlap():
    text = " ".join(f"Sentence {i} has a distinctive payload marker{i:03d}." for i in range(40))
    chunks = split_pages([{"text": text, "page": 1, "source": "d.pdf"}], 250, 80)
    assert len(chunks) > 1
    # consecutive chunks must share text, or retrieval across boundaries breaks
    assert any(
        chunks[i]["text"][-20:] in chunks[i + 1]["text"] for i in range(len(chunks) - 1)
    )


def test_gold_labels_cover_every_overlapping_chunk():
    chunks = [
        {"chunk_id": "a", "text": "Intro. For Norvik Plant, the reported headcount was 12 employees. Outro."},
        {"chunk_id": "b", "text": "For Norvik Plant, the reported headcount was 12 employees. More text."},
        {"chunk_id": "c", "text": "Unrelated passage about something else entirely."},
    ]
    got = find_chunks_for(
        "For Norvik Plant, the reported headcount was 12 employees.", chunks, "12 employees"
    )
    assert set(got) == {"a", "b"}


def test_gold_labels_fall_back_to_value():
    chunks = [{"chunk_id": "z", "text": "Only the value 4455 appears here, without the sentence."}]
    assert find_chunks_for("missing sentence", chunks, "4455") == ["z"]
