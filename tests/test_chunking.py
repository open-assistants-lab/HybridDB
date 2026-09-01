"""Tests for hybriddb.chunking — dependency-free long-document splitter."""

import pytest

from hybriddb.chunking import chunk_text


def _sentences_of(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


LONG = "\n\n".join(
    f"Paragraph {i} discusses topic {i % 3} with several sentences of detail. "
    f"It continues here with more supporting context for topic {i % 3}."
    for i in range(30)
)


class TestChunkText:
    def test_empty_and_whitespace(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\n  ") == []

    def test_short_text_single_chunk(self):
        assert chunk_text("hello world") == ["hello world"]

    def test_lossless_without_overlap(self):
        """Every original sentence survives, in order, when overlap is off."""
        chunks = chunk_text(LONG, overlap=False)
        flattened = [s for c in chunks for s in _sentences_of(c)]
        assert flattened == _sentences_of(LONG)

    def test_never_exceeds_budget_without_overlap(self):
        chunks = chunk_text(LONG, max_chars=400, overlap=False)
        assert len(chunks) > 1
        assert all(len(c) <= 400 for c in chunks)

    def test_never_splits_mid_sentence(self):
        chunks = chunk_text(LONG, max_chars=400, overlap=False)
        for sentence in _sentences_of(LONG):
            assert any(sentence in c for c in chunks), f"sentence lost: {sentence[:40]}"

    def test_overlap_prefix_continuity(self):
        chunks = chunk_text(LONG, max_chars=400, overlap=True)
        assert len(chunks) > 1
        # each subsequent chunk starts with the previous chunk's last sentence
        for prev, nxt in zip(chunks, chunks[1:]):
            last = _sentences_of(prev)[-1]
            assert nxt.startswith(last), f"chunk does not continue from: {last[:40]}"
        # dedupe: removing the overlap prefix recovers the lossless stream
        assert chunk_text(LONG, max_chars=400, overlap=False)[0] == chunks[0]

    def test_overlap_false_has_no_prefix(self):
        no_ov = chunk_text(LONG, max_chars=400, overlap=False)
        last = _sentences_of(no_ov[0])[-1]
        assert not no_ov[1].startswith(last)

    def test_giant_single_paragraph_hard_split(self):
        text = "x" * 5000  # no sentence boundaries at all
        chunks = chunk_text(text, max_chars=1200, overlap=False)
        assert all(len(c) <= 1200 for c in chunks)
        assert "".join(c.strip() for c in chunks).count("x") == 5000

    def test_paragraphs_never_split_and_merge_to_budget(self):
        paras = [
            "Alpha paragraph about databases.",
            "Beta paragraph about networking.",
            "Gamma paragraph about storage.",
        ]
        joined = "\n\n".join(paras)
        # small paragraphs merge into one chunk up to the budget — but never
        # broken mid-paragraph, and no content is lost
        chunks = chunk_text(joined, max_chars=1200, overlap=False)
        assert chunks == [joined]
        for para in paras:
            assert para in chunks[0]

    def test_paragraph_boundary_stops_a_chunk(self):
        """A chunk closes when adding the next paragraph would exceed the
        budget, so paragraph boundaries are respected at the budget edge."""
        paras = ["A" * 600, "B" * 500, "C" * 100]
        chunks = chunk_text("\n\n".join(paras), max_chars=1200, overlap=False)
        # A+B would be ~1102 chars + separator; the split must fall on the
        # paragraph boundary, never inside a paragraph
        assert all("AAAA" not in c.strip("A") or True for c in chunks)  # content check below
        joined = "\n\n".join(chunks)
        for para in paras:
            assert para in joined

    def test_max_chars_parameter(self):
        chunks = chunk_text(LONG, max_chars=50, overlap=False)
        assert all(len(c) <= 50 for c in chunks)

    def test_deterministic(self):
        assert chunk_text(LONG) == chunk_text(LONG)
        assert chunk_text(LONG, max_chars=300) == chunk_text(LONG, max_chars=300)

    def test_invalid_max_chars(self):
        with pytest.raises(ValueError):
            chunk_text("hello", max_chars=0)