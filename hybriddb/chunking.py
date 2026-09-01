"""Dependency-free long-document chunking for HybridDB ingestion.

Long documents should not be embedded as a single vector — one embedding
cannot represent a multi-page text, and retrieval quality degrades. The
documented pattern is **chunks as rows**: split the document with
:func:`chunk_text`, store each chunk in a child table (``doc_id`` /
``chunk_seq`` / ``content`` LONGTEXT), and index the child table. The parent
table keeps the full text as provenance; retrieval searches chunks and joins
back to the parent.

The splitter is deterministic, never breaks mid-sentence or mid-word, and is
lossless: with ``overlap=False`` every original sentence survives exactly
once, in order. Budgets are approximate when ``overlap=True`` (the overlap
prefix may push a chunk slightly over ``max_chars``).
"""

from __future__ import annotations

import re

__all__ = ["chunk_text"]

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(paragraph: str) -> list[str]:
    return [s for s in (part.strip() for part in _SENTENCE_SPLIT_RE.split(paragraph)) if s]


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Last-resort split of an oversize sentence at character boundaries."""
    return [sentence[i:i + max_chars] for i in range(0, len(sentence), max_chars)]


def chunk_text(text: str, max_chars: int = 1200, overlap: bool = True) -> list[str]:
    """Split ``text`` into retrieval-friendly chunks.

    - Splits on paragraph boundaries (``\\n\\n``) first, then sentence
      boundaries (``.``, ``!``, ``?``, newlines) — never mid-sentence or
      mid-word.
    - Merges adjacent pieces until roughly ``max_chars`` characters
      (~300 tokens for MiniLM-class embedding models at the default 1200).
    - A single sentence longer than ``max_chars`` is hard-split at character
      boundaries (last resort).
    - With ``overlap=True``, the final piece of chunk *n* is prepended to
      chunk *n+1* (skipped when chunk *n+1* already starts with it), giving
      boundary sentences context in both chunks.

    Args:
        text: The document text. Empty/whitespace-only input yields ``[]``.
        max_chars: Approximate per-chunk character budget (~300 tokens at
            4 chars/token). Sentences are never broken to satisfy it exactly.
        overlap: Prepend the previous chunk's final sentence to the next
            chunk. Chunks may slightly exceed ``max_chars`` when the
            sentence is long.

    Returns:
        List of chunk strings in document order. ``[]`` for empty input.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not text.strip():
        return []

    paragraphs = [p for p in (pp.strip() for pp in _PARAGRAPH_SPLIT_RE.split(text)) if p]
    if not paragraphs:
        return [text.strip()]

    # Flatten to (paragraph_index, sentence), hard-splitting oversize sentences.
    pieces: list[tuple[int, str]] = []
    for pi, para in enumerate(paragraphs):
        for sent in _split_sentences(para):
            if len(sent) > max_chars:
                pieces.extend((pi, part) for part in _hard_split(sent, max_chars))
            else:
                pieces.append((pi, sent))
    if not pieces:
        return [text.strip()]

    # Pack sentences into chunks by character budget. Sentences within a
    # paragraph join with " "; paragraph changes join with "\n\n".
    chunks: list[str] = []
    cur = ""
    cur_para = pieces[0][0]
    for pi, sent in pieces:
        if cur:
            joiner = " " if cur_para == pi else "\n\n"
            candidate = f"{cur}{joiner}{sent}"
            if len(candidate) > max_chars:
                chunks.append(cur)
                cur, cur_para = sent, pi
                continue
            cur = candidate
        else:
            cur = sent
        cur_para = pi
    if cur:
        chunks.append(cur)

    if overlap and len(chunks) > 1:
        with_overlap = [chunks[0]]
        for prev, nxt in zip(chunks, chunks[1:]):
            last_piece = _split_sentences(prev)[-1] if _split_sentences(prev) else prev
            if nxt.startswith(last_piece):
                with_overlap.append(nxt)  # already starts with it — dedupe
            else:
                with_overlap.append(f"{last_piece} {nxt}")
        chunks = with_overlap
    return chunks