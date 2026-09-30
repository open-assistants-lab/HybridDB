"""Embedding helpers for HybridDB."""

import hashlib
import threading
from typing import Any

from hybriddb.embedding_local import BUNDLED_MODEL_LABEL, default_engine
from hybriddb.types import EmbeddingModelError

EMBEDDING_DIM = 384

_default_ef: Any | None = None
_default_ef_lock = threading.Lock()


def _get_default_ef():
    global _default_ef
    if _default_ef is not None:
        return _default_ef
    with _default_ef_lock:
        if _default_ef is not None:
            return _default_ef
        try:
            from chromadb.utils import embedding_functions

            _default_ef = embedding_functions.DefaultEmbeddingFunction()
        except Exception:
            _default_ef = None
    return _default_ef


def default_embedding_fn(text: str) -> list[float]:
    """Default embedding engine for HybridDB — the bundled uint8 MiniLM.

    Requires no network. Raises :class:`EmbeddingModelError` when neither the
    bundled model nor Chroma's fp32 runtime model can be used; the previous
    behaviour was to silently substitute ``hash_embedding``, which measured a
    5.3× accuracy cliff on BEIR (nDCG 0.059 vs 0.343) and provokes a hard
    SIGSEGV in chromadb 1.5.9's native bindings (issue #5).
    """
    engine, _ = default_engine()
    return engine(text)


def default_model_label() -> str:
    """Label the default engine records in ``_schema`` for its actual vectors."""
    _, label = default_engine()
    return label


def default_engine_is_bundled() -> bool:
    """Whether the default engine resolved to the bundled offline model.

    Hash embedding is no longer on the default path: its word-hash vectors are
    the exact construct that measures the 5.3x BEIR cliff and provokes the
    chromadb segfault in #5.
    """
    try:
        return default_model_label() == BUNDLED_MODEL_LABEL
    except EmbeddingModelError:
        return False


def hash_embedding(text: str) -> list[float]:
    if not text:
        return [0.0] * EMBEDDING_DIM
    words = str(text).lower().split()
    dim = EMBEDDING_DIM
    embedding = [0.0] * dim
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
        embedding[h] += 1.0
    mag = sum(x ** 2 for x in embedding) ** 0.5
    if mag > 0:
        embedding = [x / mag for x in embedding]
    return embedding


_default_embedding_fn = default_embedding_fn
_hash_embedding = hash_embedding
_default_embedding_fn = default_embedding_fn
_hash_embedding = hash_embedding
