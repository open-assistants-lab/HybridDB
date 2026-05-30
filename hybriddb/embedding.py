"""Embedding helpers for HybridDB."""

import hashlib
import threading
from typing import Any


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
    if not text:
        return [0.0] * EMBEDDING_DIM
    ef = _get_default_ef()
    if ef is not None:
        try:
            result = ef([text])
            if result and len(result) > 0:
                return result[0]
        except Exception:
            pass
    return hash_embedding(text)


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
