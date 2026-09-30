"""Bundled MiniLM-L6-v2 (uint8) embedding engine for HybridDB.

This is the default embedding engine: it requires no network access, produces
dense 384-dimensional vectors, and is the same `all-MiniLM-L6-v2` model Chroma
bundles — quantized per-channel to uint8 (22.9 MB vs 90.4 MB fp32) with a
measured retrieval delta well under BEIR tolerance:

    NFCorpus  semantic nDCG@10 0.3149 -> 0.3111 (Δ -0.0038)
              hybrid   nDCG@10 0.3429 -> 0.3405 (Δ -0.0024)
    SciFact   semantic nDCG@10 0.6451 -> 0.6445 (Δ -0.0006)
              hybrid   nDCG@10 0.7022 -> 0.7053 (Δ +0.0031)

Why a bundled model instead of Chroma's runtime download: the previous default
silently substituted a hash embedding (`hash_embedding`) when the ONNX model
could not be fetched. That path measured a 5.3× accuracy cliff on BEIR
(nDCG 0.059 vs 0.343) and, because word-hash vectors collide terms into very
few dimensions, it is the exact construct that provokes a hard SIGSEGV inside
chromadb 1.5.9's native Rust bindings (issue #5). This engine removes both.

Vendored from chromadb's `onnx_mini_lm_l6_v2` (Apache-2.0) with two deliberate
differences, both verified to produce identical vectors:

- explicit ``CPUExecutionProvider`` (Chroma defaults to *all* available
  providers, which on macOS starts with CoreML and is ~6× slower per row);
- padding is sized to the batch rather than forced to 256 tokens. Padding is
  attention-masked out of mean pooling, so vectors are unchanged while short
  documents cost proportionally less.

Attribution: all-MiniLM-L6-v2 © UKPLab sentence-transformers, Apache-2.0.
ONNX conversion © chroma-core/onnx-embedding, Apache-2.0.
See models/all-MiniLM-L6-v2-uint8/NOTICE.md for provenance and the SHA-256 of
the fp32 archive this bundle was quantized from.
"""

from __future__ import annotations

import importlib.resources
import os
from typing import Any

import numpy as np

from hybriddb.types import EmbeddingModelError

EMBEDDING_DIM = 384
MAX_TOKENS = 256

BUNDLED_MODEL_LABEL = "hybriddb:all-MiniLM-L6-v2-uint8"
# Chroma's runtime-downloaded fp32 engine produces the same semantic space;
# stores carrying that label are accepted without force_model (below).
CHROMA_MODEL_LABEL = "chroma:all-MiniLM-L6-v2"

# Labels that describe implementations of the same model in the same vector
# space. The mismatch check treats them as one model: quantization changes
# vector components by at most ~0.03 (measured max abs), never the space.
EQUIVALENT_MODEL_LABELS = {BUNDLED_MODEL_LABEL, CHROMA_MODEL_LABEL}

_ENGINE_CACHE: dict[str, Any] = {}


def bundled_model_dir() -> str:
    """Filesystem path of the bundled model, extracted if packaged as zip."""
    root = importlib.resources.files("hybriddb") / "models" / "all-MiniLM-L6-v2-uint8"
    # as_file is needed when the wheel is a zip (zipimport) — on-disk installs
    # resolve to the real directory.
    # as_file() is needed when the wheel is a zip (zipimport). On-disk
    # installs resolve to the real directory path.
    import contextlib
    with contextlib.suppress(Exception):
        cm = importlib.resources.as_file(root)
        if hasattr(cm, "__enter__"):
            return str(cm.__enter__())
    return str(root)


class BundledMiniLM:
    """Load the bundled ONNX MiniLM and produce dense 384-d vectors.

    Deliberately no network access: the model ships inside the package, and a
    missing/failed load raises instead of degrading to a lossy fallback.
    """

    def __init__(self, model_dir: str | None = None):
        import onnxruntime as ort  # already a chromadb core dependency
        from tokenizers import Tokenizer

        self._dir = model_dir or bundled_model_dir()
        onnx_path = os.path.join(self._dir, "onnx", "model.onnx")
        if not os.path.exists(onnx_path):
            raise EmbeddingModelError(
                f"Bundled MiniLM model not found at {onnx_path} — the package "
                "appears to be incomplete. Reinstall hybriddb, or pass an "
                "embedding_fn= to HybridDB."
            )
        self._tokenizer = Tokenizer.from_file(os.path.join(self._dir, "onnx", "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")  # dynamic
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )

    def encode(self, text: str) -> list[float]:
        """Embed one document. Text longer than 256 tokens is truncated."""
        enc = self._tokenizer.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        out = self._sess.run(None, {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": np.zeros_like(ids),
        })[0]
        expanded = np.broadcast_to(np.expand_dims(mask, -1), out.shape)
        pooled = (out * expanded).sum(1) / np.clip(expanded.sum(1), 1e-9, None)
        v = pooled[0].astype(np.float32)
        norm = np.linalg.norm(v)
        return (v / (norm if norm > 0 else 1e-12)).tolist()


def default_engine() -> tuple[Any, str]:
    """Return ``(embedding_fn, model_label)`` for HybridDB's default path.

    Order: bundled uint8 model (offline, always present), then Chroma's runtime
    fp32 download only when the bundled engine cannot load. Neither works ->
    EmbeddingModelError. The old silent hash fallback is gone.
    """
    if "engine" not in _ENGINE_CACHE:
        bundled_error: Exception | None = None
        try:
            _ENGINE_CACHE["engine"] = (BundledMiniLM().encode, BUNDLED_MODEL_LABEL)
        except Exception as exc:  # noqa: BLE001 - loud, not silent
            bundled_error = exc
            try:
                from chromadb.utils.embedding_functions import (
                    DefaultEmbeddingFunction,
                )
                chroma_ef = DefaultEmbeddingFunction()
                _ENGINE_CACHE["engine"] = (
                    lambda text: chroma_ef([text])[0],
                    CHROMA_MODEL_LABEL,
                )
            except Exception as chroma_error:  # noqa: BLE001 - loud, not silent
                raise EmbeddingModelError(
                    "No usable default embedding engine: the bundled MiniLM "
                    f"could not load ({bundled_error}) and Chroma's fp32 model "
                    f"is also unavailable ({chroma_error}; it needs network on "
                    "first use). Install hybriddb with onnxruntime/tokenizers "
                    "available, or pass embedding_fn= to HybridDB. The lossy "
                    "hash fallback is no longer used — it measured a 5.3x "
                    "accuracy cliff on BEIR and provokes a chromadb segfault "
                    "(issue #5)."
                ) from chroma_error
    return _ENGINE_CACHE["engine"]