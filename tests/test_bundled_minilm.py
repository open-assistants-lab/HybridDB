"""Bundled MiniLM default engine (option D).

The default embedding engine ships a per-channel uint8 MiniLM inside the
package. It replaces the silent hash fallback, which measured a 5.3x accuracy
cliff on BEIR and was the exact vector construct that provokes the chromadb
1.5.9 segfault (issue #5).
"""

import shutil
import tempfile

import pytest

import hybriddb
from hybriddb import LONGTEXT, HybridDB
from hybriddb.embedding import (
    default_embedding_fn,
    default_model_label,
    hash_embedding,
)
from hybriddb.embedding_local import BUNDLED_MODEL_LABEL, CHROMA_MODEL_LABEL


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _density(vec) -> int:
    seq = vec if isinstance(vec, list) else list(vec)
    return sum(1 for x in seq if x != 0.0)


class TestBundledEngineDefaults:
    def test_default_engine_is_bundled_and_dense(self):
        assert default_model_label() == BUNDLED_MODEL_LABEL
        vec = default_embedding_fn("hello world memory entry")
        assert len(vec) == 384
        assert _density(vec) == 384, "bundled MiniLM must be dense"

    def test_default_is_not_hash_embedding(self, tmp_dir):
        db = HybridDB(tmp_dir)  # no embedding_fn -> default engine
        db.create_table("t", {"id": "TEXT PRIMARY KEY", "body": LONGTEXT})
        db.insert("t", {"id": "a", "body": "hello world"})
        assert db._embedding_model_name == BUNDLED_MODEL_LABEL
        assert default_embedding_fn("alpha") != hash_embedding("alpha")
        db.close()

    def test_schema_records_the_actual_engine(self, tmp_dir):
        db = HybridDB(tmp_dir)
        db.create_table("t", {"body": LONGTEXT})
        with db._connect() as cur:
            row = cur.execute("SELECT embedding_model FROM _schema").fetchone()
        assert row["embedding_model"] == BUNDLED_MODEL_LABEL
        db.close()

    def test_long_text_is_truncated_not_raised(self, tmp_dir):
        db = HybridDB(tmp_dir)
        db.create_table("t", {"id": "TEXT PRIMARY KEY", "body": LONGTEXT})
        long_text = "word " * 700  # ~700 tokens, above the 256-token limit
        db.insert("t", {"id": "long", "body": long_text})
        assert db.count("t") == 1
        db.close()

    def test_deterministic(self):
        assert default_embedding_fn("same text") == default_embedding_fn("same text")


class TestEngineFallbackIsLoud:
    def test_no_silent_hash_fallback(self, monkeypatch):
        """Neither engine available -> EmbeddingModelError, never hash."""
        import chromadb.utils.embedding_functions as chroma_ef_mod
        import hybriddb.embedding_local as engine_mod

        class BrokenEngine:
            def __init__(self):
                raise RuntimeError("bundled model unavailable (test)")

        monkeypatch.setattr(engine_mod, "BundledMiniLM", BrokenEngine)
        monkeypatch.setattr(
            chroma_ef_mod, "DefaultEmbeddingFunction",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("chroma unavailable (test)")),
        )
        engine_mod._ENGINE_CACHE.clear()
        try:
            with pytest.raises(hybriddb.EmbeddingModelError, match="No usable default embedding engine"):
                default_embedding_fn("anything")
        finally:
            monkeypatch.undo()
            engine_mod._ENGINE_CACHE.clear()
        # The cached engine must be restored for the other tests.
        assert default_model_label() == BUNDLED_MODEL_LABEL

    def test_hash_embedding_still_importable_but_off_default_path(self):
        """Kept for the BEIR comparison harness; must never be chosen by default."""
        assert hash_embedding("x")[:2] == [0.0, 0.0]
        assert default_model_label() != hash_embedding("x")


class TestStoreCompatibility:
    def test_store_labelled_with_chroma_engine_opens_without_force(self, tmp_dir):
        """Pre-bundling stores recorded chroma's fp32 label; the bundled engine
        is the same model in the same space, so they keep opening."""
        db = HybridDB(tmp_dir, embedding_model_name=CHROMA_MODEL_LABEL)
        db.create_table("t", {"body": LONGTEXT})
        db.insert("t", {"body": "hello world"})
        db.close()

        reopened = HybridDB(tmp_dir)  # bundled engine, no force_model
        assert reopened.count("t") == 1
        reopened.close()

    def test_genuinely_different_model_still_requires_force(self, tmp_dir):
        from hybriddb.types import EmbeddingModelError

        db = HybridDB(tmp_dir, embedding_fn=hash_embedding,
                      embedding_model_name="someone-elses-model")
        db.create_table("t", {"body": LONGTEXT})
        db.close()

        with pytest.raises(EmbeddingModelError, match="mismatch"):
            HybridDB(tmp_dir)  # bundled engine vs foreign label

        forced = HybridDB(tmp_dir, force_model=True)
        forced.reindex("t")
        forced.close()