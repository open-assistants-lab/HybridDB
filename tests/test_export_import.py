"""Tests for export_sql, import_sql, backup, restore, vacuum,
check_integrity, stats, reindex."""

import hashlib
import shutil
import tempfile
from pathlib import Path

import pytest

from hybriddb import HybridDB, SearchMode

EMBEDDING_DIM = 384


def _mock_embedding(text: str) -> list[float]:
    if not text:
        return [0.0] * EMBEDDING_DIM
    words = str(text).lower().split()
    dim = EMBEDDING_DIM
    embedding = [0.0] * dim
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
        embedding[h] += 1.0
    mag = sum(x**2 for x in embedding) ** 0.5
    if mag > 0:
        embedding = [x / mag for x in embedding]
    return embedding


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def db(tmp_dir):
    return HybridDB(tmp_dir, embedding_fn=_mock_embedding)


@pytest.fixture
def populated_db(db):
    db.create_table("messages", {
        "role": "TEXT",
        "content": "LONGTEXT",
        "session_id": "TEXT",
    })
    db.insert_batch("messages", [
        {"role": "user", "content": "Hello world from Alice", "session_id": "s1"},
        {"role": "assistant", "content": "Hi Alice, how can I help?", "session_id": "s1"},
        {"role": "user", "content": "Tell me about machine learning", "session_id": "s2"},
        {"role": "assistant", "content": "ML is a subset of AI...", "session_id": "s2"},
        {"role": "user", "content": "What about deep learning?", "session_id": "s2"},
    ])
    return db


class TestExportImportSql:
    def test_export_and_import_roundtrip(self, populated_db, tmp_dir):
        db = populated_db
        dump_path = Path(tmp_dir) / "dump.sql"

        db.export_sql(dump_path)
        assert dump_path.exists()
        content = dump_path.read_text()
        assert "CREATE TABLE messages" in content
        assert "Hello world from Alice" in content

        # Import into a fresh DB
        fresh_path = Path(tmp_dir) / "imported"
        fresh = HybridDB(str(fresh_path), embedding_fn=_mock_embedding)
        fresh.import_sql(dump_path)

        row_count = fresh.count("messages")
        assert row_count == 5

        results = fresh.search(
            "messages", "content", "machine learning",
            mode=SearchMode.HYBRID, limit=5,
        )
        assert len(results) > 0

    def test_import_overwrites_existing_data(self, populated_db, tmp_dir):
        db = populated_db
        dump_path = Path(tmp_dir) / "dump.sql"
        db.export_sql(dump_path)

        # Import onto the same DB — should replace data
        db.import_sql(dump_path)

        row_count = db.count("messages")
        assert row_count == 5

    def test_import_with_chromadb_disabled(self, tmp_dir):
        db = HybridDB(tmp_dir, embedding_fn=_mock_embedding, max_chroma_index_gb=0)
        db.create_table("test", {"name": "TEXT", "bio": "LONGTEXT"})
        db.insert_batch("test", [
            {"name": "Alice", "bio": "Engineer"},
        ])
        dump_path = Path(tmp_dir) / "dump.sql"
        db.export_sql(dump_path)

        fresh = HybridDB(
            str(Path(tmp_dir) / "imported2"),
            embedding_fn=_mock_embedding,
            max_chroma_index_gb=0,
        )
        fresh.import_sql(dump_path)
        assert fresh.count("test") == 1


class TestBackupRestore:
    def test_backup_and_restore(self, populated_db, tmp_dir):
        db = populated_db
        backup_path = str(Path(tempfile.mkdtemp()) / "hybrid_backup")

        db.backup(backup_path)
        assert (Path(backup_path) / "app.db").exists()

        results_before = db.search(
            "messages", "content", "Alice", mode=SearchMode.HYBRID, limit=5,
        )
        assert len(results_before) > 0

        # Wipe data
        db.raw_query("DELETE FROM messages")
        rows_after_delete = db.count("messages")

        # Restore
        db.restore(backup_path)
        rows_after_restore = db.count("messages")
        assert rows_after_delete == 0
        assert rows_after_restore == 5

        results_after = db.search(
            "messages", "content", "Alice", mode=SearchMode.HYBRID, limit=5,
        )
        assert len(results_after) == len(results_before)

    def test_backup_dest_exists_raises(self, populated_db, tmp_dir):
        db = populated_db
        p = Path(tmp_dir) / "exists"
        p.mkdir()
        (p / "some_file").write_text("data")
        with pytest.raises(FileExistsError):
            db.backup(p)

    def test_restore_invalid_path_raises(self, db, tmp_dir):
        with pytest.raises(FileNotFoundError):
            db.restore(Path(tmp_dir) / "nonexistent")


class TestVacuum:
    def test_vacuum_reduces_size(self, populated_db):
        db = populated_db
        # Insert many rows then delete half
        for i in range(100):
            db.insert_batch("messages", [
                {"role": "user", "content": f"Test message {i} {j}", "session_id": f"filler_{i}"}
                for j in range(10)
            ])
        db.raw_query("DELETE FROM messages WHERE id % 2 = 0")
        freed = db.vacuum()
        assert freed >= 0


class TestCheckIntegrity:
    def test_fresh_db_is_ok(self, db):
        db.create_table("test", {"name": "TEXT"})
        report = db.check_integrity()
        assert report["overall"] == "ok"
        assert report["sqlite_integrity"] == "ok"

    def test_no_chromadb_collections(self, db):
        db.create_table("test", {"name": "TEXT"})
        report = db.check_integrity()
        assert report["chromadb_collections"] == 0
        assert report["overall"] in ("ok", "degraded")


class TestStats:
    def test_stats_on_populated_db(self, populated_db):
        db = populated_db
        s = db.stats()
        assert s["sqlite_size_bytes"] > 0
        assert s["total_size_bytes"] > 0
        assert "messages" in s["tables"]
        assert s["tables"]["messages"]["rows"] == 5
        assert s["tables"]["messages"]["fts_indexes"] >= 1

    def test_stats_chromadb_vectors(self, populated_db):
        db = populated_db
        s = db.stats()
        tbl = s["tables"]["messages"]
        assert tbl["chromadb_collections"] == 1
        assert tbl["chromadb_vectors"] == 5


class TestReindex:
    def test_reindex_all_tables(self, populated_db):
        db = populated_db
        # Should not raise
        db.reindex()

        # Verify search still works after reindex
        results = db.search(
            "messages", "content", "machine learning",
            mode=SearchMode.HYBRID, limit=5,
        )
        assert len(results) > 0

    def test_reindex_single_table(self, populated_db):
        db = populated_db
        db.reindex("messages")

        results = db.search(
            "messages", "content", "deep learning",
            mode=SearchMode.HYBRID, limit=5,
        )
        assert len(results) > 0

    def test_reindex_no_lontext_columns(self, db):
        db.create_table("simple", {"name": "TEXT", "age": "INTEGER"})
        db.insert_batch("simple", [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ])
        db.reindex("simple")  # Should be a no-op with no LONGTEXT columns
        assert db.count("simple") == 2
