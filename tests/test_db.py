"""Tests for HybridDB: SQLite + FTS5 + ChromaDB with self-healing journal."""

import asyncio
import hashlib
import shutil
import tempfile
from datetime import UTC, datetime, timedelta

import pytest

from hybriddb import HYBRID, KEYWORD, LONGTEXT, TEXT, Column, EmbeddingModelError, HybridDB, SearchMode
from hybriddb.db import _sanitize_fts_query

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
def db_with_contacts(db):
    db.create_table(
        "contacts",
        {
            "first_name": "TEXT",
            "last_name": "TEXT",
            "company": "LONGTEXT",
            "notes": "LONGTEXT",
            "clv": "REAL",
            "is_active": "BOOLEAN",
        },
    )
    return db


class TestSanitizeFtsQuery:
    def test_basic(self):
        assert _sanitize_fts_query("hello world") == "hello OR world"

    def test_special_chars(self):
        result = _sanitize_fts_query("what's the <best>?")
        assert "<" not in result
        assert ">" not in result

    def test_empty(self):
        assert _sanitize_fts_query("") == ""
        assert _sanitize_fts_query("   ") == ""

    def test_punctuation_only(self):
        assert _sanitize_fts_query("!!! ???") == ""


class TestCreateTable:
    def test_basic(self, db):
        db.create_table(
            "items",
            {
                "name": "TEXT",
                "description": "LONGTEXT",
                "count": "INTEGER",
            },
        )
        schema = db.get_schema("items")
        assert "name" in schema
        assert schema["name"] == "TEXT"
        assert schema["description"] == "LONGTEXT"
        assert schema["count"] == "INTEGER"

    def test_list_tables(self, db):
        db.create_table("t1", {"name": "TEXT"})
        db.create_table("t2", {"val": "INTEGER"})
        tables = db.list_tables()
        assert "t1" in tables
        assert "t2" in tables
        assert "_journal" not in tables
        assert "_schema" not in tables

    def test_all_types(self, db):
        db.create_table(
            "all_types",
            {
                "a_text": "TEXT",
                "a_longtext": "LONGTEXT",
                "an_int": "INTEGER",
                "a_real": "REAL",
                "a_bool": "BOOLEAN",
                "a_json": "JSON",
            },
        )
        schema = db.get_schema("all_types")
        assert schema["a_text"] == "TEXT"
        assert schema["a_longtext"] == "LONGTEXT"
        assert schema["an_int"] == "INTEGER"
        assert schema["a_real"] == "REAL"
        assert schema["a_bool"] == "BOOLEAN"
        assert schema["a_json"] == "JSON"

    def test_duplicate_create(self, db):
        db.create_table("dup", {"name": "TEXT"})
        db.create_table("dup", {"name": "TEXT"})
        assert "dup" in db.list_tables()

    def test_rejects_fts_in_name(self, db):
        with pytest.raises(ValueError, match="_fts_"):
            db.create_table("my_fts_table", {"name": "TEXT"})

    def test_rejects_invalid_identifier_names(self, db):
        with pytest.raises(ValueError, match="Invalid identifier"):
            db.create_table("bad-name", {"name": "TEXT"})
        with pytest.raises(ValueError, match="Invalid identifier"):
            db.create_table("items", {"bad-name": "TEXT"})

    def test_accepts_typed_columns_and_constants(self, db):
        db.create_table("typed_docs", {"title": Column(TEXT), "body": LONGTEXT})
        schema = db.get_schema("typed_docs")
        assert schema == {"title": "TEXT", "body": "LONGTEXT"}

    def test_column_migration(self, db):
        db.create_table("mig", {"name": "TEXT"})
        db.create_table("mig", {"name": "TEXT", "notes": "LONGTEXT"})
        schema = db.get_schema("mig")
        assert "name" in schema
        assert "notes" in schema
        assert schema["notes"] == "LONGTEXT"


class TestCRUD:
    def test_insert_and_get(self, db_with_contacts):
        row_id = db_with_contacts.insert(
            "contacts",
            {
                "first_name": "Alice",
                "last_name": "Smith",
                "company": "Acme Corp",
                "notes": "VIP client",
                "clv": 5000.0,
                "is_active": True,
            },
        )
        assert row_id > 0
        row = db_with_contacts.get("contacts", row_id)
        assert row is not None
        assert row["first_name"] == "Alice"
        assert row["clv"] == 5000.0
        assert row["is_active"] == 1

    def test_insert_with_string_pk(self, db):
        db.create_table("customers", {"id": "TEXT PRIMARY KEY", "name": "LONGTEXT"})
        row_id = db.insert("customers", {"id": "cust_1", "name": "Alice"})
        assert row_id == "cust_1"
        row = db.get("customers", "cust_1")
        assert row["name"] == "Alice"

    def test_insert_batch(self, db_with_contacts):
        ids = db_with_contacts.insert_batch(
            "contacts",
            [
                {"first_name": "Alice", "company": "Acme"},
                {"first_name": "Bob", "company": "Beta Corp"},
                {"first_name": "Charlie", "notes": "New hire"},
            ],
        )
        assert len(ids) == 3
        assert all(iid > 0 for iid in ids)

    def test_update(self, db_with_contacts):
        row_id = db_with_contacts.insert(
            "contacts",
            {"first_name": "Alice", "company": "Acme", "clv": 1000.0},
        )
        ok = db_with_contacts.update("contacts", row_id, {"clv": 2000.0, "first_name": "Alice2"})
        assert ok
        row = db_with_contacts.get("contacts", row_id)
        assert row["clv"] == 2000.0
        assert row["first_name"] == "Alice2"

    def test_update_nonexistent(self, db_with_contacts):
        ok = db_with_contacts.update("contacts", 99999, {"first_name": "Nope"})
        assert not ok

    def test_delete(self, db_with_contacts):
        row_id = db_with_contacts.insert("contacts", {"first_name": "ToDelete", "company": "Gone"})
        ok = db_with_contacts.delete("contacts", row_id)
        assert ok
        assert db_with_contacts.get("contacts", row_id) is None

    def test_delete_nonexistent(self, db_with_contacts):
        ok = db_with_contacts.delete("contacts", 99999)
        assert not ok

    def test_query(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"first_name": "Alice", "clv": 1000.0})
        db_with_contacts.insert("contacts", {"first_name": "Bob", "clv": 2000.0})
        results = db_with_contacts.query("contacts", where="clv > 1500.0")
        assert len(results) == 1
        assert results[0]["first_name"] == "Bob"

    def test_count(self, db_with_contacts):
        db_with_contacts.insert_batch(
            "contacts",
            [
                {"first_name": "A", "clv": 100.0},
                {"first_name": "B", "clv": 200.0},
                {"first_name": "C", "clv": 300.0},
            ],
        )
        assert db_with_contacts.count("contacts") == 3
        assert db_with_contacts.count("contacts", where="clv > 150.0") == 2

    def test_raw_query(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"first_name": "Alice"})
        results = db_with_contacts.raw_query("SELECT first_name FROM contacts")
        assert len(results) == 1
        assert results[0]["first_name"] == "Alice"

    def test_read_query_rejects_writes(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"first_name": "Alice"})
        assert db_with_contacts.read_query("SELECT first_name FROM contacts")
        with pytest.raises(ValueError, match="read-only"):
            db_with_contacts.read_query("DELETE FROM contacts")

    def test_public_cursor_context_manager(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"first_name": "Alice"})
        with db_with_contacts.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM contacts")
            assert cur.fetchone()[0] == 1

    def test_vector_upsert(self, db_with_contacts):
        row_id = db_with_contacts.insert("contacts", {"first_name": "Alice", "notes": "hello"})
        ok = db_with_contacts.vector_upsert(
            "contacts_notes", row_id, "updated text",
            _mock_embedding("updated text"),
            {"first_name": "Alice"},
        )
        assert ok

    def test_row_to_metadata(self, db_with_contacts):
        row_id = db_with_contacts.insert("contacts", {"first_name": "Alice", "clv": 100.0})
        row = db_with_contacts.get("contacts", row_id)
        meta = db_with_contacts.row_to_metadata("contacts", row)
        assert "first_name" in meta
        assert meta["first_name"] == "Alice"
        assert "clv" in meta

    def test_insert_with_skip_journal(self, db):
        db.create_table("test", {"name": "TEXT", "bio": "LONGTEXT", "summary": "LONGTEXT"})
        row_id = db.insert("test", {"name": "alice", "bio": "engineer", "summary": "hello"},
                           skip_journal_columns={"summary"})
        assert row_id > 0


class TestInsertSync:
    def test_sync_false_defers_chroma(self, db_with_contacts):
        db_with_contacts.insert(
            "contacts",
            {"first_name": "Alice", "notes": "Important notes here"},
            sync=False,
        )
        assert db_with_contacts._journal_count("contacts") > 0
        db_with_contacts.process_journal()
        assert db_with_contacts._journal_count("contacts") == 0


class TestSchemaOperations:
    def test_add_column(self, db_with_contacts):
        db_with_contacts.add_column("contacts", "region", "TEXT")
        schema = db_with_contacts.get_schema("contacts")
        assert "region" in schema
        assert schema["region"] == "TEXT"

    def test_add_longtext_column(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"first_name": "Alice", "notes": "old notes"})
        db_with_contacts.add_column("contacts", "bio", "LONGTEXT")
        schema = db_with_contacts.get_schema("contacts")
        assert schema["bio"] == "LONGTEXT"

    def test_drop_column(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"first_name": "Alice", "clv": 100.0})
        db_with_contacts.drop_column("contacts", "clv")
        schema = db_with_contacts.get_schema("contacts")
        assert "clv" not in schema
        assert "first_name" in schema

    def test_drop_nonexistent_raises(self, db_with_contacts):
        with pytest.raises(ValueError):
            db_with_contacts.drop_column("contacts", "nonexistent")

    def test_rename_column(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"first_name": "Alice", "clv": 100.0})
        db_with_contacts.rename_column("contacts", "clv", "customer_lifetime_value")
        schema = db_with_contacts.get_schema("contacts")
        assert "clv" not in schema
        assert "customer_lifetime_value" in schema
        assert schema["customer_lifetime_value"] == "REAL"

    def test_rename_nonexistent_raises(self, db_with_contacts):
        with pytest.raises(ValueError):
            db_with_contacts.rename_column("contacts", "nonexistent", "something")


class TestSearch:
    def test_keyword_search(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"first_name": "Alice", "company": "Acme Corp"})
        db_with_contacts.insert("contacts", {"first_name": "Bob", "company": "Beta Inc"})
        results = db_with_contacts.search("contacts", "company", "Acme", mode=SearchMode.KEYWORD)
        assert len(results) >= 1
        found = any(r["company"] == "Acme Corp" for r in results)
        assert found

    def test_search_accepts_string_mode_and_exported_constants(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"company": "Acme Corp", "notes": "rocket launch"})
        assert db_with_contacts.search("contacts", "company", "Acme", mode="keyword")
        assert db_with_contacts.search("contacts", "company", "Acme", mode=KEYWORD)
        assert db_with_contacts.search("contacts", "notes", "rocket", mode=HYBRID)

    def test_search_without_column_searches_all_text_columns(self, db_with_contacts):
        db_with_contacts.insert(
            "contacts",
            {"company": "Acme Corp builds rockets", "notes": "space launch partner"},
        )
        results = db_with_contacts.search("contacts", "rockets", limit=5)
        assert results
        assert results[0]["company"] == "Acme Corp builds rockets"

    def test_search_columns_alias(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"notes": "important aerospace project"})
        assert db_with_contacts.search_columns("contacts", "aerospace")

    def test_semantic_search(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"notes": "key decision maker for enterprise deals"})
        db_with_contacts.insert("contacts", {"notes": "prefers morning meetings and tea"})
        results = db_with_contacts.search(
            "contacts", "notes", "executive choices", mode=SearchMode.SEMANTIC
        )
        assert isinstance(results, list)

    def test_hybrid_search(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"notes": "key decision maker for enterprise deals"})
        db_with_contacts.insert("contacts", {"notes": "handles morning standup meetings"})
        results = db_with_contacts.search("contacts", "notes", "decision", mode=SearchMode.HYBRID)
        assert isinstance(results, list)
        for r in results:
            assert "_score" in r

    def test_search_text_column_keyword_only(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"first_name": "Alice", "last_name": "Smith"})
        results = db_with_contacts.search(
            "contacts", "first_name", "Alice", mode=SearchMode.KEYWORD
        )
        assert len(results) >= 1

    def test_search_text_column_no_semantic(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"first_name": "Alice"})
        results = db_with_contacts.search(
            "contacts", "first_name", "Alice", mode=SearchMode.SEMANTIC
        )
        assert results == []

    def test_search_all(self, db_with_contacts):
        db_with_contacts.insert(
            "contacts",
            {"company": "Acme Corp builds rockets", "notes": "key decision maker for space projects"},
        )
        results = db_with_contacts.search_all("contacts", "rockets")
        assert isinstance(results, list)

    def test_search_with_recency(self, db):
        db.create_table(
            "messages",
            {"role": "TEXT", "content": "LONGTEXT", "ts": "TEXT"},
        )
        old_ts = (datetime.now(UTC) - timedelta(days=90)).isoformat()
        new_ts = datetime.now(UTC).isoformat()
        db.insert("messages", {"role": "user", "content": "old message about python", "ts": old_ts})
        db.insert("messages", {"role": "user", "content": "new message about python", "ts": new_ts})
        results = db.search("messages", "content", "python", recency_weight=0.3, recency_column="ts")
        assert len(results) >= 2
        for r in results:
            assert "_score" in r

    def test_search_empty_query(self, db_with_contacts):
        results = db_with_contacts.search("contacts", "company", "", mode=SearchMode.KEYWORD)
        assert results == []

    def test_search_nonexistent_table(self, db):
        results = db.search("nonexistent", "col", "query")
        assert results == []

    def test_search_rejects_invalid_identifier(self, db):
        with pytest.raises(ValueError, match="Invalid identifier"):
            db.search("bad-name", "content", "query")
        with pytest.raises(ValueError, match="Invalid identifier"):
            db.search("messages", "bad-name", "query")


class TestAsyncApi:
    @pytest.mark.asyncio
    async def test_async_crud_and_search(self, tmp_dir):
        db = HybridDB(tmp_dir, embedding_fn=_mock_embedding)
        await db.acreate_table("messages", {"content": LONGTEXT, "role": TEXT})
        row_id = await db.ainsert("messages", {"content": "async python notes", "role": "user"})
        row = await db.aget("messages", row_id)
        assert row["content"] == "async python notes"

        results = await db.asearch("messages", "content", "python", mode="hybrid")
        assert results
        assert await db.acount("messages") == 1
        assert await db.aread_query("SELECT role FROM messages") == [{"role": "user"}]

    @pytest.mark.asyncio
    async def test_async_concurrent_inserts_are_serialized_safely(self, tmp_dir):
        db = HybridDB(tmp_dir, embedding_fn=_mock_embedding)
        await db.acreate_table("events", {"content": LONGTEXT})

        async def insert_one(i: int) -> int | str:
            return await db.ainsert("events", {"content": f"event {i}"})

        ids = await asyncio.gather(*(insert_one(i) for i in range(10)))
        assert len(set(ids)) == 10
        assert await db.acount("events") == 10


class TestFacades:
    def test_graph_facade_delegates_to_graph_methods(self, db):
        node_id = db.graph.add_node("Alice", type="person")
        assert db.graph.get_node(node_id)["label"] == "Alice"

    def test_graph_edge_type_and_get_neighbors_aliases(self, db):
        db.add_node("a", label="A")
        db.add_node("b", label="B")
        db.add_edge("e1", "a", "b", edge_type="knows")
        neighbors = db.get_neighbors("a")
        assert neighbors[0]["node"]["id"] == "b"
        assert neighbors[0]["edge"]["type"] == "knows"

    def test_olap_facade_delegates_to_analytics(self, db):
        db.create_table("metrics", {"value": "INTEGER"})
        db.insert_batch("metrics", [{"value": 1}, {"value": 2}], sync=False)
        rows = db.olap.query("SELECT SUM(value) AS total FROM metrics")
        assert rows[0]["total"] == 3

    def test_search_with_query_embedding(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"notes": "python programming"})
        emb = _mock_embedding("python")
        results = db_with_contacts.search(
            "contacts", "notes", "python", query_embedding=emb
        )
        assert isinstance(results, list)


class TestJournal:
    def test_journal_auto_processes_on_insert(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"first_name": "Alice", "notes": "test"})
        assert db_with_contacts._journal_count("contacts") == 0

    def test_journal_deferred_with_sync_false(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"first_name": "Alice", "notes": "test"}, sync=False)
        count = db_with_contacts._journal_count("contacts")
        assert count > 0

    def test_process_journal(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"first_name": "Alice", "notes": "test"}, sync=False)
        processed = db_with_contacts.process_journal()
        assert processed > 0
        assert db_with_contacts._journal_count("contacts") == 0

    def test_journal_status(self, db_with_contacts):
        status = db_with_contacts.journal_status()
        assert "pending" in status
        assert "failed" in status


class TestHealthAndReconcile:
    def test_health_ok(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"first_name": "Alice", "notes": "test"})
        h = db_with_contacts.health("contacts")
        assert h["status"] == "ok"
        assert h["sqlite_rows"] == 1
        assert "contacts_notes" in h["chroma_docs"]

    def test_health_drift_with_pending(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"first_name": "Alice", "notes": "test"}, sync=False)
        h = db_with_contacts.health("contacts")
        assert h["status"] in ("ok", "drift")

    def test_reconcile(self, db_with_contacts):
        db_with_contacts.insert("contacts", {"first_name": "Alice", "notes": "test"})
        result = db_with_contacts.reconcile("contacts")
        assert "ghosts_deleted" in result
        assert "missing_added" in result
        assert "metadata_updated" in result


class TestChromaHealth:
    def test_force_rebuild_unavailable(self, tmp_dir):
        db = HybridDB(tmp_dir, embedding_fn=_mock_embedding, max_chroma_index_gb=0)
        result = db.force_rebuild_chroma_index()
        assert result["status"] == "unavailable"

    def test_index_health_no_vector_dir(self, tmp_dir):
        db = HybridDB(tmp_dir, embedding_fn=_mock_embedding)
        db._check_index_health()
        # Should not raise


class TestMetadataStrategy:
    def test_text_in_metadata(self, db_with_contacts):
        row_id = db_with_contacts.insert(
            "contacts", {"first_name": "Alice", "company": "Acme", "notes": "VIP"},
        )
        row = db_with_contacts.get("contacts", row_id)
        meta = db_with_contacts._row_to_metadata("contacts", row)
        assert "first_name" in meta
        assert meta["first_name"] == "Alice"

    def test_longtext_not_in_metadata(self, db_with_contacts):
        row_id = db_with_contacts.insert(
            "contacts", {"first_name": "Alice", "company": "Acme", "notes": "VIP"},
        )
        row = db_with_contacts.get("contacts", row_id)
        meta = db_with_contacts._row_to_metadata("contacts", row)
        assert "company" not in meta
        assert "notes" not in meta

    def test_json_not_in_metadata(self, db):
        db.create_table("items", {"name": "TEXT", "meta": "JSON", "desc": "LONGTEXT"})
        row_id = db.insert("items", {"name": "test", "meta": '{"k":"v"}', "desc": "description"})
        row = db.get("items", row_id)
        meta = db._row_to_metadata("items", row)
        assert "meta" not in meta

    def test_null_omitted_from_metadata(self, db_with_contacts):
        row_id = db_with_contacts.insert("contacts", {"first_name": "Alice", "clv": None})
        row = db_with_contacts.get("contacts", row_id)
        meta = db_with_contacts._row_to_metadata("contacts", row)
        assert "clv" not in meta

    def test_boolean_in_metadata(self, db_with_contacts):
        row_id = db_with_contacts.insert("contacts", {"first_name": "Alice", "is_active": True})
        row = db_with_contacts.get("contacts", row_id)
        meta = db_with_contacts._row_to_metadata("contacts", row)
        assert "is_active" in meta
        assert meta["is_active"] is True


class TestConversationSchema:
    def test_messages_table(self, db):
        db.create_table(
            "messages",
            {"ts": "TEXT NOT NULL", "role": "TEXT NOT NULL", "content": "LONGTEXT", "metadata": "JSON"},
        )
        schema = db.get_schema("messages")
        assert schema["content"] == "LONGTEXT"
        assert schema["metadata"] == "JSON"
        row_id = db.insert("messages", {"ts": "2026-01-01T00:00:00Z", "role": "user", "content": "Hello world"})
        row = db.get("messages", row_id)
        assert row["role"] == "user"
        results = db.search("messages", "content", "Hello", mode=SearchMode.KEYWORD)
        assert len(results) >= 1


class TestMemorySchema:
    def test_memories_table(self, db):
        db.create_table(
            "memories",
            {
                "id": "TEXT PRIMARY KEY",
                "trigger": "LONGTEXT",
                "action": "LONGTEXT",
                "confidence": "REAL",
                "domain": "TEXT",
                "structured_data": "LONGTEXT",
                "linked_to": "JSON",
            },
        )
        schema = db.get_schema("memories")
        assert schema["trigger"] == "LONGTEXT"
        assert schema["action"] == "LONGTEXT"
        assert schema["linked_to"] == "JSON"
        row_id = db.insert(
            "memories",
            {
                "id": "abc123", "trigger": "user likes python", "action": "suggest python tools",
                "confidence": 0.8, "domain": "preferences", "structured_data": "{}",
            },
        )
        row = db.get("memories", row_id)
        assert row["domain"] == "preferences"

    def test_insights_table(self, db):
        db.create_table(
            "insights",
            {"id": "TEXT PRIMARY KEY", "summary": "LONGTEXT", "domain": "TEXT", "confidence": "REAL"},
        )
        db.insert("insights", {"id": "i1", "summary": "User prefers morning meetings", "domain": "preferences", "confidence": 0.5})
        results = db.search("insights", "summary", "morning meetings")
        assert len(results) >= 1


class TestAutoIncrement:
    def test_ids_not_reused(self, db_with_contacts):
        id1 = db_with_contacts.insert("contacts", {"first_name": "A"})
        id2 = db_with_contacts.insert("contacts", {"first_name": "B"})
        db_with_contacts.delete("contacts", id1)
        id3 = db_with_contacts.insert("contacts", {"first_name": "C"})
        assert id3 > id1
        assert id3 > id2


class TestGraphNodes:
    def test_add_and_get_node(self, db):
        nid = db.add_node("n1", label="Alice", type="person", domain="users")
        assert nid == "n1"
        node = db.get_node("n1")
        assert node["label"] == "Alice"
        assert node["type"] == "person"
        assert node["domain"] == "users"

    def test_add_nodes_batch(self, db):
        ids = db.add_nodes([
            {"id": "n1", "label": "Alice", "domain": "users"},
            {"id": "n2", "label": "Bob", "domain": "users"},
        ])
        assert ids == ["n1", "n2"]

    def test_update_node(self, db):
        db.add_node("n1", label="Alice")
        ok = db.update_node("n1", {"label": "Alice Smith", "confidence": 0.9})
        assert ok
        node = db.get_node("n1")
        assert node["label"] == "Alice Smith"
        assert node["confidence"] == 0.9

    def test_update_nonexistent_node(self, db):
        ok = db.update_node("nonexistent", {"label": "Nope"})
        assert not ok

    def test_delete_node(self, db):
        db.add_node("n1", label="ToDelete")
        ok = db.delete_node("n1")
        assert ok
        assert db.get_node("n1") is None

    def test_delete_nonexistent_node(self, db):
        assert not db.delete_node("nonexistent")

    def test_list_nodes(self, db):
        db.add_node("n1", type="person", domain="users")
        db.add_node("n2", type="company", domain="biz")
        nodes = db.list_nodes(type="person")
        assert len(nodes) == 1
        assert nodes[0]["id"] == "n1"

    def test_list_nodes_with_properties(self, db):
        db.add_node("n1", type="person", properties={"age": 30, "city": "SF"})
        node = db.get_node("n1")
        assert node["properties"]["age"] == 30


class TestGraphEdges:
    def test_add_and_get_edge(self, db):
        db.add_node("n1", label="Alice")
        db.add_node("n2", label="Bob")
        eid = db.add_edge(None, "n1", "n2", type="knows", weight=0.8)
        assert eid is not None
        edge = db.get_edge(eid)
        assert edge["source_id"] == "n1"
        assert edge["target_id"] == "n2"
        assert edge["type"] == "knows"

    def test_add_edges_batch(self, db):
        db.add_node("n1"), db.add_node("n2"), db.add_node("n3")
        ids = db.add_edges([
            {"source_id": "n1", "target_id": "n2", "type": "knows"},
            {"source_id": "n1", "target_id": "n3", "type": "knows"},
        ])
        assert len(ids) == 2

    def test_update_edge(self, db):
        db.add_node("n1"), db.add_node("n2")
        eid = db.add_edge(None, "n1", "n2", weight=0.5)
        ok = db.update_edge(eid, {"weight": 0.9})
        assert ok
        edge = db.get_edge(eid)
        assert edge["weight"] == 0.9

    def test_delete_edge(self, db):
        db.add_node("n1"), db.add_node("n2")
        eid = db.add_edge(None, "n1", "n2")
        ok = db.delete_edge(eid)
        assert ok
        assert db.get_edge(eid) is None

    def test_get_edges_filtered(self, db):
        db.add_node("n1"), db.add_node("n2"), db.add_node("n3")
        db.add_edge(None, "n1", "n2", type="knows")
        db.add_edge(None, "n1", "n3", type="reports_to")
        edges = db.get_edges(source_id="n1")
        assert len(edges) == 2
        edges = db.get_edges(source_id="n1", type="knows")
        assert len(edges) == 1

    def test_neighbors(self, db):
        db.add_node("n1"), db.add_node("n2"), db.add_node("n3")
        db.add_edge(None, "n1", "n2", type="knows")
        db.add_edge(None, "n1", "n3", type="knows")
        neigh = db.neighbors("n1")
        assert len(neigh) == 2
        nids = {n["node"]["id"] for n in neigh}
        assert "n2" in nids
        assert "n3" in nids

    def test_traverse(self, db):
        nodes = ["a", "b", "c", "d"]
        for n in nodes:
            db.add_node(n)
        db.add_edge(None, "a", "b", weight=0.9)
        db.add_edge(None, "b", "c", weight=0.8)
        db.add_edge(None, "c", "d", weight=0.7)
        path = db.traverse("a", max_depth=3, direction="out")
        assert len(path) >= 3

    def test_decay_edges(self, db):
        db.add_node("n1"), db.add_node("n2")
        past = (datetime.now(UTC) - timedelta(days=400)).isoformat()
        eid = db.add_edge(None, "n1", "n2", valid_until=past, weight=0.1)
        dec = db.decay_edges()
        assert dec == 1
        # weight 0.1 - 0.15 = -0.05, clamped to 0.05 by max(), 0.05 <= 0.05 → delete
        assert db.get_edge(eid) is None

    def test_register_entity_node(self, db):
        db.create_table("users", {"id": "TEXT PRIMARY KEY", "name": "LONGTEXT"})
        ok = db.register_entity_node("users", type="user", id_column="id")
        assert ok

    def test_register_edge_rule(self, db):
        ok = db.register_edge_rule("orders", "customers", target_match="customer_id", edge_type="belongs_to")
        assert ok

    def test_register_edge_rule_invalid(self, db):
        with pytest.raises(ValueError, match="must be provided together"):
            db.register_edge_rule("a", "b", source_column="x")

    def test_register_edge_rule_no_match(self, db):
        with pytest.raises(ValueError, match="required"):
            db.register_edge_rule("a", "b")


class TestGraphAlgorithms:
    def test_to_networkx(self, db):
        db.add_node("n1"), db.add_node("n2")
        db.add_edge(None, "n1", "n2", weight=0.8)
        g = db.to_networkx()
        assert g.has_node("n1")
        assert g.has_node("n2")
        assert g.has_edge("n1", "n2")
        assert g.edges["n1", "n2"]["weight"] == 0.8

    def test_to_networkx_cache(self, db):
        db.add_node("n1"), db.add_node("n2")
        db.add_edge(None, "n1", "n2")
        g1 = db.to_networkx()
        g2 = db.to_networkx()
        assert g1 is g2  # cached

    def test_to_networkx_invalidates(self, db):
        db.add_node("n1"), db.add_node("n2")
        db.add_edge(None, "n1", "n2")
        g1 = db.to_networkx()
        db.add_node("n3")
        g2 = db.to_networkx()
        assert g1 is not g2  # invalidated

    def test_pagerank(self, db):
        db.add_node("n1"), db.add_node("n2"), db.add_node("n3")
        db.add_edge(None, "n1", "n2", weight=0.9)
        db.add_edge(None, "n1", "n3", weight=0.8)
        pr = db.pagerank()
        assert "n1" in pr

    def test_shortest_path(self, db):
        db.add_node("a"), db.add_node("b"), db.add_node("c")
        db.add_edge(None, "a", "b", weight=0.9)
        db.add_edge(None, "b", "c", weight=0.8)
        path = db.shortest_path("a", "c")
        assert path == ["a", "b", "c"]

    def test_connected_components(self, db):
        db.add_node("a"), db.add_node("b")
        db.add_edge(None, "a", "b")
        db.add_node("c")
        comps = db.connected_components()
        assert len(comps) <= 2

    def test_community_detect(self, db):
        for n in ["a", "b", "c"]:
            db.add_node(n)
        db.add_edge(None, "a", "b", weight=1.0)
        db.add_edge(None, "b", "c", weight=1.0)
        comms = db.community_detect()
        assert isinstance(comms, list)

    def test_search_graph(self, db):
        db.create_table("items", {"id": "TEXT PRIMARY KEY", "name": "LONGTEXT"})
        db.register_entity_node("items", type="item", id_column="id")
        db.insert("items", {"id": "item_1", "name": "gadget"})
        results = db.search_graph("gadget")
        assert isinstance(results, list)


class TestDuckDBAnalytics:
    def test_analytics(self, db):
        db.create_table("analytics_test", {"val": "INTEGER"})
        db.insert("analytics_test", {"val": 42})
        db.register_duckdb_table("analytics_test")
        result = db.analytics("SELECT val + 1 AS result FROM analytics_test")
        assert result == [{"result": 43}]

    def test_register_duckdb_table(self, db):
        db.create_table("analytics_test", {"val": "INTEGER", "name": "LONGTEXT"})
        db.insert("analytics_test", {"val": 42, "name": "test"})
        ok = db.register_duckdb_table("analytics_test")
        assert ok
        result = db.analytics("SELECT COUNT(*) AS cnt FROM analytics_test")
        assert result[0]["cnt"] >= 1

    def test_unregister_duckdb_table(self, db):
        db.create_table("analytics_test", {"val": "INTEGER"})
        db.register_duckdb_table("analytics_test")
        ok = db.unregister_duckdb_table("analytics_test")
        assert ok
        assert not db.unregister_duckdb_table("analytics_test")


class TestEmbeddingModelError:
    def test_model_mismatch(self, tmp_dir):
        db1 = HybridDB(tmp_dir, embedding_fn=_mock_embedding, embedding_model_name="model_a")
        db1.create_table("test", {"name": "LONGTEXT"})
        db1.close()
        with pytest.raises(EmbeddingModelError, match="model mismatch"):
            HybridDB(tmp_dir, embedding_fn=_mock_embedding, embedding_model_name="model_b")

    def test_force_override(self, tmp_dir):
        db1 = HybridDB(tmp_dir, embedding_fn=_mock_embedding, embedding_model_name="model_a")
        db1.create_table("test", {"name": "LONGTEXT"})
        db1.close()
        db2 = HybridDB(tmp_dir, embedding_fn=_mock_embedding, embedding_model_name="model_b", force_model=True)
        assert db2 is not None
        db2.close()


class TestDisableChroma:
    def test_init_with_zero_max_chroma(self, tmp_dir):
        db = HybridDB(tmp_dir, embedding_fn=_mock_embedding, max_chroma_index_gb=0)
        assert db._chroma is None
        db.create_table("test", {"name": "TEXT"})
        db.insert("test", {"name": "hello"})
        results = db.search("test", "name", "hello", mode=SearchMode.KEYWORD)
        assert len(results) >= 1
