"""Regression tests for bugs found in the 2026-08 bug hunt.

Covers:
- C1: DuckDB analytics broken / __init__ crash for custom-PK tables
- C2: journal wedge (DuplicateIDError) on TEXT-PK delete+reinsert
- C3: reindex/drop_column/rename_column/import_sql empty the FTS index
- C4: add_node/add_nodes on existing id deletes its edges (FK cascade)
- H5: reconcile() silently fails on custom-PK tables (hardcoded `id`)
- H6: update() PK change crashes; insert() into TEXT-PK without key crashes
- H7: GraphAPI facade missing search_graph_ppr / sync_graph_nodes
- H8: hybriddb.__version__ stale
"""

import hashlib
import importlib.metadata
import shutil
import sqlite3
import tempfile
from pathlib import Path

import pytest

import hybriddb
from hybriddb import LONGTEXT, TEXT, HybridDB

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
def db_no_chroma(tmp_dir):
    return HybridDB(tmp_dir, embedding_fn=_mock_embedding, max_chroma_index_gb=0)


# ── C1: DuckDB analytics for custom-PK tables ─────────────────────────────

class TestDuckDBCustomPk:
    def test_text_pk_reopen_and_sync(self, tmp_dir):
        pytest.importorskip("duckdb")
        path = tmp_dir
        db = HybridDB(path, embedding_fn=_mock_embedding, max_chroma_index_gb=0)
        db.create_table("items", {"uid": "TEXT PRIMARY KEY", "name": TEXT})
        db.insert("items", {"uid": "a1", "name": "first"})
        db.insert("items", {"uid": "a2", "name": "second"})
        db.close()

        # reopening used to raise BinderException in __init__
        db = HybridDB(path, embedding_fn=_mock_embedding, max_chroma_index_gb=0)
        rows = db.olap.query("SELECT uid, name FROM items ORDER BY uid")
        assert [(r["uid"], r["name"]) for r in rows] == [("a1", "first"), ("a2", "second")]

        # incremental journal sync (insert / update / delete)
        db.insert("items", {"uid": "a3", "name": "third"})
        assert db.olap.query("SELECT count(*) c FROM items")[0]["c"] == 3
        db.update("items", "a1", {"name": "first!"})
        assert db.olap.query("SELECT name FROM items WHERE uid = 'a1'")[0]["name"] == "first!"
        db.delete("items", "a2")
        assert db.olap.query("SELECT count(*) c FROM items")[0]["c"] == 2
        db.close()

    def test_custom_integer_pk_reopen(self, tmp_dir):
        pytest.importorskip("duckdb")
        path = str(tmp_dir)
        db = HybridDB(path, embedding_fn=_mock_embedding, max_chroma_index_gb=0)
        db.create_table("items", {"uid": "INTEGER PRIMARY KEY", "name": TEXT})
        db.insert("items", {"uid": 10, "name": "ten"})
        db.close()

        db = HybridDB(path, embedding_fn=_mock_embedding, max_chroma_index_gb=0)
        rows = db.olap.query("SELECT uid, name FROM items")
        assert [(r["uid"], r["name"]) for r in rows] == [(10, "ten")]
        db.insert("items", {"uid": 20, "name": "twenty"})
        assert db.olap.query("SELECT count(*) c FROM items")[0]["c"] == 2
        db.close()

    def test_default_pk_still_syncs(self, tmp_dir):
        pytest.importorskip("duckdb")
        db = HybridDB(tmp_dir, embedding_fn=_mock_embedding, max_chroma_index_gb=0)
        db.create_table("docs", {"title": TEXT})
        r1 = db.insert("docs", {"title": "x"})
        db.insert("docs", {"title": "y"})
        db.close()

        db = HybridDB(tmp_dir, embedding_fn=_mock_embedding, max_chroma_index_gb=0)
        assert db.olap.query("SELECT count(*) c FROM docs")[0]["c"] == 2
        db.delete("docs", r1)
        assert db.olap.query("SELECT count(*) c FROM docs")[0]["c"] == 1
        db.close()


# ── C2: journal wedge on TEXT-PK delete + reinsert ────────────────────────

class TestJournalDeleteReinsert:
    def test_text_pk_delete_reinsert_no_wedge(self, db):
        db.create_table("items", {"uid": "TEXT PRIMARY KEY", "name": TEXT, "body": LONGTEXT})
        db.insert("items", {"uid": "a", "name": "n", "body": "widgets and gears"}, sync=False)
        db.delete("items", "a", sync=False)
        db.insert("items", {"uid": "b", "name": "m", "body": "widgets and gears"}, sync=False)

        db.process_journal()  # used to raise DuplicateIDError and wedge forever

        assert db.journal_status()["pending"] == 0
        assert db.count("items") == 1
        # chroma must contain the re-inserted row (rowid was reused)
        assert len(db.search("items", "body", "widgets", mode="semantic")) == 1
        assert len(db.search("items", "body", "widgets", mode="keyword")) == 1
        assert db.health("items")["status"] == "ok"

    def test_ops_still_work_after_batch(self, db):
        db.create_table("items", {"uid": "TEXT PRIMARY KEY", "name": TEXT, "body": LONGTEXT})
        db.insert("items", {"uid": "a", "name": "n", "body": "widgets"}, sync=False)
        db.delete("items", "a", sync=False)
        db.insert("items", {"uid": "b", "name": "m", "body": "widgets"}, sync=False)
        db.process_journal()
        # a normal sync=True op after a wedged batch used to raise
        db.insert("items", {"uid": "c", "name": "o", "body": "more text"})
        assert db.count("items") == 2

    def test_delete_then_update_in_batch(self, db):
        db.create_table("items", {"uid": "TEXT PRIMARY KEY", "name": TEXT, "body": LONGTEXT})
        db.insert("items", {"uid": "a", "name": "n", "body": "v1 widgets"}, sync=False)
        db.update("items", "a", {"body": "v2 widgets"}, sync=False)
        db.delete("items", "a", sync=False)
        db.insert("items", {"uid": "b", "name": "m", "body": "v3 widgets"}, sync=False)
        db.process_journal()
        sem = db.search("items", "body", "widgets", mode="semantic")
        assert len(sem) == 1
        assert sem[0]["body"] == "v3 widgets"


# ── C3: FTS backfill after rebuilds ───────────────────────────────────────

class TestFtsBackfill:
    def _seed(self, db):
        db.create_table("docs", {"title": TEXT, "body": LONGTEXT})
        for i in range(5):
            db.insert("docs", {"title": f"Title {i}", "body": f"body number {i}"})

    def test_reindex_preserves_keyword_search(self, db_no_chroma):
        self._seed(db_no_chroma)
        db_no_chroma.reindex("docs")
        assert len(db_no_chroma.search("docs", "title", "title", mode="keyword")) == 5
        assert len(db_no_chroma.search("docs", "body", "number", mode="keyword")) == 5

    def test_drop_column_preserves_keyword_search(self, db_no_chroma):
        self._seed(db_no_chroma)
        db_no_chroma.add_column("docs", "extra", "TEXT")
        db_no_chroma.drop_column("docs", "extra")
        assert len(db_no_chroma.search("docs", "title", "title", mode="keyword")) == 5

    def test_rename_column_preserves_keyword_search(self, db_no_chroma):
        self._seed(db_no_chroma)
        db_no_chroma.rename_column("docs", "title", "headline")
        assert len(db_no_chroma.search("docs", "headline", "title", mode="keyword")) == 5

    def test_import_sql_preserves_keyword_search(self, db_no_chroma, tmp_dir):
        self._seed(db_no_chroma)
        dump = str(Path(tmp_dir) / "dump.sql")
        db_no_chroma.export_sql(dump)
        db_no_chroma.import_sql(dump)
        assert len(db_no_chroma.search("docs", "title", "title", mode="keyword")) == 5


# ── C4: add_node must not cascade-delete edges ────────────────────────────

class TestGraphNodeUpsert:
    def test_add_node_preserves_edges(self, db_no_chroma):
        db_no_chroma.add_node("a", label="A")
        db_no_chroma.add_node("b", label="B")
        db_no_chroma.add_edge("e1", "a", "b", type="knows")
        db_no_chroma.add_node("a", label="A2")
        edges = db_no_chroma.graph.get_edges()
        assert len(edges) == 1
        assert edges[0]["id"] == "e1"
        assert db_no_chroma.get_node("a")["label"] == "A2"

    def test_add_nodes_batch_preserves_edges(self, db_no_chroma):
        db_no_chroma.add_node("a", label="A")
        db_no_chroma.add_node("b", label="B")
        db_no_chroma.add_edge("e1", "a", "b", type="knows")
        db_no_chroma.add_nodes([{"id": "a", "label": "A2"}, {"id": "c", "label": "C"}])
        assert len(db_no_chroma.graph.get_edges()) == 1


# ── H5: reconcile() on custom-PK tables ───────────────────────────────────

class TestReconcileCustomPk:
    def test_reconcile_restores_custom_pk_table(self, db):
        db.create_table("items", {"uid": "TEXT PRIMARY KEY", "name": TEXT, "body": LONGTEXT})
        db.insert("items", {"uid": "a1", "name": "x", "body": "alpha beta"})
        coll = db._get_collection("items_body")
        coll.delete(ids=[coll.get()["ids"][0]])
        assert coll.count() == 0

        result = db.reconcile("items")
        assert result["missing_added"] >= 1
        assert coll.count() == 1
        assert db.health("items")["status"] == "ok"


# ── H6: clean errors instead of TypeError crashes ─────────────────────────

class TestPkValidation:
    def test_update_changes_custom_pk(self, db_no_chroma):
        db_no_chroma.create_table("items", {"uid": "TEXT PRIMARY KEY", "name": TEXT})
        db_no_chroma.insert("items", {"uid": "a1", "name": "first"})
        assert db_no_chroma.update("items", "a1", {"uid": "a2"}) is True
        assert db_no_chroma.get("items", "a2")["name"] == "first"
        assert db_no_chroma.get("items", "a1") is None

    def test_insert_text_pk_missing_key_raises_clean_error(self, db_no_chroma):
        db_no_chroma.create_table("items", {"uid": "TEXT PRIMARY KEY", "name": TEXT})
        with pytest.raises(ValueError, match="primary key"):
            db_no_chroma.insert("items", {"name": "no key"})

    def test_insert_batch_text_pk_missing_key_raises_clean_error(self, db_no_chroma):
        db_no_chroma.create_table("items", {"uid": "TEXT PRIMARY KEY", "name": TEXT})
        with pytest.raises(ValueError, match="primary key"):
            db_no_chroma.insert_batch("items", [{"name": "no key"}])

    def test_insert_batch_text_pk_works_with_key(self, db_no_chroma):
        db_no_chroma.create_table("items", {"uid": "TEXT PRIMARY KEY", "name": TEXT})
        ids = db_no_chroma.insert_batch("items", [{"uid": "k1", "name": "a"}, {"uid": "k2", "name": "b"}])
        assert ids == ["k1", "k2"]


# ── H7: facade coverage ───────────────────────────────────────────────────

class TestGraphFacade:
    def test_search_graph_ppr_exposed(self, db_no_chroma):
        assert callable(db_no_chroma.graph.search_graph_ppr)

    def test_sync_graph_nodes_exposed(self, db_no_chroma):
        assert callable(db_no_chroma.graph.sync_graph_nodes)


# ── H8: version consistency ───────────────────────────────────────────────

class TestVersion:
    def test_version_matches_package_metadata(self):
        try:
            metadata_version = importlib.metadata.version("hybriddb")
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("hybriddb not installed as a package")
        assert hybriddb.__version__ == metadata_version


# ── Round 2: self-review findings ──────────────────────────────────────────

class TestHnswHeaderCheck:
    def test_healthy_header_not_flagged_corrupt(self, db):
        db.create_table("docs", {"title": TEXT, "body": LONGTEXT})
        db.insert("docs", {"title": "t", "body": "some body text"})
        headers = list(Path(db._vector_path).rglob("header.bin"))
        assert headers
        seg_id = headers[0].parent.name
        dim = db._segment_dimension(seg_id)
        assert dim == 384
        assert db._is_hnsw_header_corrupt(str(headers[0]), expected_size=dim * 4 + 140) is False

    def test_auto_rebuild_not_triggered_on_healthy_index(self, tmp_dir):
        db = HybridDB(tmp_dir, embedding_fn=_mock_embedding, auto_rebuild_chroma=True)
        db.create_table("docs", {"title": TEXT, "body": LONGTEXT})
        db.insert("docs", {"title": "t", "body": "some body text"})
        seg_dirs = {p.name for p in Path(db._vector_path).iterdir() if p.is_dir()}
        db._check_index_health(auto_rebuild=True)  # used to rebuild on every call
        assert {p.name for p in Path(db._vector_path).iterdir() if p.is_dir()} == seg_dirs
        db.close()


class TestReindexMetadata:
    def test_reindex_preserves_chroma_metadata(self, db):
        db.create_table("docs", {"title": TEXT, "body": LONGTEXT, "n": "INTEGER"})
        db.insert("docs", {"title": "t", "body": "some body text", "n": 7})
        coll = db._get_collection("docs_body")
        before = coll.get(include=["metadatas"])["metadatas"]
        db.reindex("docs")
        coll = db._get_collection("docs_body")
        after = coll.get(include=["metadatas"])["metadatas"]
        assert before == after
        assert after and "n" in after[0]


class TestFalsyPk:
    def test_insert_empty_string_pk_returns_pk(self, db_no_chroma):
        db_no_chroma.create_table("kv", {"key": "TEXT PRIMARY KEY", "val": TEXT})
        assert db_no_chroma.insert("kv", {"key": "", "val": "x"}) == ""


# ── Round 3: remaining identified bugs ─────────────────────────────────────

class TestExplicitPkOnDefaultTable:
    def test_insert_honors_explicit_id(self, db_no_chroma):
        db_no_chroma.create_table("docs", {"title": TEXT})
        rid = db_no_chroma.insert("docs", {"id": 100, "title": "explicit"})
        assert rid == 100
        assert db_no_chroma.get("docs", 100)["title"] == "explicit"
        # autoincrement continues after the explicit value
        rid2 = db_no_chroma.insert("docs", {"title": "next"})
        assert rid2 == 101

    def test_insert_batch_honors_explicit_id(self, db_no_chroma):
        db_no_chroma.create_table("docs", {"title": TEXT})
        ids = db_no_chroma.insert_batch("docs", [{"id": 7, "title": "a"}, {"title": "b"}])
        assert ids == [7, 8]

    def test_update_honors_explicit_id(self, db_no_chroma):
        db_no_chroma.create_table("docs", {"title": TEXT})
        rid = db_no_chroma.insert("docs", {"title": "a"})
        assert db_no_chroma.update("docs", rid, {"id": 999}) is True
        assert db_no_chroma.get("docs", 999)["title"] == "a"
        assert db_no_chroma.get("docs", rid) is None

    def test_update_pk_change_keeps_search_and_sync(self, db):
        db.create_table("items", {"uid": "INTEGER PRIMARY KEY", "name": TEXT, "body": LONGTEXT})
        db.insert("items", {"uid": 10, "name": "n", "body": "alpha beta gamma"})
        assert db.update("items", 10, {"uid": 20}) is True
        # INTEGER PRIMARY KEY aliases rowid — the row's Chroma key moved
        sem = db.search("items", "body", "alpha", mode="semantic")
        assert len(sem) == 1 and sem[0]["uid"] == 20
        assert db.get("items", 20)["name"] == "n"
        assert db.get("items", 10) is None
        assert db.health("items")["status"] == "ok"

    def test_text_pk_change_keeps_chroma_key(self, db):
        db.create_table("items", {"uid": "TEXT PRIMARY KEY", "name": TEXT, "body": LONGTEXT})
        db.insert("items", {"uid": "a1", "name": "n", "body": "alpha beta gamma"})
        assert db.update("items", "a1", {"uid": "a2"}) is True
        sem = db.search("items", "body", "alpha", mode="semantic")
        assert len(sem) == 1 and sem[0]["uid"] == "a2"
        assert db.health("items")["status"] == "ok"


class TestGraphRuleValidation:
    def test_register_entity_node_rejects_missing_id_column(self, db_no_chroma):
        db_no_chroma.create_table("items", {"uid": "TEXT PRIMARY KEY", "name": TEXT})
        with pytest.raises(ValueError, match="id_column"):
            db_no_chroma.register_entity_node("items", id_column="nope")

    def test_stale_edge_rule_does_not_crash_sync_or_search(self, db_no_chroma):
        db_no_chroma.create_table("orders", {"oid": "TEXT PRIMARY KEY", "customer_id": TEXT})
        db_no_chroma.create_table("customers", {"cid": "TEXT PRIMARY KEY", "name": TEXT})
        db_no_chroma.insert("orders", {"oid": "o1", "customer_id": "c1"})
        db_no_chroma.insert("customers", {"cid": "c1", "name": "Acme"})
        # rule referencing a column that does not exist in the source table
        db_no_chroma.register_edge_rule("orders", "customers",
                                        source_column="nope", target_column="cid")
        db_no_chroma._auto_sync_graph_edges()  # used to raise OperationalError
        db_no_chroma.reconcile("orders")

    def test_stale_node_rule_does_not_crash_sync(self, db_no_chroma):
        db_no_chroma.create_table("items", {"uid": "TEXT PRIMARY KEY", "name": TEXT})
        db_no_chroma.insert("items", {"uid": "a1", "name": "x"})
        db_no_chroma.register_entity_node("items", id_column="uid")
        # rename the id column the rule references — sync must warn, not crash
        db_no_chroma.rename_column("items", "uid", "uid2")
        db_no_chroma.sync_graph_nodes()


class TestMetaUpdateJournal:
    def test_rename_column_refreshes_chroma_metadata(self, db):
        db.create_table("docs", {"title": TEXT, "body": LONGTEXT, "n": "INTEGER"})
        db.insert("docs", {"title": "t", "body": "some body text", "n": 7})
        db.rename_column("docs", "n", "m")
        db.process_journal()  # runs the meta_update entries
        coll = db._get_collection("docs_body")
        meta = coll.get(include=["metadatas"])["metadatas"][0]
        assert "m" in meta and "n" not in meta
        assert meta["m"] == 7

    def test_drop_column_refreshes_chroma_metadata(self, db):
        db.create_table("docs", {"title": TEXT, "body": LONGTEXT, "n": "INTEGER"})
        db.insert("docs", {"title": "t", "body": "some body text", "n": 7})
        db.drop_column("docs", "n")
        db.process_journal()
        coll = db._get_collection("docs_body")
        meta = coll.get(include=["metadatas"])["metadatas"][0]
        assert "n" not in meta
        assert meta["title"] == "t"

    def test_meta_update_self_heals_empty_collection(self, db):
        db.create_table("docs", {"title": TEXT, "body": LONGTEXT})
        db.insert("docs", {"title": "t", "body": "widgets and gears"})
        db.rename_column("docs", "body", "content")
        # simulate the rename's Chroma copy having failed: collection empty
        db._chroma.delete_collection("docs_content")
        db._get_collection("docs_content")
        db.process_journal()
        assert db._get_collection("docs_content").count() == 1
        sem = db.search("docs", "content", "widgets", mode="semantic")
        assert len(sem) == 1


class TestPagerankValidation:
    def test_pagerank_rejects_unknown_personalization_nodes(self, db_no_chroma):
        db_no_chroma.add_node("n1")
        db_no_chroma.add_node("n2")
        db_no_chroma.add_edge(None, "n1", "n2", weight=0.8)
        with pytest.raises(ValueError, match="not in the graph"):
            db_no_chroma.pagerank(personalization={"ghost": 1.0})


class TestCustomEmbeddingDim:
    def test_empty_doc_uses_custom_dim(self, tmp_dir):
        dim = 8

        def small_embedding(text):
            if not text:
                return [0.0] * dim
            return [1.0] * dim

        db = HybridDB(tmp_dir, embedding_fn=small_embedding)
        db.create_table("docs", {"title": TEXT, "body": LONGTEXT})
        # NULL/empty longtext must not produce a 384-dim vector
        db.insert("docs", {"title": "t", "body": None})
        db.insert("docs", {"title": "u", "body": ""})
        db.insert("docs", {"title": "v", "body": "real text"})
        assert db._get_collection("docs_body").count() == 3
        assert db.search("docs", "body", "real", mode="semantic")


class TestReadQueryReadOnly:
    def test_with_wrapped_dml_is_denied(self, db_no_chroma):
        db_no_chroma.create_table("docs", {"title": TEXT})
        db_no_chroma.insert("docs", {"title": "a"})
        db_no_chroma.insert("docs", {"title": "b"})
        with pytest.raises(sqlite3.DatabaseError):
            db_no_chroma.read_query("WITH x AS (SELECT 1) DELETE FROM docs WHERE title='a'")
        assert db_no_chroma.count("docs") == 2

    def test_with_wrapped_insert_is_denied(self, db_no_chroma):
        db_no_chroma.create_table("docs", {"title": TEXT})
        with pytest.raises(sqlite3.DatabaseError):
            db_no_chroma.read_query("WITH x AS (SELECT 1) INSERT INTO docs (title) VALUES ('z')")
        assert db_no_chroma.count("docs") == 0

    def test_plain_select_still_works(self, db_no_chroma):
        db_no_chroma.create_table("docs", {"title": TEXT})
        db_no_chroma.insert("docs", {"title": "a"})
        assert db_no_chroma.read_query("SELECT * FROM docs")[0]["title"] == "a"
        assert db_no_chroma.read_query("WITH x AS (SELECT 1) SELECT * FROM docs")[0]["title"] == "a"


class TestDropPkColumn:
    def test_drop_custom_pk_rejected(self, db_no_chroma):
        db_no_chroma.create_table("items", {"uid": "TEXT PRIMARY KEY", "name": TEXT})
        db_no_chroma.insert("items", {"uid": "abc", "name": "x"})
        with pytest.raises(ValueError, match="primary key"):
            db_no_chroma.drop_column("items", "uid")
        # data untouched
        assert db_no_chroma.get("items", "abc")["name"] == "x"


class TestFtsLikeFallback:
    def test_like_fallback_escapes_wildcards(self, db_no_chroma):
        db_no_chroma.create_table("docs", {"title": TEXT, "body": LONGTEXT})
        db_no_chroma.insert("docs", {"title": "100% done", "body": "x"})
        db_no_chroma.insert("docs", {"title": "100 done", "body": "y"})
        # break the FTS table to force the LIKE fallback path
        with db_no_chroma._connect() as cur:
            cur.execute("DROP TABLE docs_fts_title")
        hits = db_no_chroma._fts_search("docs", "title", "100%", 10)
        assert [h[0] for h in hits] == [1]  # literal %, not a wildcard
        hits = db_no_chroma._fts_search("docs", "title", "100", 10)
        assert len(hits) == 2


class TestSyncTrueDrainsJournal:
    def test_insert_batch_sync_true_drains_fully(self, db_no_chroma):
        """sync=True must leave no pending journal, even for batches larger
        than the per-call processing limit (5000)."""
        db_no_chroma.create_table("docs", {"title": TEXT})
        rows = [{"title": f"t{i}"} for i in range(6_000)]
        db_no_chroma.insert_batch("docs", rows, sync=True)
        assert db_no_chroma._journal_count("docs") == 0
        assert db_no_chroma.count("docs") == 6_000

    def test_insert_batch_sync_false_leaves_pending(self, db_no_chroma):
        db_no_chroma.create_table("docs", {"title": TEXT})
        rows = [{"title": f"t{i}"} for i in range(100)]
        db_no_chroma.insert_batch("docs", rows, sync=False)
        assert db_no_chroma._journal_count("docs") > 0
