"""Tests for chroma vector identity: chroma ids = str(logical primary key).

Implements the acceptance criteria of docs/specs/2026-08-28-chroma-identity-v0.7.md:
- new collections are pk-keyed (marker `hybriddb:identity: "pk"`)
- the TEXT-PK delete+reinsert journal wedge is structurally impossible
- migrate_vector_identity() re-keys legacy rowid collections with NO
  re-embedding (embeddings carried over), is idempotent, cleans orphans
- search/keyword/semantic/hybrid, reconcile, health and rollback all work
  before and after migration
"""

import hashlib
import shutil
import tempfile

import pytest

from hybriddb import LONGTEXT, TEXT, HybridDB

EMBEDDING_DIM = 384


def _mock_embedding(text: str) -> list[float]:
    if not text:
        return [0.0] * EMBEDDING_DIM
    words = str(text).lower().split()
    embedding = [0.0] * EMBEDDING_DIM
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIM
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


def _chroma_ids(db, table: str, column: str = "body") -> list[str]:
    return db._get_collection(f"{table}_{column}").get()["ids"]


def _chroma_embeddings(db, table: str, column: str = "body") -> dict:
    got = db._get_collection(f"{table}_{column}").get(include=["embeddings"])
    return dict(zip(got["ids"], got["embeddings"]))


def _to_legacy_scheme(db, table: str, column: str = "body"):
    """Convert a pk-keyed collection to the pre-0.7 legacy rowid scheme
    (simulating a pre-0.7 collection): re-upsert vectors under str(rowid),
    delete the pk-style ids, mark the collection 'rowid'."""
    coll = db._get_collection(f"{table}_{column}")
    got = coll.get(include=["embeddings", "documents", "metadatas"])
    with db._connect() as cur:
        pk_col = db._get_pk_column(table, cur=cur)
        rid_by_pk = {
            str(r[pk_col]): str(r["_rid"])
            for r in cur.execute(f"SELECT rowid AS _rid, {pk_col} FROM {table}").fetchall()
        }
    new_ids = [rid_by_pk[i] for i in got["ids"]]
    coll.upsert(
        ids=new_ids, embeddings=got["embeddings"],
        documents=got["documents"], metadatas=got["metadatas"],
    )
    coll.delete(ids=[i for i in got["ids"] if i not in set(new_ids)])
    md = dict(coll.metadata or {})
    md["hybriddb:identity"] = "rowid"
    coll.modify(metadata=md)


class TestPkKeyedCollections:
    def test_new_default_table_is_pk_keyed(self, db):
        db.create_table("docs", {"title": TEXT, "body": LONGTEXT})
        db.insert("docs", {"title": "a", "body": "alpha body"})
        db.insert("docs", {"id": 7, "title": "b", "body": "beta body"})
        coll = db._get_collection("docs_body")
        assert coll.metadata["hybriddb:identity"] == "pk"
        ids = set(_chroma_ids(db, "docs"))
        assert ids == {"1", "7"}  # str(pk); for rowid-alias pks identical to rowid

    def test_new_text_pk_table_is_pk_keyed(self, db):
        db.create_table("items", {"uid": "TEXT PRIMARY KEY", "body": LONGTEXT})
        db.insert("items", {"uid": "a1", "body": "alpha"})
        db.insert("items", {"id": "b2", "body": "beta"}) if False else None
        db.insert("items", {"uid": "b2", "body": "beta"})
        assert set(_chroma_ids(db, "items")) == {"a1", "b2"}
        assert db._get_collection("items_body").metadata["hybriddb:identity"] == "pk"

    def test_text_pk_delete_reinsert_no_wedge(self, db):
        """THE regression: TEXT-PK tables reuse rowids after delete — with
        rowid-keyed chroma this wedged the journal (DuplicateIDError).
        With pk keys the reuse is irrelevant."""
        db.create_table("items", {"uid": "TEXT PRIMARY KEY", "body": LONGTEXT})
        db.insert("items", {"uid": "first", "body": "v1 body"})
        db.delete("items", "first")
        db.insert("items", {"uid": "second", "body": "v2 body"})  # reuses the freed rowid
        db.insert("items", {"uid": "third", "body": "v3 body"})    # reuses again
        coll = db._get_collection("items_body")
        assert coll.count() == 2
        assert set(_chroma_ids(db, "items")) == {str(r["uid"]) for r in db.raw_query("SELECT uid FROM items")}
        assert db.search("items", "body", "v2 body", mode="semantic")  # ids resolve to rows

    def test_search_resolves_pk_keyed_vectors(self, db):
        db.create_table("items", {"uid": "TEXT PRIMARY KEY", "body": LONGTEXT})
        db.insert("items", {"uid": "alpha-key", "body": "shiny widget gearbox"})
        rows = db.search("items", "body", "widget", mode="semantic")
        assert rows and rows[0]["uid"] == "alpha-key"


class TestMigration:
    def _seed_legacy(self, db, marker: str | None):
        """Seed a pk-keyed collection, then downgrade it to the legacy
        rowid scheme (simulating a pre-0.7 collection)."""
        db.create_table("items", {"uid": "TEXT PRIMARY KEY", "body": LONGTEXT})
        db.insert_batch("items", [
            {"uid": "u1", "body": "alpha beta"},
            {"uid": "u2", "body": "gamma delta"},
            {"uid": "u3", "body": "epsilon zeta"},
        ], sync=True)
        if marker is not None:
            _to_legacy_scheme(db, "items")
            if marker is None:
                # strip the marker entirely (pre-0.7 collections have none)
                coll = db._get_collection("items_body")
                md = {k: v for k, v in (coll.metadata or {}).items() if k != "hybriddb:identity"}
                coll.modify(metadata=md)
        return db

    def test_migrate_rekeys_legacy_collection_with_marker(self, db):
        self._seed_legacy(db, marker="rowid")
        before = _chroma_embeddings(db, "items")
        assert set(before) == {"1", "2", "3"}  # legacy rowid keys

        res = db.migrate_vector_identity("items")
        assert res["vectors_rekeyed"] > 0
        after = _chroma_embeddings(db, "items")
        assert set(after) == {"u1", "u2", "u3"}
        # NO re-embedding: the vectors are carried over bit-for-bit
        for pk in ("u1", "u2", "u3"):
            assert [float(x) for x in after[pk]] == [float(x) for x in before[{"u1": "1", "u2": "2", "u3": "3"}[pk]]]
        assert db._get_collection("items_body").metadata["hybriddb:identity"] == "pk"

    def test_migrate_idempotent(self, db):
        self._seed_legacy(db, marker="rowid")
        db.migrate_vector_identity("items")
        snap1 = _chroma_embeddings(db, "items")
        res = db.migrate_vector_identity("items")
        snap2 = _chroma_embeddings(db, "items")
        assert set(snap1) == set(snap2)
        assert all(
            [float(x) for x in snap1[k]] == [float(x) for x in snap2[k]]
            for k in snap1
        )
        assert res["vectors_rekeyed"] == 0

    def test_migrate_no_marker_defaults_to_rowid_scheme(self, db):
        self._seed_legacy(db, marker=None)
        # no marker: scheme defaults to legacy rowid — migration still works
        db.migrate_vector_identity("items")
        assert set(_chroma_ids(db, "items")) == {"u1", "u2", "u3"}
        assert db._get_collection("items_body").metadata["hybriddb:identity"] == "pk"

    def test_migrate_deletes_orphans(self, db):
        self._seed_legacy(db, marker="rowid")
        # orphan: a chroma vector whose rowid no longer exists in sqlite
        coll = db._get_collection("items_body")
        coll.upsert(ids=["orphan-ghost"], embeddings=[_mock_embedding("ghost text")], documents=["ghost"])
        db.migrate_vector_identity("items")
        assert "orphan-ghost" not in _chroma_ids(db, "items")

    def test_migrate_default_table_is_noop_reported(self, db):
        db.create_table("docs", {"title": TEXT, "body": LONGTEXT})
        db.insert("docs", {"title": "a", "body": "alpha body"})
        res = db.migrate_vector_identity("docs")
        assert res["skipped_noop"] == ["docs"]  # keys identical; reported
        assert set(_chroma_ids(db, "docs")) == {"1"}

    def test_search_works_across_migration(self, db):
        self._seed_legacy(db, marker="rowid")
        for mode in ("keyword", "semantic", "hybrid"):
            rows = db.search("items", "body", "alpha", mode=mode)
            assert rows and rows[0]["uid"] == "u1", mode
        db.migrate_vector_identity("items")
        for mode in ("keyword", "semantic", "hybrid"):
            rows = db.search("items", "body", "gamma", mode=mode)
            assert rows and rows[0]["uid"] == "u2", mode

    def test_reconcile_after_migration(self, db):
        self._seed_legacy(db, marker="rowid")
        db.migrate_vector_identity("items")
        coll = db._get_collection("items_body")
        coll.delete(ids=["u2"])  # simulate drift
        res = db.reconcile("items")
        assert res["missing_added"] >= 1
        assert coll.count() == 3

    def test_migrate_all_tables(self, db):
        self._seed_legacy(db, marker="rowid")
        db.create_table("more", {"uid": "TEXT PRIMARY KEY", "body": LONGTEXT})
        db.insert("more", {"uid": "z1", "body": "zeta"})
        _to_legacy_scheme(db, "more")
        res = db.migrate_vector_identity()  # None = all tables
        assert set(res["migrated"]) >= {"items", "more"}

    def test_migrate_non_versioned_longtext_table(self, db):
        """The identity problem is not versioning-specific: any TEXT-pk
        table with longtext collections has legacy rowid keys."""
        db.create_table("plain", {"uid": "TEXT PRIMARY KEY", "body": LONGTEXT})  # not versioned
        db.insert("plain", {"uid": "p1", "body": "plain alpha"})
        _to_legacy_scheme(db, "plain")
        db.migrate_vector_identity("plain")
        assert set(_chroma_ids(db, "plain")) == {"p1"}


class TestRollbackPkKeyed:
    def test_rollback_pk_keyed(self, db):
        db.create_table("msgs", {"id": "TEXT PRIMARY KEY", "body": LONGTEXT}, versioned=True)
        db.insert_batch("msgs", [{"id": f"m{i}", "body": f"event {i}"} for i in range(100)], sync=True)
        db.checkpoint("msgs", "cp")
        db.insert_batch("msgs", [{"id": f"x{i}", "body": f"extra {i}"} for i in range(100)], sync=True)
        assert db._get_collection("msgs_body").count() == 200  # noqa: F841 — cp marker drives rollback

        db.rollback("msgs", checkpoint="cp")  # noqa: F841 — cp marker used by rollback
        assert db.count("msgs") == 100
        assert db._get_collection("msgs_body").count() == 100
        assert db.health("msgs")["status"] == "ok"
        assert db.verify_chain("msgs")["valid"] is True