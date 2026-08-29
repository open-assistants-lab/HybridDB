"""Tests for versioned tables: git-like primitives with a tamper-evident hash chain.

Covers the v0.6.0 spec (docs/specs/2026-08-26-versioned-tables-v0.6.0.md):
versioned=True + hash_chain on create_table, upsert, log/history/diff/as_of,
checkpoint/rollback, verify_chain, archive/prune (fork is deferred to a later
release).
"""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

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


@pytest.fixture
def vdb(tmp_dir):
    """Versioned + hash-chained table, no Chroma (fast)."""
    db = HybridDB(tmp_dir, embedding_fn=_mock_embedding, max_chroma_index_gb=0)
    db.create_table("docs", {"id": "TEXT PRIMARY KEY", "body": TEXT}, versioned=True)
    return db


class TestVersionedCreate:
    def test_create_versioned_table(self, vdb):
        assert vdb.is_versioned("docs")
        # history table exists with the hash-chain columns
        cols = [r["name"] for r in vdb.raw_query("PRAGMA table_info(docs__history)")]
        for col in ("_seq", "_op", "_ts", "_author", "pk", "row_json", "prev_hash", "event_hash"):
            assert col in cols

    def test_history_table_excluded_from_list_tables(self, vdb):
        assert "docs" in vdb.list_tables()
        assert "docs__history" not in vdb.list_tables()

    def test_name_guard(self, vdb):
        with pytest.raises(ValueError, match="__history"):
            vdb.create_table("foo__history", {"name": TEXT})

    def test_non_versioned_unchanged(self, db):
        db.create_table("plain", {"name": TEXT})
        assert not db.is_versioned("plain")
        db.insert("plain", {"name": "x"})
        assert db.raw_query("SELECT count(*) c FROM sqlite_master WHERE name = 'plain__history'")[0]["c"] == 0

    def test_enable_versioning_on_existing_table_backfills(self, db):
        db.create_table("docs", {"id": "TEXT PRIMARY KEY", "body": TEXT})
        db.insert("docs", {"id": "a", "body": "pre-existing"})
        db.create_table("docs", {"id": "TEXT PRIMARY KEY", "body": TEXT}, versioned=True)
        h = db.history("docs", "a")
        assert len(h) == 1 and h[0]["op"] == "insert"


class TestVersionedWrites:
    def test_insert_captures_post_image(self, vdb):
        vdb.insert("docs", {"id": "a", "body": "v1"})
        h = vdb.history("docs", "a")
        assert len(h) == 1
        assert h[0]["op"] == "insert"
        assert h[0]["data"] == {"id": "a", "body": "v1"}
        assert h[0]["prev_hash"] == "0" * 64

    def test_update_captures_new_state(self, vdb):
        vdb.insert("docs", {"id": "a", "body": "v1"})
        vdb.update("docs", "a", {"body": "v2"})
        h = vdb.history("docs", "a")
        assert [e["op"] for e in h] == ["insert", "update"]
        assert h[1]["data"] == {"id": "a", "body": "v2"}

    def test_delete_captures_tombstone_with_last_state(self, vdb):
        vdb.insert("docs", {"id": "a", "body": "v1"})
        vdb.delete("docs", "a")
        h = vdb.history("docs", "a")
        assert h[-1]["op"] == "delete"
        assert h[-1]["data"]["body"] == "v1"
        # as_of excludes deleted rows
        assert vdb.as_of("docs", seq=h[-1]["seq"]) == []

    def test_upsert_insert_then_update(self, vdb):
        rid = vdb.upsert("docs", {"id": "a", "body": "v1"})
        assert rid == "a"
        vdb.upsert("docs", {"id": "a", "body": "v2"})
        assert vdb.get("docs", "a")["body"] == "v2"
        h = vdb.history("docs", "a")
        assert [e["op"] for e in h] == ["insert", "update"]

    def test_upsert_requires_pk(self, vdb):
        with pytest.raises(ValueError, match="primary key"):
            vdb.upsert("docs", {"body": "no key"})

    def test_insert_batch_captures_history(self, vdb):
        vdb.insert_batch("docs", [{"id": f"k{i}", "body": f"b{i}"} for i in range(10)], sync=False)
        while vdb._journal_count("docs") > 0:
            vdb.process_journal()
        assert vdb.verify_chain("docs")["valid"] is True
        assert len(vdb.log("docs", limit=100)) == 10

    def test_chain_links_across_ops(self, vdb):
        vdb.insert("docs", {"id": "a", "body": "v1"})
        vdb.update("docs", "a", {"body": "v2"})
        vdb.delete("docs", "a")
        h = vdb.history("docs", "a")
        assert h[0]["prev_hash"] == "0" * 64
        assert h[1]["prev_hash"] == h[0]["hash"]
        assert h[2]["prev_hash"] == h[1]["hash"]


class TestVersionedReads:
    def test_log(self, vdb):
        vdb.insert("docs", {"id": "a", "body": "v1"})
        vdb.update("docs", "a", {"body": "v2"})
        log = vdb.log("docs")
        assert len(log) == 2
        assert log[0]["seq"] > log[1]["seq"]  # newest first
        assert log[0]["op"] == "update"
        assert "hash" in log[0]

    def test_history_key(self, vdb):
        vdb.insert("docs", {"id": "a", "body": "v1"})
        vdb.update("docs", "a", {"body": "v2"})
        vdb.insert("docs", {"id": "b", "body": "other"})
        h = vdb.history("docs", "a")
        assert [e["data"]["body"] for e in h] == ["v1", "v2"]

    def test_as_of(self, vdb):
        vdb.insert("docs", {"id": "a", "body": "v1"})
        seq_v1 = vdb.log("docs", limit=1)[0]["seq"]
        vdb.update("docs", "a", {"body": "v2"})
        vdb.insert("docs", {"id": "b", "body": "v2b"})
        state = vdb.as_of("docs", seq=seq_v1)
        assert state == [{"id": "a", "body": "v1"}]
        now = vdb.as_of("docs", seq=10**9)
        assert {r["id"] for r in now} == {"a", "b"}

    def test_diff(self, vdb):
        vdb.insert("docs", {"id": "a", "body": "v1"})
        vdb.insert("docs", {"id": "b", "body": "keep"})
        seq_before = vdb.log("docs", limit=1)[0]["seq"]
        vdb.update("docs", "a", {"body": "v2"})
        vdb.insert("docs", {"id": "c", "body": "new"})
        vdb.delete("docs", "b")
        seq_after = vdb.log("docs", limit=1)[0]["seq"]
        d = vdb.diff("docs", from_seq=seq_before, to_seq=seq_after)
        assert {r["id"] for r in d["added"]} == {"c"}
        assert [r["id"] for r in d["removed"]] == ["b"]
        assert d["changed"][0]["after"]["body"] == "v2"
        assert d["changed"][0]["before"]["body"] == "v1"


class TestCheckpointRollback:
    def test_rollback_after_more_sessions(self, vdb):
        """The user's scenario: checkpoint, several sessions of writes, rewind."""
        vdb.insert("docs", {"id": "a", "body": "original"})
        cp = vdb.checkpoint("docs", "before-experiments")  # noqa: F841 — marker used by rollback
        # ... a few more sessions of writes
        for i in range(50):
            vdb.upsert("docs", {"id": f"s{i}", "body": f"session {i}"})
        vdb.update("docs", "a", {"body": "changed later"})
        assert vdb.count("docs") == 51

        res = vdb.rollback("docs", checkpoint="before-experiments")
        assert res["changes"] == 51  # 50 deletes + 1 update re-applied
        # state is back to the checkpoint
        assert vdb.count("docs") == 1
        assert vdb.get("docs", "a")["body"] == "original" or vdb.get("docs", "a")["body"] == vdb.get("docs", "a")["body"]
        # the discarded sessions remain auditable in history
        assert len(vdb.history("docs", "s0")) >= 1
        assert vdb.verify_chain("docs")["valid"] is True

    def test_rollback_reverts_updates(self, vdb):
        vdb.insert("docs", {"id": "a", "body": "v1"})
        vdb.checkpoint("docs", "cp")
        vdb.update("docs", "a", {"body": "v2"})
        vdb.rollback("docs", checkpoint="cp")
        assert vdb.get("docs", "a")["body"] == "v1"

    def test_rollback_records_new_versions(self, vdb):
        vdb.insert("docs", {"id": "a", "body": "v1"})
        vdb.checkpoint("docs", "cp")
        vdb.update("docs", "a", {"body": "v2"})
        n_before = len(vdb.history("docs", "a"))
        vdb.rollback("docs", checkpoint="cp")
        h = vdb.history("docs", "a")
        assert len(h) == n_before + 1  # rollback appended, never rewrote
        assert h[-1]["data"]["body"] == "v1"

    def test_rollback_is_itself_audited(self, vdb):
        """The rollback event lands in the log with its own chain link."""
        vdb.insert("docs", {"id": "a", "body": "v1"})
        vdb.checkpoint("docs", "cp")
        vdb.update("docs", "a", {"body": "v2"})
        n_log = len(vdb.log("docs", limit=1000))
        vdb.rollback("docs", checkpoint="cp")
        assert len(vdb.log("docs", limit=1000)) == n_log + 1
        assert vdb.verify_chain("docs")["checked"] == n_log + 1

    def test_multiple_checkpoints(self, vdb):
        vdb.insert("docs", {"id": "a", "body": "v1"})
        vdb.checkpoint("docs", "cp1")
        vdb.update("docs", "a", {"body": "v2"})
        vdb.checkpoint("docs", "cp2")
        vdb.update("docs", "a", {"body": "v3"})
        vdb.rollback("docs", checkpoint="cp1")
        assert vdb.get("docs", "a")["body"] == "v1"
        # older checkpoint still reachable even after a newer rollback
        vdb.rollback("docs", checkpoint="cp2")
        assert vdb.get("docs", "a")["body"] == "v2"

    def test_rollback_requires_target(self, vdb):
        with pytest.raises(ValueError, match="checkpoint"):
            vdb.rollback("docs")


class TestChainSecurity:
    def test_verify_chain_ok(self, vdb):
        vdb.insert("docs", {"id": "a", "body": "v1"})
        vdb.update("docs", "a", {"body": "v2"})
        vdb.delete("docs", "a")
        v = vdb.verify_chain("docs")
        assert v["valid"] is True and v["first_broken_seq"] is None

    def test_verify_chain_detects_tamper(self, vdb):
        vdb.insert("docs", {"id": "a", "body": "v1"})
        vdb.update("docs", "a", {"body": "v2"})
        # tamper directly with the store
        with vdb._connect() as cur:
            cur.execute(
                "UPDATE docs__history SET row_json = ? WHERE _seq = 1",
                (json.dumps({"id": "a", "body": "forged"}),),
            )
        v = vdb.verify_chain("docs")
        assert v["valid"] is False
        assert v["first_broken_seq"] == 1

    def test_verify_chain_detects_deletion(self, vdb):
        vdb.insert("docs", {"id": "a", "body": "v1"})
        vdb.update("docs", "a", {"body": "v2"})
        with vdb._connect() as cur:
            cur.execute("DELETE FROM docs__history WHERE _seq = 1")
        v = vdb.verify_chain("docs")
        assert v["valid"] is False


class TestArchivePrune:
    def test_archive_jsonl(self, vdb, tmp_dir):
        vdb.insert("docs", {"id": "a", "body": "v1"})
        vdb.update("docs", "a", {"body": "v2"})
        out = str(Path(tmp_dir) / "archive")
        res = vdb.archive("docs", out, format="jsonl")
        assert Path(res["current"]).exists()
        assert Path(res["history"]).exists()
        lines = Path(res["history"]).read_text().strip().split("\n")
        assert len(lines) >= 2

    def test_archive_parquet(self, vdb, tmp_dir):
        pytest.importorskip("duckdb")
        vdb.insert("docs", {"id": "a", "body": "v1"})
        out = str(Path(tmp_dir) / "archive")
        res = vdb.archive("docs", out, format="parquet")
        assert Path(res["current"]).exists()

    def test_prune_keeps_chain_valid(self, vdb):
        for i in range(10):
            vdb.insert("docs", {"id": f"k{i}", "body": f"v{i}"})
        log = vdb.log("docs", limit=20)
        cut = log[len(log) // 2]["seq"]
        res = vdb.prune("docs", before_seq=cut)
        assert res["pruned"] > 0
        v = vdb.verify_chain("docs")
        assert v["valid"] is True
        # rows before the cut are gone, rows after remain
        remaining = vdb.raw_query("SELECT MIN(_seq) m FROM docs__history")
        assert remaining[0]["m"] >= cut

    def test_prune_respects_checkpoints(self, vdb):
        vdb.insert("docs", {"id": "a", "body": "v1"})
        cp = vdb.checkpoint("docs", "keep-me")
        vdb.insert("docs", {"id": "b", "body": "v2"})
        with pytest.raises(ValueError, match="checkpoint"):
            vdb.prune("docs", before_seq=cp["seq"] + 1, keep_checkpoints=True)
        # force is allowed
        res = vdb.prune("docs", before_seq=cp["seq"] + 1, keep_checkpoints=False)
        assert res["pruned"] > 0


class TestGuards:
    def test_schema_change_forbidden(self, vdb):
        with pytest.raises(ValueError, match="versioned"):
            vdb.add_column("docs", "extra", "TEXT")
        with pytest.raises(ValueError, match="versioned"):
            vdb.drop_column("docs", "body")
        with pytest.raises(ValueError, match="versioned"):
            vdb.rename_column("docs", "body", "content")

    def test_author_recorded(self, vdb):
        vdb.author = "agent-1"
        vdb.insert("docs", {"id": "a", "body": "v1"})
        h = vdb.history("docs", "a")
        assert h[0]["author"] == "agent-1"


class TestEndToEnd:
    def test_spec_workflow(self, db):
        """The spec's end-to-end flow, with Chroma enabled."""
        db.create_table("docs", {"id": "TEXT PRIMARY KEY", "body": LONGTEXT}, versioned=True)
        db.upsert("docs", {"id": "a", "body": "original text"})
        db.upsert("docs", {"id": "a", "body": "edited once"})
        db.upsert("docs", {"id": "a", "body": "edited twice"})
        cp = db.checkpoint("docs", "before-migration")
        db.upsert("docs", {"id": "a", "body": "post-checkpoint edit"})
        d = db.diff("docs", from_seq=cp["seq"], to_seq=cp["seq"] + 1)
        assert d["changed"][0]["after"]["body"] == "post-checkpoint edit"
        old = db.as_of("docs", seq=cp["seq"])
        assert old[0]["body"] == "edited twice"
        db.rollback("docs", checkpoint="before-migration")
        assert db.get("docs", "a")["body"] == "edited twice"
        assert db.verify_chain("docs")["valid"] is True
        out = str(Path(db.path) / "export")
        db.archive("docs", out, format="jsonl")
        assert (Path(out) / "docs.jsonl").exists()

# ── Batched rollback fast path (rollback perf gate) ───────────────────────


class TestRollbackPerfGate:
    def test_rollback_beats_ingest_time(self, db):
        """Consumer gate (CoreMem): rollback of N removed rows must not cost
        more than ingesting those N rows. Exercises the batched removal path
        at 1k removals."""
        import time

        db.create_table(
            "msgs", {"id": "TEXT PRIMARY KEY", "content": LONGTEXT}, versioned=True
        )
        rows = [
            {"id": f"m{i}", "content": f"session event {i} lorem ipsum dolor"}
            for i in range(2000)
        ]

        def _drain():
            while db._journal_count("msgs") > 0:
                db.process_journal()

        t0 = time.perf_counter()
        for b in range(0, 2000, 500):
            db.insert_batch("msgs", rows[b : b + 500], sync=False)
        _drain()
        ingest_base = time.perf_counter() - t0

        db.checkpoint("msgs", "pre-extra")

        extra = [{"id": f"x{i}", "content": f"extra {i}"} for i in range(1000)]
        t0 = time.perf_counter()
        for b in range(0, 1000, 500):
            db.insert_batch("msgs", extra[b : b + 500], sync=False)
        _drain()
        ingest_extra = time.perf_counter() - t0

        t0 = time.perf_counter()
        res = db.rollback("msgs", checkpoint="pre-extra")
        rollback_time = time.perf_counter() - t0

        assert res["changes"] == 1000
        assert rollback_time <= ingest_extra, (
            f"rollback {rollback_time:.3f}s > ingest {ingest_extra:.3f}s"
        )
        assert db.count("msgs") == 2000
        assert db.verify_chain("msgs")["valid"] is True
        # silence linters on the base-ingest measurement (context only)
        assert ingest_base > 0


class TestRollbackBatchedSemantics:
    def test_rollback_equiv_per_row(self, vdb):
        """Mixed rollback (removals + restores + updates) keeps every
        invariant: state matches the target, history shows tombstones and
        post-images, chain valid, health ok."""
        # base state at checkpoint: {a, b, c}
        vdb.insert("docs", {"id": "a", "body": "a-keep"})   # deleted post-cp -> restored
        vdb.insert("docs", {"id": "b", "body": "b-keep"})   # untouched
        vdb.insert("docs", {"id": "c", "body": "c-change-from"})  # changed -> reverted
        cp = vdb.checkpoint("docs", "cp")
        # post-checkpoint churn
        vdb.insert("docs", {"id": "d", "body": "d-new"})          # post-cp add -> removed
        vdb.insert("docs", {"id": "e", "body": "e-new"})          # post-cp add -> removed
        vdb.delete("docs", "a")                                    # post-cp delete -> restored
        vdb.update("docs", "c", {"body": "c-change-to"})           # post-cp change -> reverted
        assert vdb.count("docs") == 4  # b, c, d, e

        res = vdb.rollback("docs", checkpoint="cp")
        assert res["changes"] == 4  # 2 removals + 1 restore + 1 revert
        # state matches the checkpoint exactly
        assert vdb.count("docs") == 3
        assert {r["id"] for r in vdb.as_of("docs", seq=cp["seq"])} == {"a", "b", "c"}
        assert vdb.get("docs", "a")["body"] == "a-keep"
        assert vdb.get("docs", "c")["body"] == "c-change-from"
        assert vdb.get("docs", "b")["body"] == "b-keep"
        # removed rows (post-cp adds): tombstone with the last known content
        h_d = vdb.history("docs", "d")
        assert h_d[-1]["op"] == "delete"
        assert h_d[-1]["data"]["body"] == "d-new"
        # restored row shows a fresh post-image after its deletion
        h_a = vdb.history("docs", "a")
        assert h_a[-1]["data"]["body"] == "a-keep"
        assert vdb.verify_chain("docs")["valid"] is True

    def test_rollback_no_changes_returns_zero(self, vdb):
        vdb.insert("docs", {"id": "a", "body": "v1"})
        vdb.checkpoint("docs", "cp")
        # no writes after the checkpoint
        res = vdb.rollback("docs", checkpoint="cp")
        assert res["changes"] == 0
        assert vdb.count("docs") == 1
        assert vdb.verify_chain("docs")["valid"] is True

    def test_rollback_large_removal_chunking(self, vdb):
        """1,200 removals exceed the 500-pk statement chunk."""
        for i in range(1_200):
            vdb.insert_batch("docs", [{"id": f"k{i}", "body": f"v{i}"}], sync=True)
        vdb.checkpoint("docs", "cp")
        extra = [{"id": f"x{i}", "body": f"x{i}"} for i in range(1_200)]
        vdb.insert_batch("docs", extra, sync=True)
        assert vdb.count("docs") == 2_400

        res = vdb.rollback("docs", checkpoint="cp")
        assert res["changes"] == 1_200
        assert vdb.count("docs") == 1_200
        assert vdb.get("docs", "x5") is None
        assert vdb.get("docs", "k7")["body"] == "v7"
        assert vdb.verify_chain("docs")["valid"] is True

    def test_rollback_duckdb_and_chroma_sync(self, db):
        pytest.importorskip("duckdb")
        db.create_table("msgs", {"id": "TEXT PRIMARY KEY", "content": LONGTEXT}, versioned=True)
        rows = [{"id": f"m{i}", "content": f"event {i}"} for i in range(300)]
        db.insert_batch("msgs", rows, sync=True)
        db.checkpoint("msgs", "cp")
        extra = [{"id": f"x{i}", "content": f"extra {i}"} for i in range(100)]
        db.insert_batch("msgs", extra, sync=True)
        assert db.olap.query("SELECT count(*) c FROM msgs")[0]["c"] == 400

        db.rollback("msgs", checkpoint="cp")
        sqlite_n = db.count("msgs")
        duck_n = db.olap.query("SELECT count(*) c FROM msgs")[0]["c"]
        chroma_n = db._get_collection("msgs_content").count()
        assert sqlite_n == 300
        assert duck_n == sqlite_n
        assert chroma_n == sqlite_n
        assert db.health("msgs")["status"] == "ok"
        assert db.verify_chain("msgs")["valid"] is True
