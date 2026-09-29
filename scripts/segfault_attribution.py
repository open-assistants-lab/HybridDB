"""Decisive experiment for HybridDB #5: is the chromadb segfault upstream, or
does HybridDB provoke it?

Runs the same shape of work three ways, with identical call parameters:
  A. pure chromadb, single batch per step        (no HybridDB at all)
  B. pure chromadb, chunked at HybridDB's 5000   (still no HybridDB)
  C. the failing HybridDB path                   (test_versioning perf gate)

If A and B segfault, HybridDB is not required and the bug is in chromadb.
"""

import os
import shutil
import sys
import tempfile

import chromadb
from chromadb.config import Settings as ChromaSettings

CHROMA_BATCH = 5000          # hybriddb.db.CHROMA_BATCH
DIM = 384


def hash_embedding(text: str) -> list[float]:
    """Non-triggering control embedding (hash-based)."""
    if not text:
        return [0.0] * DIM
    v = [((hash(text) >> (i % 17)) & 0xFF) / 255.0 for i in range(DIM)]
    mag = sum(x * x for x in v) ** 0.5
    return [x / mag for x in v] if mag else v


def mock_embedding(text: str) -> list[float]:
    """The embedding the failing test actually uses: md5-hashed sparse
    unit-norm 384-d vectors. This is the variant that provokes the crash."""
    import hashlib
    if not text:
        return [0.0] * DIM
    e = [0.0] * DIM
    for w in str(text).lower().split():
        e[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    mag = sum(x * x for x in e) ** 0.5
    return [x / mag for x in e] if mag else e
    if not text:
        return [0.0] * DIM
    v = [((hash(text) >> (i % 17)) & 0xFF) / 255.0 for i in range(DIM)]
    mag = sum(x * x for x in v) ** 0.5
    return [x / mag for x in v] if mag else v


def fresh_client(path):
    return chromadb.PersistentClient(
        path=path, settings=ChromaSettings(anonymized_telemetry=False)
    )


def run_pure_round(client, col, round_no, per_round, chunk, do_delete,
                   send_metadata=True, marker=False):
    """One round, chunked at `chunk` — mirrors our insert_batch(500) + drain.

    send_metadata=False mirrors HybridDB's journal: chroma rejects an empty
    metadata dict, so `metadatas` is omitted from the kwargs entirely when no
    row carries metadata. marker=True mimics the one-time `modify()` that sets
    the `hybriddb:identity` marker on the collection.
    """
    if marker:
        col.modify(metadata={"hybriddb:identity": "pk"})
    rows = [
        {
            "id": f"x{round_no}_{i}",
            "embedding": mock_embedding(f"extra entry {round_no} {i}"),
            "document": f"extra entry {round_no} {i}",
            "metadata": {"round": str(round_no)} if send_metadata else None,
        }
        for i in range(per_round)
    ]
    for start in range(0, len(rows), chunk):
        part = rows[start : start + chunk]
        kwargs = {
            "ids": [r["id"] for r in part],
            "embeddings": [r["embedding"] for r in part],
            "documents": [r["document"] for r in part],
        }
        if send_metadata:
            kwargs["metadatas"] = [r["metadata"] for r in part]
        col.upsert(**kwargs)
    if do_delete:
        col.delete(ids=[r["id"] for r in rows])


def run_pure(label: str, collection_name: str, per_round: int, chunk: int,
             rounds: int, base: int, do_delete: bool,
             send_metadata: bool = True, marker: bool = False):
    """Pure chromadb, no HybridDB. Optionally seeds a base table like ours."""
    tmp = tempfile.mkdtemp()
    try:
        client = fresh_client(os.path.join(tmp, "vectors"))
        col = client.get_or_create_collection(collection_name)
        if base:
            base_rows = [
                {
                    "id": f"k{i}",
                    "embedding": mock_embedding(f"knowledge entry {i} lorem ipsum"),
                    "document": f"knowledge entry {i} lorem ipsum",
                    "metadata": {"base": "1"} if send_metadata else None,
                }
                for i in range(base)
            ]
            for start in range(0, len(base_rows), chunk):
                part = base_rows[start : start + chunk]
                kwargs = {
                    "ids": [r["id"] for r in part],
                    "embeddings": [r["embedding"] for r in part],
                    "documents": [r["document"] for r in part],
                }
                if send_metadata:
                    kwargs["metadatas"] = [r["metadata"] for r in part]
                col.upsert(**kwargs)
        for round_no in range(rounds):
            run_pure_round(client, col, round_no, per_round, chunk, do_delete,
                           send_metadata, marker and round_no == 0)
        print(f"{label}: {rounds} rounds x {per_round} (chunk {chunk}, "
              f"base {base}, delete={do_delete}, meta={send_metadata}, "
              f"marker={marker}), no crash")
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_hybrid_variant(label, per_round, rounds, versioned, do_rollback):
    """Bisect our own layer: same journal write path, but toggle the HybridDB
    features (versioned history, rollback) that the pure-chroma arms lack."""
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hybriddb import LONGTEXT, HybridDB

    tmp = tempfile.mkdtemp()
    try:
        db = HybridDB(tmp, embedding_fn=mock_embedding)
        db.create_table("kb", {"id": "TEXT PRIMARY KEY", "content": LONGTEXT},
                        versioned=versioned)
        db.insert_batch(
            "kb",
            [{"id": f"k{i}", "content": f"knowledge entry {i} lorem ipsum"} for i in range(2000)],
            sync=True,
        )
        if do_rollback:
            db.checkpoint("kb", "pre-churn")
        for round_no in range(rounds):
            db.insert_batch(
                "kb",
                [{"id": f"x{round_no}_{i}", "content": f"extra entry {round_no} {i}"}
                 for i in range(per_round)],
                sync=True,
            )
            while db._journal_count("kb") > 0:
                db.process_journal()
            if do_rollback:
                db.rollback("kb", checkpoint="pre-churn")
        print(f"{label}: completed, no crash")
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_hybrid(label: str, per_round: int, rounds: int):
    """The actual HybridDB path, same volumes."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hybriddb import LONGTEXT, HybridDB

    tmp = tempfile.mkdtemp()
    try:
        db = HybridDB(tmp, embedding_fn=mock_embedding)
        db.create_table("kb", {"id": "TEXT PRIMARY KEY", "content": LONGTEXT}, versioned=True)
        db.insert_batch(
            "kb",
            [{"id": f"k{i}", "content": f"knowledge entry {i} lorem ipsum"} for i in range(2000)],
            sync=True,
        )
        db.checkpoint("kb", "pre-churn")
        for round_no in range(rounds):
            extra = [
                {"id": f"x{round_no}_{i}", "content": f"extra entry {round_no} {i}"}
                for i in range(per_round)
            ]
            db.insert_batch("kb", extra, sync=True)
            while db._journal_count("kb") > 0:
                db.process_journal()
            db.rollback("kb", checkpoint="pre-churn")
        print(f"{label}: completed {rounds} rounds x {per_round} records, no crash")
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("a", "all"):
        # pure chromadb, 1000 in ONE upsert per round, no base table
        run_pure("A pure 1x1000, no base, delete  ", "a_single", 1000, 1000, 3, 0, True)
    if which in ("b", "all"):
        # pure chromadb, our batch shape (2 x 500) + deletes, with a 2000 base
        run_pure("B pure 2x500 + 2000 base + delete", "b_ourshape", 1000, 500, 3, 2000, True)
    if which in ("d", "all"):
        # pure chromadb, our batch shape, NO deletes
        run_pure("D pure 2x500 + 2000 base, no del ", "d_nodel", 1000, 500, 3, 2000, False)
    if which in ("e", "all"):
        # pure chromadb, our shape, but metadatas OMITTED (as our journal does)
        run_pure("E pure 2x500, metadatas OMITTED  ", "e_nometa", 1000, 500, 3, 2000, True,
                 send_metadata=False, marker=True)
    if which in ("f", "all"):
        # same as E but without the identity-marker modify()
        run_pure("F pure 2x500, no meta, no marker ", "f_nometa_nomark", 1000, 500, 3, 2000, True,
                 send_metadata=False, marker=False)
    if which in ("i", "all"):
        # pure chromadb, our shape, but a FRESH collection handle per round --
        # HybridDB's journal calls get_or_create_collection() on every apply
        tmp = tempfile.mkdtemp()
        try:
            client = fresh_client(os.path.join(tmp, "vectors"))
            name = "i_freshhandle"
            client.get_or_create_collection(name)
            for round_no in range(3):
                col = client.get_or_create_collection(name)   # fresh handle, like our journal
                col.modify(metadata={"hybriddb:identity": "pk"})
                rows = [{"id": f"x{round_no}_{i}",
                         "embedding": mock_embedding(f"extra entry {round_no} {i}"),
                         "document": f"extra entry {round_no} {i}"} for i in range(1000)]
                for start in range(0, 1000, 500):
                    part = rows[start:start + 500]
                    col.upsert(ids=[r["id"] for r in part],
                               embeddings=[r["embedding"] for r in part],
                               documents=[r["document"] for r in part])
                col.delete(ids=[r["id"] for r in rows])
            print("I pure, fresh handle per round: no crash")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if which in ("k", "all"):
        # pure chromadb, our shape, but SMALLER upsert calls (10 x 100 per round)
        run_pure("K pure 10x100 per round          ", "k_smallcalls", 1000, 100, 3, 2000, True)
    if which in ("g", "all"):
        # pure chromadb, our shape, but a DuckDB connection is also open in the
        # process (HybridDB creates one in __init__). Tests whether two native
        # libraries coexisting is the trigger.
        import duckdb
        _con = duckdb.connect(":memory:")
        _con.execute("CREATE TABLE probe(i INTEGER)")
        _con.execute("INSERT INTO probe VALUES (1)")
        run_pure("G pure 2x500 + duckdb resident   ", "g_duckdb", 1000, 500, 3, 2000, True)
        _con.close()
    if which in ("h", "all"):
        # pure chromadb, our shape, with duckdb imported but no connection
        import duckdb  # noqa: F401
        run_pure("H pure 2x500 + duckdb imported   ", "h_ddkt", 1000, 500, 3, 2000, True)
    if which in ("l", "all"):
        run_hybrid_variant("L versioned, 3 rounds, WITH rollback   ", 1000, 3, True, True)
    if which in ("m", "all"):
        run_hybrid_variant("M versioned, 3 rounds, NO rollback     ", 1000, 3, True, False)
    if which in ("n", "all"):
        run_hybrid_variant("N NOT versioned, 3 rounds, no rollback", 1000, 3, False, False)
    if which in ("c", "all"):
        run_hybrid("C hybriddb actual path       ", 1000, 3)
    print("DONE-NO-CRASH")
