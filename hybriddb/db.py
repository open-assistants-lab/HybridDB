"""HybridDB: SQLite + FTS5 + ChromaDB + Graph + DuckDB with self-healing journal.

TEXT columns get keyword search. LONGTEXT columns get keyword + semantic search.
Graph capabilities: SQLite-backed nodes/edges, recursive CTE traversal, NetworkX algorithms.
Analytics: DuckDB columnar store synced via unified journal for fast OLAP queries.

All backed by an operation journal that guarantees consistency across all engines.
"""

import logging
import os
import sqlite3
import threading
import weakref
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from hybriddb.analytics import AnalyticsMixin
from hybriddb.async_api import AsyncMixin
from hybriddb.crud import CrudMixin
from hybriddb.embedding import (
    default_embedding_fn as _default_embedding_fn,
)
from hybriddb.export_import import ExportImportMixin
from hybriddb.facades import AnalyticsAPI, GraphAPI
from hybriddb.graph import GraphMixin
from hybriddb.journal import JournalMixin
from hybriddb.versioning import VersioningMixin
from hybriddb.maintenance import MaintenanceMixin
from hybriddb.schema import SchemaMixin
from hybriddb.search import SearchMixin
from hybriddb.types import (
    EmbeddingModelError,
)

logger = logging.getLogger("hybriddb")

JOURNAL_CAP = 50_000
CHROMA_BATCH = 5000
RRF_K = 60

_CHROMA_INDEX_WARN_FACTOR = 0.5
_CHROMA_INDEX_MAX_M0 = 256
_CHROMA_INDEX_MAX_ELEMENTS = 10_000_000
_CHROMA_REBUILD_BATCH = 5000

_chroma_client_pool: dict[str, Any] = {}
_chroma_pool_lock = threading.Lock()

# Live HybridDB instances holding each vector path's Chroma client. Replacing
# the vectors/ directory (force_rebuild_chroma_index / restore) is a
# process-global operation, so those paths must know whether another instance
# would be left holding a client bound to the replaced directory.
_chroma_path_holders: dict[str, "weakref.WeakSet[Any]"] = {}
# Per-path re-entrant lock serialising Chroma client *acquisition* against the
# directory swaps in force_rebuild_chroma_index / restore. Without it, a
# HybridDB constructed mid-swap re-attaches to the outgoing directory (or, if
# it recreates vectors/ in the gap between the two moves, ends up nested in
# vectors/vectors/). One lock per vector path, never per instance.
_chroma_path_locks: dict[str, threading.RLock] = {}


def chroma_path_lock(vector_path: str) -> threading.RLock:
    """Return the per-path lock guarding acquisition vs. directory swap.

    Re-entrant: the swap path re-acquires it when re-initialising a client after
    a failure. Always acquired *before* _chroma_pool_lock to keep one order.
    """
    with _chroma_pool_lock:
        lock = _chroma_path_locks.get(vector_path)
        if lock is None:
            lock = threading.RLock()
            _chroma_path_locks[vector_path] = lock
        return lock


def _register_chroma_path_holder(vector_path: str, db: Any) -> None:
    """Record `db` as a live holder of `vector_path`'s Chroma client."""
    with _chroma_pool_lock:
        holders = _chroma_path_holders.get(vector_path)
        if holders is None:
            holders = weakref.WeakSet()
            _chroma_path_holders[vector_path] = holders
        holders.add(db)


def _release_chroma_path_holder(vector_path: str, db: Any) -> None:
    """Forget `db` as a holder (called on close(); GC drops it via weakrefs)."""
    with _chroma_pool_lock:
        holders = _chroma_path_holders.get(vector_path)
        if holders is None:
            return
        holders.discard(db)
        if not holders:
            _chroma_path_holders.pop(vector_path, None)


def other_chroma_path_holders(vector_path: str, db: Any) -> int:
    """Count live HybridDB instances other than `db` holding this vector path."""
    with _chroma_pool_lock:
        holders = _chroma_path_holders.get(vector_path)
        if not holders:
            return 0
        return sum(1 for holder in holders if holder is not db)


def _evict_chroma_path_clients(vector_path: str) -> None:
    """Drop every cached Chroma client and System for `vector_path`.

    ChromaDB keeps a class-level ``_identifier_to_system`` dict keyed by
    persist_directory; each System holds an open SQLite handle to chroma.sqlite.
    Swapping the directory out from under a live System leaves that handle bound
    to the moved-away inode, so every later read/write through the shared client
    fails with SQLITE_READONLY_DBMOVED (1032) or a NotFoundError naming a stale
    collection ID.

    Must run BEFORE a new client is constructed for the path, otherwise the new
    client re-attaches to the stale System. Deliberately scoped to one path:
    ``SharedSystemClient.clear_system_cache()`` wipes every path, which would
    break clients for unrelated databases in the same process.

    ``_identifier_to_refcount`` only exists from chromadb 1.5.2; on 1.5.0/1.5.1
    SharedSystemClient has no refcount table, so there is nothing extra to evict
    and the attribute is skipped rather than raising (issue #1 regression).
    """
    from chromadb.api.shared_system_client import SharedSystemClient

    with _chroma_pool_lock:
        _chroma_client_pool.pop(vector_path, None)
        system = SharedSystemClient._identifier_to_system.pop(vector_path, None)
        refcounts = getattr(SharedSystemClient, "_identifier_to_refcount", None)
        if refcounts is not None:
            with getattr(SharedSystemClient, "_refcount_lock", nullcontext()):
                refcounts.pop(vector_path, None)
    if system is not None:
        try:
            system.stop()
        except Exception:  # noqa: BLE001 - best effort; cache entry already dropped
            logger.debug("chroma_system_stop_failed path=%s", vector_path, exc_info=True)


_SKIP_SEARCH_COLUMNS: set[str] = {
    "rowid", "id", "memory_id", "fact_key", "scope", "project_id",
    "created_at", "updated_at", "previous_value",
}

class HybridDB(
    SchemaMixin,
    CrudMixin,
    SearchMixin,
    JournalMixin,
    VersioningMixin,
    GraphMixin,
    AnalyticsMixin,
    MaintenanceMixin,
    ExportImportMixin,
    AsyncMixin,
):
    """Hybrid search database: SQLite + FTS5 + ChromaDB + Graph + DuckDB with self-healing journal.

    Args:
        path: Directory path for database files (app.db + vectors/).
        embedding_fn: Optional callable that takes text and returns a list of floats.
                      Defaults to hash-based embedding if not provided.
        embedding_model_name: Label for the embedding model (persisted for validation).
        force_model: If True, skip embedding model mismatch check on init.
        max_chroma_index_gb: Maximum ChromaDB HNSW index size in GB before warning/rebuild.
        auto_rebuild_chroma: If True, automatically rebuild bloated/corrupt ChromaDB indexes.

    Example:
        >>> db = HybridDB("./my_data")
        >>> db.create_table("contacts", {"name": "TEXT", "bio": "LONGTEXT"})
        >>> db.insert("contacts", {"name": "Alice", "bio": "Engineer at Acme"})
        >>> results = db.search("contacts", "bio", "engineering")
    """

    def __init__(
        self,
        path: str,
        embedding_fn: Any | None = None,
        embedding_model_name: str | None = None,
        force_model: bool = False,
        max_chroma_index_gb: int = 5,
        auto_rebuild_chroma: bool = False,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self._db_path = str((self.path / "app.db").resolve())
        self._vector_path = str((self.path / "vectors").resolve())
        Path(self._vector_path).mkdir(parents=True, exist_ok=True)

        self._embedding_fn = embedding_fn or _default_embedding_fn
        self._embedding_model_name = embedding_model_name or (
            "custom" if embedding_fn is not None else "chroma:all-MiniLM-L6-v2"
        )
        self._max_chroma_index_gb = max_chroma_index_gb
        self._db_lock = threading.RLock()
        self._hybrid_disabled: dict[str, bool] = {}
        self.author: str | None = None  # recorded in versioned-table history
        self.graph = GraphAPI(self)
        self.olap = AnalyticsAPI(self)

        self._chroma = None
        self._nx_cache: dict[str, Any] = {"graph": None, "dirty": True, "directed": None}

        self._init_system_tables()
        if self._max_chroma_index_gb > 0:
            self._init_chroma(force_model)
        # DuckDB mirrors are created lazily on first OLAP use (db.olap.query /
        # analytics()); tables the user never queries cost nothing to maintain.
        self._init_duckdb()
        if self._max_chroma_index_gb > 0:
            self._check_index_health(auto_rebuild_chroma)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Cursor, None, None]:
        with self._db_lock:
            conn = sqlite3.connect(self._db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.cursor()
                yield cursor
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Return a managed SQLite cursor for custom SQL.

        Prefer high-level methods when possible. This public context manager is
        provided for advanced read queries and small custom migrations.
        """
        return self._connect()

    def connect(self) -> Generator[sqlite3.Cursor, None, None]:
        """Alias for cursor()."""
        return self.cursor()

    def _init_system_tables(self) -> None:
        with self._connect() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS _journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    app_table TEXT NOT NULL,
                    row_id INTEGER,
                    column_name TEXT,
                    op TEXT NOT NULL,
                    data TEXT,
                    metadata TEXT,
                    status TEXT DEFAULT 'pending',
                    error TEXT,
                    created_at TEXT NOT NULL,
                    retries INTEGER DEFAULT 0
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_journal_pending "
                "ON _journal(status, app_table)"
            )

            cur.execute("""
                CREATE TABLE IF NOT EXISTS _schema (
                    table_name TEXT PRIMARY KEY,
                    columns_json TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    is_dirty INTEGER DEFAULT 0,
                    embedding_model TEXT,
                    embedding_dim INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

        self._init_graph_tables()
        self._init_versioning_tables()


    def _init_chroma(self, force: bool = False) -> None:
        key = os.fspath(self._vector_path)
        with chroma_path_lock(key):
            self._init_chroma_locked(key, force)

    def _init_chroma_locked(self, key: str, force: bool) -> None:
        with _chroma_pool_lock:
            if key in _chroma_client_pool:
                try:
                    _chroma_client_pool[key].heartbeat()
                except Exception:
                    _chroma_client_pool.pop(key, None)
                else:
                    self._chroma = _chroma_client_pool[key]

        if self._chroma is None:
            try:
                client = chromadb.PersistentClient(
                    path=self._vector_path,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
            except Exception:
                logger.warning("chroma_init_failed vector_path=%s", self._vector_path)
                self._chroma = None
                return

            with _chroma_pool_lock:
                _chroma_client_pool[key] = client
            self._chroma = client

        _register_chroma_path_holder(key, self)

        with self._connect() as cur:
            cur.execute("SELECT table_name, embedding_model, embedding_dim FROM _schema")
            rows = cur.fetchall()

        for row in rows:
            if row["embedding_model"] and row["embedding_model"] != "unknown":
                if row["embedding_model"] != self._embedding_model_name and not force:
                    raise EmbeddingModelError(
                        f"Embedding model mismatch for table '{row['table_name']}': "
                        f"stored='{row['embedding_model']}', "
                        f"current='{self._embedding_model_name}'. "
                        "Pass force=True to override, then call reconcile()."
                    )
