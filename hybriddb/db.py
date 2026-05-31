"""HybridDB: SQLite + FTS5 + ChromaDB + Graph + DuckDB with self-healing journal.

TEXT columns get keyword search. LONGTEXT columns get keyword + semantic search.
Graph capabilities: SQLite-backed nodes/edges, recursive CTE traversal, NetworkX algorithms.
Analytics: DuckDB columnar store synced via unified journal for fast OLAP queries.

All backed by an operation journal that guarantees consistency across all engines.
"""

import json
import logging
import os
import shutil
import sqlite3
import struct
import tempfile
import threading
import uuid
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from hybriddb.analytics import AnalyticsMixin
from hybriddb.async_api import AsyncMixin
from hybriddb.crud import CrudMixin
from hybriddb.embedding import (
    EMBEDDING_DIM,
    default_embedding_fn,
    default_embedding_fn as _default_embedding_fn,
    hash_embedding as _hash_embedding,
)
from hybriddb.export_import import ExportImportMixin
from hybriddb.facades import AnalyticsAPI, GraphAPI
from hybriddb.graph import GraphMixin
from hybriddb.journal import JournalMixin
from hybriddb.maintenance import MaintenanceMixin
from hybriddb.schema import SchemaMixin
from hybriddb.search import SearchMixin
from hybriddb.types import (
    BOOLEAN,
    HYBRID,
    INTEGER,
    JSON,
    KEYWORD,
    LONGTEXT,
    REAL,
    SEMANTIC,
    TEXT,
    Column,
    EmbeddingModelError,
    SearchMode,
)
from hybriddb.utils import (
    _SYSTEM_TABLES,
    _coerce_search_mode,
    _column_spec,
    _is_safe_identifier,
    _now_iso,
    _sanitize_fts_query,
    _validate_identifier,
    _validate_order_by,
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

_SKIP_SEARCH_COLUMNS: set[str] = {
    "rowid", "id", "memory_id", "fact_key", "scope", "project_id",
    "created_at", "updated_at", "previous_value",
}

class HybridDB(
    SchemaMixin,
    CrudMixin,
    SearchMixin,
    JournalMixin,
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
        self.graph = GraphAPI(self)
        self.olap = AnalyticsAPI(self)

        self._chroma = None
        self._nx_cache: dict[str, Any] = {"graph": None, "dirty": True, "directed": None}

        self._init_system_tables()
        if self._max_chroma_index_gb > 0:
            self._init_chroma(force_model)
        self._init_duckdb()
        self._auto_register_duckdb_tables()
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


    def _init_chroma(self, force: bool = False) -> None:
        key = os.fspath(self._vector_path)
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
