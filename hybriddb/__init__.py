"""HybridDB: SQLite + FTS5 + ChromaDB + Graph + DuckDB with self-healing journal.

TEXT columns get keyword search. LONGTEXT columns get keyword + semantic search.
Graph capabilities: SQLite-backed nodes/edges, recursive CTE traversal, NetworkX algorithms.
Analytics: DuckDB columnar store synced via unified journal for fast OLAP queries.
"""

__version__ = "0.4.3"

from hybriddb.embedding import default_embedding_fn
from hybriddb.db import HybridDB
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

__all__ = [
    "HybridDB", "SearchMode", "EmbeddingModelError", "Column", "default_embedding_fn",
    "TEXT", "LONGTEXT", "INTEGER", "REAL", "BOOLEAN", "JSON",
    "KEYWORD", "SEMANTIC", "HYBRID",
]
