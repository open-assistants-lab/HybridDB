"""HybridDB: SQLite + FTS5 + ChromaDB + Graph + DuckDB with self-healing journal.

TEXT columns get keyword search. LONGTEXT columns get keyword + semantic search.
Graph capabilities: SQLite-backed nodes/edges, recursive CTE traversal, NetworkX algorithms.
Analytics: DuckDB columnar store synced via unified journal for fast OLAP queries.
"""

__version__ = "0.2.0"

from hybriddb.db import EmbeddingModelError, HybridDB, SearchMode

__all__ = ["HybridDB", "SearchMode", "EmbeddingModelError"]
