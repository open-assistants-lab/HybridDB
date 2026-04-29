"""HybridDB: SQLite + FTS5 + ChromaDB with self-healing journal.

TEXT columns get keyword search. LONGTEXT columns get keyword + semantic search.
All backed by an operation journal that guarantees consistency across SQLite, FTS5, and ChromaDB.
"""

__version__ = "0.1.0"

from hybriddb.db import EmbeddingModelError, HybridDB, SearchMode

__all__ = ["HybridDB", "SearchMode", "EmbeddingModelError"]
