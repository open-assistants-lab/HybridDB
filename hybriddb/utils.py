"""Shared internal HybridDB utilities."""

import functools
import re
import sqlite3
from datetime import UTC, datetime
from typing import Any

from hybriddb.types import Column, FTS5UnavailableError, SearchMode

_SYSTEM_TABLES = {
    "_journal", "_schema", "_graph_nodes", "_graph_edges",
    "_graph_sync", "_edge_rules", "_versioned_tables",
    "_version_checkpoints", "_chain_anchors",
}

JOURNAL_CAP = 50_000
CHROMA_BATCH = 5000
RRF_K = 60

_CHROMA_INDEX_WARN_FACTOR = 0.5
_CHROMA_INDEX_MAX_M0 = 256
_CHROMA_INDEX_MAX_ELEMENTS = 10_000_000
_CHROMA_REBUILD_BATCH = 5000
# chroma-hnswlib stores label + extra data per element on top of the
# vector itself: size_data_per_element = 4 * dim + this overhead.
_CHROMA_HNSW_DATA_OVERHEAD = 140

_SKIP_SEARCH_COLUMNS: set[str] = {
    "rowid", "id", "memory_id", "fact_key", "scope", "project_id",
    "created_at", "updated_at", "previous_value",
}

_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _is_safe_identifier(name: str) -> bool:
    return bool(_SAFE_IDENTIFIER_RE.match(name))


def _validate_identifier(name: str, kind: str = "identifier") -> None:
    if not isinstance(name, str) or not _is_safe_identifier(name):
        raise ValueError(f"Invalid identifier for {kind}: {name!r}")


def _validate_order_by(order_by: str) -> None:
    if not order_by:
        return
    for part in order_by.split(","):
        tokens = part.strip().split()
        if not tokens or len(tokens) > 2:
            raise ValueError(f"Invalid order_by expression: {order_by!r}")
        _validate_identifier(tokens[0], "order_by column")
        if len(tokens) == 2 and tokens[1].upper() not in {"ASC", "DESC"}:
            raise ValueError(f"Invalid order_by direction: {tokens[1]!r}")


def _coerce_search_mode(mode: SearchMode | str | Any) -> SearchMode:
    if isinstance(mode, SearchMode):
        return mode
    value = mode.value if hasattr(mode, "value") else mode
    if isinstance(value, str):
        try:
            return SearchMode(value.lower())
        except ValueError as e:
            raise ValueError(f"Invalid search mode: {mode!r}") from e
    raise ValueError(f"Invalid search mode: {mode!r}")


def _column_spec(spec: str | Column) -> str:
    return str(spec)


def _sanitize_fts_query(query: str) -> str:
    q = re.sub(r"[^\w\s]", " ", query.strip())
    q = " ".join(q.split())
    if not q:
        return ""
    return " OR ".join(q.split())


@functools.lru_cache(maxsize=1)
def sqlite_has_fts5() -> bool:
    """Whether this interpreter's SQLite was built with FTS5.

    FTS5 is a *compile-time* option (SQLITE_ENABLE_FTS5), so it is not present
    in every Python build. HybridDB's keyword search is built on it, so probe
    once and report the problem clearly instead of letting a raw
    "no such module: fts5" surface from the CREATE VIRTUAL TABLE statement.

    The probe is cached: the answer cannot change within a process.
    """
    try:
        con = sqlite3.connect(":memory:")
        try:
            con.execute(
                "CREATE VIRTUAL TABLE _hybriddb_fts5_probe USING fts5(x)"
            )
        finally:
            con.close()
    except sqlite3.Error:
        return False
    return True


def require_fts5() -> None:
    """Raise a clear error if this SQLite build lacks FTS5."""
    if sqlite_has_fts5():
        return
    raise FTS5UnavailableError(
        "This Python's SQLite was built without FTS5 "
        f"(sqlite3.sqlite_version={sqlite3.sqlite_version}), which HybridDB "
        "requires for keyword and hybrid search. Use a Python build with "
        "SQLITE_ENABLE_FTS5 (the standard CPython builds on PyPI, Homebrew and "
        "the major Linux distributions all include it), or rebuild SQLite with "
        "the flag. Check with: "
        "python -c \"import sqlite3;"
        "print(sqlite3.sqlite_version)\""
    )
