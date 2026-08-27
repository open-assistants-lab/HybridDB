"""Shared internal HybridDB utilities."""

import re
from datetime import UTC, datetime
from typing import Any

from hybriddb.types import Column, SearchMode

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
