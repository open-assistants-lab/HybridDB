from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import struct
import tempfile
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hybriddb.embedding import EMBEDDING_DIM
from hybriddb.types import Column, SearchMode
from hybriddb.utils import (
    CHROMA_BATCH,
    JOURNAL_CAP,
    RRF_K,
    _CHROMA_INDEX_MAX_ELEMENTS,
    _CHROMA_INDEX_MAX_M0,
    _CHROMA_INDEX_WARN_FACTOR,
    _CHROMA_REBUILD_BATCH,
    _SKIP_SEARCH_COLUMNS,
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

class SearchMixin:
    def search(
        self, table: str, column: str, query: str | None = None,
        mode: SearchMode = SearchMode.HYBRID, limit: int = 10,
        fts_weight: float = 0.5, recency_weight: float = 0.0,
        recency_column: str | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[dict]:
        _validate_identifier(table, "table")
        if query is None:
            pk_col = self._get_pk_column(table)
            rows = self.query(table, order_by=f"{pk_col} DESC", limit=limit)
            for r in rows:
                r["_score"] = 0.0
                r["_search_mode"] = "none"
            return rows
        _validate_identifier(column, "column")
        if recency_column:
            _validate_identifier(recency_column, "recency column")
        mode = _coerce_search_mode(mode)
        pending = self._journal_count(table)
        if pending > 0:
            self._process_journal()
        if self._hybrid_disabled.get(table) and mode != SearchMode.KEYWORD:
            mode = SearchMode.KEYWORD

        fts_results: list[tuple[int, float]] = []
        vec_results: list[tuple[int, float]] = []

        meta = self._table_meta(table)
        col_type = meta["columns"].get(column, "") if meta else ""
        if col_type not in ("TEXT", "LONGTEXT"):
            return []

        if mode in (SearchMode.KEYWORD, SearchMode.HYBRID):
            fts_results = self._fts_search(table, column, query, limit * 2)
        if mode in (SearchMode.SEMANTIC, SearchMode.HYBRID) and col_type == "LONGTEXT":
            vec_results = self._vector_search(table, column, query, limit * 2, query_embedding=query_embedding)

        ranked = (fts_results if mode == SearchMode.KEYWORD
                  else vec_results if mode == SearchMode.SEMANTIC
                  else self._fuse_hybrid(fts_results, vec_results, fts_weight))

        if not ranked:
            return []

        row_ids = [r[0] for r in ranked]
        rows = self._fetch_rows_by_ids(table, row_ids)
        results = []
        for row_id, score in ranked:
            row = rows.get(row_id)
            if row is None:
                continue
            final_score = score
            if recency_weight > 0 and recency_column:
                ts_str = row.get(recency_column)
                recency = self._compute_recency(ts_str)
                final_score = score * (1 - recency_weight) + recency * recency_weight
            row["_score"] = final_score
            row["_search_mode"] = mode.value
            results.append(row)
        return results[:limit]

    def _matches_where(self, row: dict, where: dict) -> bool:
        for key, value in where.items():
            if key not in row:
                return False
            if row[key] != value:
                return False
        return True

    def search_all(
        self, table: str, query: str, where: dict | None = None,
        limit: int = 10, fts_weight: float = 0.5,
        recency_weight: float = 0.0, recency_column: str | None = None,
    ) -> list[dict]:
        _validate_identifier(table, "table")
        if recency_column:
            _validate_identifier(recency_column, "recency column")
        pending = self._journal_count(table)
        if pending > 0:
            self._process_journal()

        lt_cols = self._get_longtext_columns(table)
        all_text_cols = [c for c in self._get_text_columns(table) if c not in _SKIP_SEARCH_COLUMNS]

        all_fts: list[tuple[int, float]] = []
        for col in all_text_cols:
            all_fts.extend(self._fts_search(table, col, query, limit * 2))

        all_vec: list[tuple[int, float]] = []
        for col in lt_cols:
            all_vec.extend(self._vector_search(table, col, query, limit * 2))

        ranked = self._fuse_hybrid(all_fts, all_vec, fts_weight)
        if not ranked:
            return []

        row_ids = [r[0] for r in ranked]
        rows = self._fetch_rows_by_ids(table, row_ids)
        results = []
        for row_id, score in ranked:
            row = rows.get(row_id)
            if row is None:
                continue
            if where and not self._matches_where(row, where):
                continue
            final_score = score
            if recency_weight > 0 and recency_column:
                ts_str = row.get(recency_column)
                recency = self._compute_recency(ts_str)
                final_score = score * (1 - recency_weight) + recency * recency_weight
            row["_score"] = final_score
            row["_search_mode"] = "hybrid"
            results.append(row)
        return results[:limit]

    def search_columns(
        self, table: str, query: str, where: dict | None = None,
        limit: int = 10, fts_weight: float = 0.5,
        recency_weight: float = 0.0, recency_column: str | None = None,
    ) -> list[dict]:
        """Search across all text columns in a table."""
        return self.search_all(
            table, query, where=where, limit=limit, fts_weight=fts_weight,
            recency_weight=recency_weight, recency_column=recency_column,
        )

    def _fts_search(self, table: str, column: str, query: str, limit: int) -> list[tuple[int, float]]:
        fts_query = _sanitize_fts_query(query)
        if not fts_query:
            return []

        fts_table = f"{table}_fts_{column}"
        rowid_col = self._get_rowid_ref(table)
        join_col = "m.rowid" if rowid_col == "rowid" else f"m.{rowid_col}"

        try:
            with self._connect() as cur:
                cur.execute(
                    f"SELECT {join_col} as id, bm25({fts_table}) as score "
                    f"FROM {fts_table} fts JOIN {table} m ON {join_col} = fts.rowid "
                    f"WHERE {fts_table} MATCH ? ORDER BY score LIMIT ?",
                    (fts_query, limit),
                )
                rows = cur.fetchall()
                return [(r["id"], r["score"]) for r in rows]
        except Exception:
            try:
                escaped = query.replace("%", "\\%").replace("_", "\\_")
                with self._connect() as cur:
                    cur.execute(
                        f"SELECT {rowid_col} as id, 0.0 as score FROM {table} "
                        f"WHERE {column} LIKE ? ESCAPE '\\' LIMIT ?",
                        (f"%{escaped}%", limit),
                    )
                    rows = cur.fetchall()
                    return [(r["id"], r["score"]) for r in rows]
            except Exception:
                return []

    def _vector_search(
        self, table: str, column: str, query: str,
        limit: int = 10, query_embedding: list[float] | None = None,
    ) -> list[tuple[int, float]]:
        collection_name = f"{table}_{column}"
        try:
            collection = self._get_collection(collection_name)
            embedding = query_embedding if query_embedding is not None else self._get_embedding(query)
            results = collection.query(
                query_embeddings=[embedding], n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
            if not results["ids"] or not results["ids"][0]:
                return []
            out = []
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if "distances" in results else 0
                similarity = max(0.0, 1.0 - distance)
                try:
                    out.append((int(doc_id), similarity))
                except ValueError:
                    out.append((doc_id, similarity))
            return out
        except Exception:
            return []

    @staticmethod
    def _fuse_hybrid(
        fts_results: list[tuple[int, float]],
        vec_results: list[tuple[int, float]],
        fts_weight: float = 0.5,
    ) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for rank, (row_id, _) in enumerate(fts_results):
            scores[row_id] = scores.get(row_id, 0) + fts_weight / (RRF_K + rank + 1)
        for rank, (row_id, _) in enumerate(vec_results):
            scores[row_id] = scores.get(row_id, 0) + (1 - fts_weight) / (RRF_K + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def _compute_recency(ts_str: str | None) -> float:
        if not ts_str:
            return 0.0
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            days_ago = max((datetime.now(UTC) - ts).days, 0)
            return 1.0 / (1 + days_ago / 30)
        except (ValueError, TypeError):
            return 0.0

    def _fetch_rows_by_ids(self, table: str, ids: list[int | str]) -> dict[int | str, dict]:
        if not ids:
            return {}
        with self._connect() as cur:
            lookup_col = self._get_rowid_ref(table, cur=cur)
            placeholders = ",".join("?" * len(ids))
            cur.execute(
                f"SELECT *, {lookup_col} as _lookup_id FROM {table} WHERE {lookup_col} IN ({placeholders})", ids,
            )
            rows = cur.fetchall()
        result = {}
        for r in rows:
            d = dict(r)
            lookup_id = d.pop("_lookup_id", None)
            result[lookup_id] = d
        return result

