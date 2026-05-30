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

class CrudMixin:
    def _row_to_metadata(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        meta = self._table_meta(table)
        if not meta:
            return {}
        result = {}
        for col, ctype in meta["columns"].items():
            if ctype in ("LONGTEXT", "JSON"):
                continue
            val = row.get(col)
            if val is None:
                continue
            base = ctype.replace("_PK", "")
            if base == "BOOLEAN":
                result[col] = bool(val)
            else:
                result[col] = val
        return result

    def insert(
        self, table: str, data: dict, sync: bool = True,
        skip_journal_columns: set[str] | None = None,
    ) -> int | str:
        _validate_identifier(table, "table")
        meta = self._table_meta(table)
        if not meta:
            raise ValueError(f"Table '{table}' not found")
        filtered = {k: v for k, v in data.items() if k in meta["columns"]}
        columns = list(filtered.keys())
        placeholders = ", ".join("?" * len(columns))
        values = list(filtered.values())

        with self._connect() as cur:
            cur.execute(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})", values)
            internal_rowid = cur.lastrowid
            has_auto_id = self._has_autoincrement_id(table)
            if has_auto_id:
                user_pk = internal_rowid
                row = dict(cur.execute(f"SELECT * FROM {table} WHERE id = ?", (user_pk,)).fetchone())
            elif "id" in filtered:
                user_pk = filtered["id"]
                row = dict(cur.execute(f"SELECT * FROM {table} WHERE id = ?", (user_pk,)).fetchone())
            else:
                user_pk = internal_rowid
                row = dict(cur.execute(f"SELECT * FROM {table} WHERE rowid = ?", (internal_rowid,)).fetchone())

            metadata = self._row_to_metadata(table, row)
            now = _now_iso()
            for col in self._get_longtext_columns(table):
                if skip_journal_columns and col in skip_journal_columns:
                    continue
                cur.execute(
                    "INSERT INTO _journal (app_table, row_id, column_name, op, data, metadata, created_at) "
                    "VALUES (?, ?, ?, 'add', ?, ?, ?)",
                    (table, internal_rowid, col, row.get(col, ""), json.dumps(metadata), now),
                )
            cur.execute(
                "INSERT INTO _journal (app_table, row_id, op, data, created_at) "
                "VALUES (?, ?, 'row_add', ?, ?)",
                (table, internal_rowid, json.dumps(dict(row), default=str), now),
            )
        if sync:
            self._process_journal()
        return user_pk or 0

    def row_to_metadata(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        _validate_identifier(table, "table")
        return self._row_to_metadata(table, row)

    def vector_upsert(
        self, collection_name: str, row_id: int | str, document: str,
        embedding: list[float], metadata: dict[str, Any] | None = None,
    ) -> bool:
        _validate_identifier(collection_name, "collection")
        collection = self._get_collection(collection_name)
        if collection is None:
            return False
        collection.upsert(
            ids=[str(row_id)], embeddings=[embedding], documents=[document],
            metadatas=[metadata or {}],
        )
        return True

    def insert_batch(self, table: str, rows: list[dict], sync: bool = True) -> list[int | str]:
        _validate_identifier(table, "table")
        if len(rows) > JOURNAL_CAP:
            logger.warning("insert_batch.large_batch table=%s rows=%d limit=%d", table, len(rows), JOURNAL_CAP)
        meta = self._table_meta(table)
        if not meta:
            raise ValueError(f"Table '{table}' not found")
        ids: list[int | str] = []
        with self._connect() as cur:
            now = _now_iso()
            for data in rows:
                filtered = {k: v for k, v in data.items() if k in meta["columns"]}
                columns = list(filtered.keys())
                placeholders = ", ".join("?" * len(columns))
                values = list(filtered.values())
                cur.execute(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})", values)
                internal_rowid = cur.lastrowid
                has_auto_id = self._has_autoincrement_id(table)
                if has_auto_id:
                    user_pk = internal_rowid
                    row = dict(cur.execute(f"SELECT * FROM {table} WHERE id = ?", (user_pk,)).fetchone())
                elif "id" in filtered:
                    user_pk = filtered["id"]
                    row = dict(cur.execute(f"SELECT * FROM {table} WHERE id = ?", (user_pk,)).fetchone())
                else:
                    assert internal_rowid is not None
                    user_pk = internal_rowid
                    row = dict(cur.execute(f"SELECT * FROM {table} WHERE rowid = ?", (internal_rowid,)).fetchone())
                ids.append(user_pk)
                metadata = self._row_to_metadata(table, row)
                for col in self._get_longtext_columns(table):
                    cur.execute(
                        "INSERT INTO _journal (app_table, row_id, column_name, op, data, metadata, created_at) "
                        "VALUES (?, ?, ?, 'add', ?, ?, ?)",
                        (table, internal_rowid, col, row.get(col, ""), json.dumps(metadata), now),
                    )
                cur.execute(
                    "INSERT INTO _journal (app_table, row_id, op, data, created_at) "
                    "VALUES (?, ?, 'row_add', ?, ?)",
                    (table, internal_rowid, json.dumps(dict(row), default=str), now),
                )
        if sync:
            self._process_journal()
        return ids

    def update(self, table: str, row_id: int | str, data: dict, sync: bool = True) -> bool:
        _validate_identifier(table, "table")
        meta = self._table_meta(table)
        if not meta:
            raise ValueError(f"Table '{table}' not found")
        filtered = {k: v for k, v in data.items() if k in meta["columns"]}
        if not filtered:
            return False

        with self._connect() as cur:
            internal_rowid = self._resolve_internal_rowid(cur, table, row_id)
            if internal_rowid is None:
                return False
            set_clause = ", ".join(f"{k} = ?" for k in filtered.keys())
            cur.execute(f"UPDATE {table} SET {set_clause} WHERE id = ?", list(filtered.values()) + [row_id])
            if cur.rowcount == 0:
                return False
            row = dict(cur.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone())
            metadata = self._row_to_metadata(table, row)
            for col in self._get_longtext_columns(table):
                now = _now_iso()
                cur.execute(
                    "INSERT INTO _journal (app_table, row_id, column_name, op, data, metadata, created_at) "
                    "VALUES (?, ?, ?, 'update', ?, ?, ?)",
                    (table, internal_rowid, col, row.get(col, ""), json.dumps(metadata), now),
                )
            now = _now_iso()
            cur.execute(
                "INSERT INTO _journal (app_table, row_id, op, data, created_at) "
                "VALUES (?, ?, 'row_update', ?, ?)",
                (table, internal_rowid, json.dumps(dict(row), default=str), now),
            )
        if sync:
            self._process_journal()
        return True

    def delete(self, table: str, row_id: int | str, sync: bool = True) -> bool:
        _validate_identifier(table, "table")
        with self._connect() as cur:
            internal_rowid = self._resolve_internal_rowid(cur, table, row_id)
            if internal_rowid is None:
                return False
            cur.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
            if cur.rowcount == 0:
                return False
            for col in self._get_longtext_columns(table):
                now = _now_iso()
                cur.execute(
                    "INSERT INTO _journal (app_table, row_id, column_name, op, created_at) "
                    "VALUES (?, ?, ?, 'delete', ?)",
                    (table, internal_rowid, col, now),
                )
            now = _now_iso()
            cur.execute(
                "INSERT INTO _journal (app_table, row_id, op, created_at) "
                "VALUES (?, ?, 'row_delete', ?)",
                (table, internal_rowid, now),
            )
        if sync:
            self._process_journal()
        return True

    def get(self, table: str, row_id: int | str) -> dict | None:
        _validate_identifier(table, "table")
        with self._connect() as cur:
            cur.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,))
            row = cur.fetchone()
        if not row:
            return None
        return dict(row)

    def query(
        self, table: str, where: str = "", params: tuple = (),
        order_by: str = "", limit: int = 100,
    ) -> list[dict]:
        _validate_identifier(table, "table")
        _validate_order_by(order_by)
        sql = f"SELECT * FROM {table}"
        if where:
            sql += f" WHERE {where}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        sql += f" LIMIT {limit}"
        with self._connect() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def raw_query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._connect() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def read_query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Run a read-only SQL query against SQLite."""
        first = sql.lstrip().split(None, 1)[0].upper() if sql.strip() else ""
        if first not in {"SELECT", "WITH", "PRAGMA", "EXPLAIN"}:
            raise ValueError("read_query only accepts read-only SQL")
        return self.raw_query(sql, params)

    def count(self, table: str, where: str = "", params: tuple = ()) -> int:
        _validate_identifier(table, "table")
        sql = f"SELECT COUNT(*) FROM {table}"
        if where:
            sql += f" WHERE {where}"
        with self._connect() as cur:
            result = cur.execute(sql, params).fetchone()
        return result[0] if result else 0

