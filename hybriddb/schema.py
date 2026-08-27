from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from hybriddb.embedding import EMBEDDING_DIM
from hybriddb.types import Column
from hybriddb.utils import (
    _SYSTEM_TABLES,
    _column_spec,
    _now_iso,
    _validate_identifier,
)
from hybriddb.versioning import (
    GENESIS_HASH,
    HISTORY_SUFFIX,
)

logger = logging.getLogger("hybriddb")

class SchemaMixin:
    def _table_meta(self, table: str, cur: sqlite3.Cursor | None = None) -> dict[str, Any] | None:
        if cur is None:
            with self._connect() as cur:
                cur.execute("SELECT * FROM _schema WHERE table_name = ?", (table,))
                row = cur.fetchone()
        else:
            cur.execute("SELECT * FROM _schema WHERE table_name = ?", (table,))
            row = cur.fetchone()
        if not row:
            return None
        return {
            "table_name": row["table_name"],
            "columns": json.loads(row["columns_json"]),
            "version": row["version"],
            "is_dirty": bool(row["is_dirty"]),
        }

    def _save_table_meta(
        self, cur: sqlite3.Cursor, table: str, columns: dict[str, str], dirty: bool = False
    ) -> None:
        now = _now_iso()
        cur.execute(
            "INSERT OR REPLACE INTO _schema "
            "(table_name, columns_json, version, is_dirty, "
            "embedding_model, embedding_dim, created_at, updated_at) "
            "VALUES (?, ?, "
            "COALESCE((SELECT version FROM _schema WHERE table_name = ?), 0) + 1, "
            "?, ?, ?, ?, ?)",
            (table, json.dumps(columns), table, int(dirty),
             self._embedding_model_name, EMBEDDING_DIM, now, now),
        )

    def _get_text_columns(self, table: str, cur: sqlite3.Cursor | None = None) -> list[str]:
        meta = self._table_meta(table, cur=cur)
        if not meta:
            return []
        return [col for col, ctype in meta["columns"].items() if ctype in ("TEXT", "LONGTEXT")]

    def _get_longtext_columns(self, table: str, cur: sqlite3.Cursor | None = None) -> list[str]:
        meta = self._table_meta(table, cur=cur)
        if not meta:
            return []
        return [col for col, ctype in meta["columns"].items() if ctype == "LONGTEXT"]

    def _has_autoincrement_id(self, table: str, cur: sqlite3.Cursor | None = None) -> bool:
        if cur is None:
            with self._connect() as cur:
                cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name = ?", (table,))
                row = cur.fetchone()
        else:
            cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name = ?", (table,))
            row = cur.fetchone()
        if not row or not row["sql"]:
            return False
        return "INTEGER PRIMARY KEY AUTOINCREMENT" in row["sql"]

    @staticmethod
    def _pk_column_type(table: str, cur: sqlite3.Cursor) -> str | None:
        cur.execute(f"PRAGMA table_info({table})")
        for row in cur.fetchall():
            if row["pk"] > 0:
                return row["type"]
        return None

    def _get_pk_column(self, table: str, cur: sqlite3.Cursor | None = None) -> str:
        if cur is None:
            with self._connect() as cur:
                cur.execute(f"PRAGMA table_info({table})")
                rows = cur.fetchall()
        else:
            cur.execute(f"PRAGMA table_info({table})")
            rows = cur.fetchall()
        for row in rows:
            if row["pk"] > 0:
                return row["name"]
        return "rowid"

    def _get_rowid_ref(self, table: str, cur: sqlite3.Cursor | None = None) -> str:
        if cur is None:
            with self._connect() as cur:
                cur.execute(f"PRAGMA table_info({table})")
                rows = cur.fetchall()
        else:
            cur.execute(f"PRAGMA table_info({table})")
            rows = cur.fetchall()
        for row in rows:
            if row["pk"] > 0 and "INT" in (row["type"] or "").upper():
                return row["name"]
        return "rowid"

    @staticmethod
    def _resolve_internal_rowid(cur: sqlite3.Cursor, table: str, user_pk: int | str, pk_col: str = "id") -> int | None:
        row = cur.execute(f"SELECT rowid FROM {table} WHERE {pk_col} = ?", (user_pk,)).fetchone()
        return row[0] if row else None

    def _create_fts5(
        self, cur: sqlite3.Cursor, table: str, col: str, rowid_col: str | None = None
    ) -> None:
        fts_name = f"{table}_fts_{col}"
        if rowid_col is None:
            rowid_col = self._get_rowid_ref(table, cur=cur)
        rowid_ref = f"new.{rowid_col}"
        old_rowid_ref = f"old.{rowid_col}"
        content_rowid = rowid_col

        cur.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {fts_name} USING fts5("
            f"{col}, content='{table}', content_rowid='{content_rowid}')"
        )
        cur.execute(
            f"CREATE TRIGGER IF NOT EXISTS {table}_ai_{col} AFTER INSERT ON {table} BEGIN "
            f"INSERT INTO {fts_name}(rowid, {col}) VALUES ({rowid_ref}, new.{col}); END"
        )
        cur.execute(
            f"CREATE TRIGGER IF NOT EXISTS {table}_ad_{col} AFTER DELETE ON {table} BEGIN "
            f"INSERT INTO {fts_name}({fts_name}, rowid, {col}) "
            f"VALUES ('delete', {old_rowid_ref}, old.{col}); END"
        )
        cur.execute(
            f"CREATE TRIGGER IF NOT EXISTS {table}_au_{col} AFTER UPDATE ON {table} BEGIN "
            f"INSERT INTO {fts_name}({fts_name}, rowid, {col}) "
            f"VALUES ('delete', {old_rowid_ref}, old.{col}); "
            f"INSERT INTO {fts_name}(rowid, {col}) VALUES ({rowid_ref}, new.{col}); END"
        )
        # Backfill: external-content FTS5 tables start empty; triggers only
        # index future writes. Without this, keyword search silently returns
        # nothing for existing rows after any index rebuild (reindex,
        # drop_column, rename_column, import_sql).
        cur.execute(f"INSERT INTO {fts_name}({fts_name}) VALUES('rebuild')")

    def _drop_fts5(self, cur: sqlite3.Cursor, table: str, col: str) -> None:
        fts_name = f"{table}_fts_{col}"
        cur.execute(f"DROP TABLE IF EXISTS {fts_name}")
        for suffix in ("ai", "ad", "au"):
            cur.execute(f"DROP TRIGGER IF EXISTS {table}_{suffix}_{col}")

    def _rebuild_all_fts5(self, cur: sqlite3.Cursor, table: str) -> None:
        meta = self._table_meta(table, cur=cur)
        if not meta:
            return
        rowid_col = self._get_rowid_ref(table, cur=cur)
        for col in self._get_text_columns(table, cur=cur):
            self._drop_fts5(cur, table, col)
            self._create_fts5(cur, table, col, rowid_col)

    def create_table(self, table: str, columns: dict[str, str | Column],
                     versioned: bool = False, hash_chain: bool = True) -> None:
        _validate_identifier(table, "table")
        if "_fts_" in table:
            raise ValueError(
                f"Table name '{table}' contains '_fts_' which conflicts with FTS5 naming convention"
            )
        if table.endswith(HISTORY_SUFFIX):
            raise ValueError(
                f"Table name '{table}' ends with '{HISTORY_SUFFIX}' which is reserved "
                f"for versioned-table history"
            )
        col_defs: list[str] = []
        parsed: dict[str, str] = {}
        has_custom_pk = any("PRIMARY KEY" in _column_spec(spec).upper() for name, spec in columns.items())
        if not has_custom_pk:
            if "id" in columns:
                raise ValueError(
                    f"Column 'id' conflicts with the implicit primary key — declare it as "
                    f"'id ... PRIMARY KEY' or rename the column"
                )
            col_defs.append("id INTEGER PRIMARY KEY AUTOINCREMENT")

        for col_name, col_spec in columns.items():
            _validate_identifier(col_name, "column")
            parts = _column_spec(col_spec).split()
            base_type = parts[0].upper()
            extras = " ".join(parts[1:]) if len(parts) > 1 else ""
            is_pk = "PRIMARY KEY" in extras.upper()
            if base_type == "TEXT":
                col_defs.append(f"{col_name} TEXT{' ' + extras if extras else ''}")
                parsed[col_name] = "TEXT" if not is_pk else "TEXT_PK"
            elif base_type == "LONGTEXT":
                col_defs.append(f"{col_name} TEXT{' ' + extras if extras else ''}")
                parsed[col_name] = "LONGTEXT"
            elif base_type == "INTEGER":
                col_defs.append(f"{col_name} INTEGER{' ' + extras if extras else ''}")
                parsed[col_name] = "INTEGER" if not is_pk else "INTEGER_PK"
            elif base_type == "REAL":
                col_defs.append(f"{col_name} REAL{' ' + extras if extras else ''}")
                parsed[col_name] = "REAL"
            elif base_type == "BOOLEAN":
                col_defs.append(f"{col_name} INTEGER{' ' + extras if extras else ''}")
                parsed[col_name] = "BOOLEAN"
            elif base_type == "JSON":
                col_defs.append(f"{col_name} TEXT{' ' + extras if extras else ''}")
                parsed[col_name] = "JSON"
            else:
                col_defs.append(f"{col_name} TEXT{' ' + extras if extras else ''}")
                parsed[col_name] = base_type

        with self._connect() as cur:
            cur.execute(f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(col_defs)})")
            existing_meta = self._table_meta(table)
            existing_cols: dict[str, str] = existing_meta["columns"] if existing_meta else {}
            cur.execute(f"PRAGMA table_info({table})")
            actual_cols = {row["name"] for row in cur.fetchall()}
            for col_name, col_parsed_type in parsed.items():
                if col_name in existing_cols or col_name in actual_cols:
                    continue
                sqlite_type = (
                    "INTEGER" if col_parsed_type == "BOOLEAN"
                    else "TEXT" if col_parsed_type in ("LONGTEXT", "JSON")
                    else col_parsed_type.replace("_PK", "")
                )
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {sqlite_type}")
                logger.info("migrate_column_added table=%s column=%s type=%s", table, col_name, col_parsed_type)

            text_cols = [c for c, t in parsed.items() if t in ("TEXT", "LONGTEXT")]
            rowid_col = self._get_rowid_ref(table, cur=cur)
            for col in text_cols:
                self._create_fts5(cur, table, col, rowid_col)
            for col in self._get_longtext_columns_from_parsed(parsed):
                self._get_collection(f"{table}_{col}")
            self._save_table_meta(cur, table, parsed)
            if versioned:
                self._enable_versioning(cur, table, hash_chain)

        self._refresh_duckdb_table_if_registered(table)

    def _enable_versioning(self, cur, table: str, hash_chain: bool) -> None:
        """Register a table as versioned: create the shadow history table and
        backfill existing rows as insert events (so as_of/history cover rows
        that pre-date versioning)."""
        cur.execute(
            "INSERT OR IGNORE INTO _versioned_tables (table_name, hash_chain, created_at) "
            "VALUES (?, ?, ?)",
            (table, int(hash_chain), _now_iso()),
        )
        self._create_history_table(cur, table)
        # backfill: history for rows that pre-date this call
        existing = cur.execute(f"SELECT * FROM {table}").fetchall()
        if not existing:
            return
        pk_col = self._get_pk_column(table, cur=cur)
        prev = GENESIS_HASH
        for r in existing:
            row = dict(r)
            prev = self._capture_history(
                cur, table, "insert", row[pk_col], row,
                hash_chain=hash_chain, prev_hash=prev,
            )

    @staticmethod
    def _get_longtext_columns_from_parsed(parsed: dict[str, str]) -> list[str]:
        return [col for col, ctype in parsed.items() if ctype == "LONGTEXT"]

    def add_column(self, table: str, column: str, col_type: str) -> None:
        _validate_identifier(table, "table")
        _validate_identifier(column, "column")
        if self.is_versioned(table):
            raise ValueError(
                f"Table '{table}' is versioned; schema changes on versioned "
                f"tables are not supported (disable versioning first)"
            )
        meta = self._table_meta(table)
        if not meta:
            raise ValueError(f"Table '{table}' not found")
        base_type = col_type.split()[0].upper()
        sqlite_type = {
            "LONGTEXT": "TEXT", "BOOLEAN": "INTEGER", "JSON": "TEXT",
        }.get(base_type, base_type)
        extras = " ".join(col_type.split()[1:]) if len(col_type.split()) > 1 else ""
        col_def = f"{column} {sqlite_type}{' ' + extras if extras else ''}"

        with self._connect() as cur:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
            new_columns = dict(meta["columns"])
            new_columns[column] = base_type
            if base_type in ("TEXT", "LONGTEXT"):
                self._create_fts5(cur, table, column, self._get_rowid_ref(table, cur=cur))
            if base_type == "LONGTEXT":
                self._get_collection(f"{table}_{column}")
            self._save_table_meta(cur, table, new_columns, dirty=(base_type == "LONGTEXT"))
        self._refresh_duckdb_table_if_registered(table)

    def drop_column(self, table: str, column: str) -> None:
        _validate_identifier(table, "table")
        _validate_identifier(column, "column")
        if self.is_versioned(table):
            raise ValueError(
                f"Table '{table}' is versioned; schema changes on versioned "
                f"tables are not supported (disable versioning first)"
            )
        meta = self._table_meta(table)
        if not meta or column not in meta["columns"]:
            raise ValueError(f"Column '{column}' not found in table '{table}'")
        with self._connect() as cur:
            pk_col = self._get_pk_column(table, cur=cur)
            if column == pk_col:
                raise ValueError(
                    f"Cannot drop primary key column '{column}' from table '{table}'"
                )
        col_type = meta["columns"][column]
        old_columns = {k: v for k, v in meta["columns"].items() if k != column}

        with self._connect() as cur:
            pk_col = self._get_pk_column(table, cur=cur)
            pk_ctype = old_columns.get(pk_col, "INTEGER_PK")
            pk_base = pk_ctype.replace("_PK", "").replace("_PK", "")
            pk_sql_type = {"LONGTEXT": "TEXT", "BOOLEAN": "INTEGER", "JSON": "TEXT"}.get(pk_base, pk_base)
            old_table = f"_{table}_old"
            cur.execute(f"ALTER TABLE {table} RENAME TO {old_table}")
            pk_extras = "PRIMARY KEY AUTOINCREMENT" if pk_sql_type == "INTEGER" else "PRIMARY KEY"
            new_col_defs = [f"{pk_col} {pk_sql_type} {pk_extras}"]
            for cname, ctype in old_columns.items():
                if cname == pk_col:
                    continue
                sqlite_type = {"LONGTEXT": "TEXT", "BOOLEAN": "INTEGER", "JSON": "TEXT"}.get(ctype, ctype)
                new_col_defs.append(f"{cname} {sqlite_type}")
            cur.execute(f"CREATE TABLE {table} ({', '.join(new_col_defs)})")
            shared = [c for c in old_columns if c in meta["columns"]]
            col_list = ", ".join(shared)
            cur.execute(f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM {old_table}")
            cur.execute(f"DROP TABLE {old_table}")
            self._save_table_meta(cur, table, old_columns, dirty=True)
            if col_type in ("TEXT", "LONGTEXT"):
                self._drop_fts5(cur, table, column)
            self._rebuild_all_fts5(cur, table)
            if col_type == "LONGTEXT" and self._chroma is not None:
                try:
                    self._chroma.delete_collection(f"{table}_{column}")
                except Exception:
                    pass
            for lt_col in self._get_longtext_columns_from_parsed(old_columns):
                now = _now_iso()
                cur.execute(
                    "INSERT INTO _journal (app_table, row_id, column_name, op, created_at) "
                    "VALUES (?, NULL, ?, 'meta_update', ?)",
                    (table, lt_col, now),
                )
        self._refresh_duckdb_table_if_registered(table)

    def rename_column(self, table: str, old_name: str, new_name: str) -> None:
        _validate_identifier(table, "table")
        _validate_identifier(old_name, "column")
        _validate_identifier(new_name, "column")
        if self.is_versioned(table):
            raise ValueError(
                f"Table '{table}' is versioned; schema changes on versioned "
                f"tables are not supported (disable versioning first)"
            )
        meta = self._table_meta(table)
        if not meta or old_name not in meta["columns"]:
            raise ValueError(f"Column '{old_name}' not found in table '{table}'")
        col_type = meta["columns"][old_name]
        new_columns = {}
        for k, v in meta["columns"].items():
            new_columns[new_name if k == old_name else k] = v

        with self._connect() as cur:
            cur.execute(f"ALTER TABLE {table} RENAME COLUMN {old_name} TO {new_name}")
            if col_type in ("TEXT", "LONGTEXT"):
                self._drop_fts5(cur, table, old_name)
                self._create_fts5(cur, table, new_name, self._get_rowid_ref(table, cur=cur))
            if col_type == "LONGTEXT" and self._chroma is not None:
                try:
                    old_coll = self._get_collection(f"{table}_{old_name}")
                    if old_coll is not None:
                        all_data = old_coll.get(include=["embeddings", "documents", "metadatas"])
                        if all_data.get("ids"):
                            new_coll = self._get_collection(f"{table}_{new_name}")
                            new_coll.upsert(
                                ids=all_data["ids"], embeddings=all_data["embeddings"],
                                documents=all_data["documents"], metadatas=all_data.get("metadatas"),
                            )
                        self._chroma.delete_collection(f"{table}_{old_name}")
                except Exception as e:
                    logger.warning("rename_chroma_failed table=%s old=%s new=%s error=%s", table, old_name, new_name, e)
            for lt_col in self._get_longtext_columns_from_parsed(new_columns):
                now = _now_iso()
                cur.execute(
                    "INSERT INTO _journal (app_table, row_id, column_name, op, created_at) "
                    "VALUES (?, NULL, ?, 'meta_update', ?)",
                    (table, lt_col, now),
                )
            self._save_table_meta(cur, table, new_columns, dirty=True)
        self._refresh_duckdb_table_if_registered(table)

    def list_tables(self) -> list[str]:
        with self._connect() as cur:
            cur.execute("SELECT table_name FROM _schema ORDER BY table_name")
            tables = [row["table_name"] for row in cur.fetchall()]
        return [t for t in tables if t not in _SYSTEM_TABLES and not t.startswith("_")]

    def get_schema(self, table: str) -> dict[str, str]:
        _validate_identifier(table, "table")
        meta = self._table_meta(table)
        if not meta:
            return {}
        return meta["columns"]

