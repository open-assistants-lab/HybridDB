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

class AnalyticsMixin:
    def _init_duckdb(self) -> None:
        self._duckdb_path = ""
        self._duckdb_synced_tables: dict[str, dict] = {}
        self._duckdb_conn = None

        try:
            import duckdb

            self._duckdb_path = str((self.path / "analytics.duckdb").resolve())
            self._duckdb_conn = duckdb.connect(self._duckdb_path)
            self._duckdb_conn.execute("SET threads = 4")
            self._duckdb_conn.execute("""
                CREATE TABLE IF NOT EXISTS _duckdb_sync (
                    table_name TEXT PRIMARY KEY,
                    synced_count INTEGER DEFAULT 0
                )
            """)
            existing = self._duckdb_conn.execute(
                "SELECT table_name FROM _duckdb_sync"
            ).fetchall()
            for (tname,) in existing:
                quoted_tname = self._duckdb_quote_identifier(tname)
                count = self._duckdb_conn.execute(
                    f"SELECT count(*) FROM {quoted_tname}"
                ).fetchone()[0]
                cols_info = self._duckdb_conn.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_name = ?",
                    (tname,),
                ).fetchall()
                columns = {c: t for c, t in cols_info}
                self._duckdb_synced_tables[tname] = {"columns": columns, "count": count}
        except ImportError:
            pass
        except Exception as e:
            logger.warning("duckdb.init_failed error=%s", e)
            self._duckdb_path = ""
            self._duckdb_conn = None

    def _auto_register_duckdb_tables(self) -> None:
        if not self._duckdb_path:
            return
        app_tables = self.list_tables()
        new_tables = [t for t in app_tables if t not in self._duckdb_synced_tables]
        if not new_tables:
            return
        for table in new_tables:
            self.register_duckdb_table(table)
        logger.info("duckdb.auto_registered tables=%s count=%d", new_tables, len(new_tables))

    def register_duckdb_table(self, table: str) -> bool:
        _validate_identifier(table, "table")
        if not self._duckdb_path or self._duckdb_conn is None:
            return False

        meta = self._table_meta(table)
        if not meta:
            logger.warning("duckdb.register_missing_table table=%s", table)
            return False

        cols = []
        for col_name, col_type in meta["columns"].items():
            base = col_type.replace("_PK", "")
            quoted_col = self._duckdb_quote_identifier(col_name)
            if base == "BOOLEAN":
                cols.append(f"{quoted_col} INTEGER")
            elif base == "JSON":
                cols.append(f"{quoted_col} TEXT")
            elif base in ("TEXT", "LONGTEXT"):
                cols.append(f"{quoted_col} TEXT")
            elif base == "INTEGER" or base == "INTEGER_PK":
                cols.append(f"{quoted_col} BIGINT")
            else:
                cols.append(f"{quoted_col} {base}")
        if "id" not in meta["columns"]:
            cols.insert(0, "id BIGINT")

        quoted_table = self._duckdb_quote_identifier(table)
        with self._db_lock:
            dk = self._duckdb_conn
            dk.execute(f"DROP TABLE IF EXISTS {quoted_table}")
            dk.execute(f"CREATE TABLE {quoted_table} ({', '.join(cols)})")
            dk.execute(
                "INSERT OR REPLACE INTO _duckdb_sync (table_name, synced_count) VALUES (?, 0)",
                (table,),
            )

        self._duckdb_synced_tables[table] = {"columns": dict(meta["columns"]), "count": 0}
        self._full_sync_duckdb_table(table)
        return True

    @staticmethod
    def _duckdb_quote_identifier(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    def _refresh_duckdb_table_if_registered(self, table: str) -> None:
        if table in self._duckdb_synced_tables:
            self.register_duckdb_table(table)

    def _full_sync_duckdb_table(self, table: str) -> None:
        if not self._duckdb_path or table not in self._duckdb_synced_tables:
            return

        with self._db_lock:
            dk = self._duckdb_conn
            quoted_table = self._duckdb_quote_identifier(table)
            dk.execute(f"DELETE FROM {quoted_table}")
            try:
                dk.execute("DETACH src")
            except Exception:
                pass
            try:
                dk.execute(f"ATTACH '{self._db_path}' AS src (TYPE sqlite)")
                dk.execute(f"INSERT INTO {quoted_table} SELECT * FROM src.{quoted_table}")
            finally:
                dk.execute("DETACH src")
            count = dk.execute(f"SELECT count(*) FROM {quoted_table}").fetchone()[0]
            dk.execute(
                "UPDATE _duckdb_sync SET synced_count = ? WHERE table_name = ?",
                (count, table),
            )
            self._duckdb_synced_tables[table]["count"] = count

    def unregister_duckdb_table(self, table: str) -> bool:
        _validate_identifier(table, "table")
        if table not in self._duckdb_synced_tables:
            return False

        with self._db_lock:
            dk = self._duckdb_conn
            dk.execute(f"DROP TABLE IF EXISTS {self._duckdb_quote_identifier(table)}")
            dk.execute("DELETE FROM _duckdb_sync WHERE table_name = ?", (table,))
        self._duckdb_synced_tables.pop(table, None)
        return True

    def analytics(self, sql: str) -> list[dict]:
        if not self._duckdb_path:
            raise RuntimeError(
                "DuckDB analytics not available — DuckDB initialization failed "
                "or module not installed"
            )
        with self._db_lock:
            dk = self._duckdb_conn
            result = dk.execute(sql)
            columns = [desc[0] for desc in result.description]
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
        return rows

    def _sync_duckdb_from_journal(self, entries: list[dict]) -> None:
        if not self._duckdb_path or not self._duckdb_synced_tables:
            return

        by_table: dict[str, dict[str, list[int]]] = {}
        seen_ids: set[int] = set()

        for e in entries:
            if e["id"] in seen_ids:
                continue
            seen_ids.add(e["id"])
            tbl = e["app_table"]
            if tbl not in self._duckdb_synced_tables:
                continue
            if tbl not in by_table:
                by_table[tbl] = {"add": [], "delete": []}
            row_id = e["row_id"]
            if row_id is not None:
                by_table[tbl]["delete"].append(row_id)
                if e["op"] != "row_delete":
                    by_table[tbl]["add"].append(row_id)

        if not by_table:
            return

        with self._db_lock:
            dk = self._duckdb_conn
            dk.execute(f"ATTACH '{self._db_path}' AS src (TYPE sqlite)")
            for tbl, ops in by_table.items():
                quoted_tbl = self._duckdb_quote_identifier(tbl)
                if ops["delete"]:
                    ids = ",".join(str(i) for i in ops["delete"])
                    dk.execute(f"DELETE FROM {quoted_tbl} WHERE id IN ({ids})")
                if ops["add"]:
                    ids = ",".join(str(i) for i in ops["add"])
                    dk.execute(
                        f"INSERT INTO {quoted_tbl} SELECT * FROM src.{quoted_tbl} WHERE id IN ({ids})"
                    )
                count = dk.execute(f"SELECT count(*) FROM {quoted_tbl}").fetchone()[0]
                dk.execute(
                    "UPDATE _duckdb_sync SET synced_count = ? WHERE table_name = ?",
                    (count, tbl),
                )
                self._duckdb_synced_tables[tbl]["count"] = count
            dk.execute("DETACH src")

    def sync_duckdb_table(self, table: str) -> None:
        _validate_identifier(table, "table")
        self._full_sync_duckdb_table(table)

