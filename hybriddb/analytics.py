from __future__ import annotations

import logging

from hybriddb.utils import (
    _is_safe_identifier,
    _validate_identifier,
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
                if not _is_safe_identifier(tname):
                    continue
                quoted_tname = self._duckdb_quote_identifier(tname)
                try:
                    count = self._duckdb_conn.execute(
                        f"SELECT count(*) FROM {quoted_tname}"
                    ).fetchone()[0]
                    cols_info = self._duckdb_conn.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = ?",
                        (tname,),
                    ).fetchall()
                    actual_cols = {c[0] for c in cols_info}
                    expected_cols = {name for name, _ in self._duckdb_table_layout(tname)[0]}
                    if actual_cols != expected_cols:
                        # Stale/broken mirror from an older version (e.g. a
                        # synthetic "id" column for custom-PK tables) — drop it;
                        # _auto_register_duckdb_tables recreates it correctly.
                        self._duckdb_conn.execute(f"DROP TABLE IF EXISTS {quoted_tname}")
                        self._duckdb_conn.execute(
                            "DELETE FROM _duckdb_sync WHERE table_name = ?", (tname,)
                        )
                        continue
                    pk_col, pk_is_int = self._duckdb_pk_info(tname)
                    self._duckdb_synced_tables[tname] = {
                        "columns": {c: "" for c in actual_cols},
                        "pk_column": pk_col,
                        "pk_is_int": pk_is_int,
                        "count": count,
                    }
                except Exception:
                    # Broken mirror entry (e.g. table missing) — drop it so
                    # _auto_register_duckdb_tables can recreate it.
                    try:
                        self._duckdb_conn.execute(f"DROP TABLE IF EXISTS {quoted_tname}")
                        self._duckdb_conn.execute(
                            "DELETE FROM _duckdb_sync WHERE table_name = ?", (tname,)
                        )
                    except Exception:
                        pass
                    continue
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

    def _duckdb_pk_info(self, table: str) -> tuple[str, bool]:
        """Return (pk_column, pk_is_int) from the SQLite schema."""
        pk_col = self._get_pk_column(table)
        if pk_col == "rowid":
            return pk_col, True
        with self._connect() as cur:
            cur.execute(f"PRAGMA table_info({table})")
            for row in cur.fetchall():
                if row["pk"] > 0:
                    return pk_col, "INT" in (row["type"] or "").upper()
        return pk_col, True

    def _duckdb_table_layout(self, table: str) -> tuple[list[tuple[str, str]], str, bool]:
        """Return (duckdb_columns, pk_column, pk_is_int) mirroring the SQLite table.

        The DuckDB table mirrors the real SQLite columns — including the
        actual primary key column under its real name. Older versions
        injected a synthetic ``id BIGINT`` column for custom-PK tables,
        which broke full syncs and crashed ``HybridDB.__init__`` on reopen.
        """
        meta = self._table_meta(table)
        meta_cols = (meta or {}).get("columns", {})
        pk_col, pk_is_int = self._duckdb_pk_info(table)
        with self._connect() as cur:
            cur.execute(f"PRAGMA table_info({table})")
            sqlite_cols = [(row["name"], row["type"]) for row in cur.fetchall()]
        type_map = {
            "BOOLEAN": "INTEGER", "JSON": "TEXT", "TEXT": "TEXT",
            "LONGTEXT": "TEXT", "INTEGER": "BIGINT", "REAL": "DOUBLE",
        }
        duck_cols: list[tuple[str, str]] = []
        for name, sqlite_type in sqlite_cols:
            mtype = meta_cols.get(name)
            if mtype:
                dtype = type_map.get(mtype.replace("_PK", ""), "TEXT")
            elif "INT" in (sqlite_type or "").upper():
                dtype = "BIGINT"
            else:
                dtype = "TEXT"
            duck_cols.append((name, dtype))
        return duck_cols, pk_col, pk_is_int

    def register_duckdb_table(self, table: str) -> bool:
        _validate_identifier(table, "table")
        if not self._duckdb_path or self._duckdb_conn is None:
            return False

        meta = self._table_meta(table)
        if not meta:
            logger.warning("duckdb.register_missing_table table=%s", table)
            return False

        duck_cols, pk_col, pk_is_int = self._duckdb_table_layout(table)
        col_defs = ", ".join(
            f"{self._duckdb_quote_identifier(name)} {dtype}" for name, dtype in duck_cols
        )
        quoted_table = self._duckdb_quote_identifier(table)
        with self._db_lock:
            dk = self._duckdb_conn
            dk.execute(f"DROP TABLE IF EXISTS {quoted_table}")
            dk.execute(f"CREATE TABLE {quoted_table} ({col_defs})")
            dk.execute(
                "INSERT OR REPLACE INTO _duckdb_sync (table_name, synced_count) VALUES (?, 0)",
                (table,),
            )

        self._duckdb_synced_tables[table] = {
            "columns": dict(meta["columns"]),
            "pk_column": pk_col,
            "pk_is_int": pk_is_int,
            "count": 0,
        }
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
                col_list = ", ".join(
                    self._duckdb_quote_identifier(name) for name, _ in self._duckdb_table_layout(table)[0]
                )
                dk.execute(
                    f"INSERT INTO {quoted_table} ({col_list}) SELECT {col_list} FROM src.{quoted_table}"
                )
            finally:
                try:
                    dk.execute("DETACH src")
                except Exception:
                    pass
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

        # Collect rowids (int) for add/update entries (row still exists in SQLite)
        # and app_ids (str) for delete entries (row is gone, id stored in data field)
        by_table: dict[str, dict[str, list[int | str]]] = {}
        seen_ids: set[int] = set()

        for e in entries:
            if e["id"] in seen_ids:
                continue
            seen_ids.add(e["id"])
            tbl = e["app_table"]
            if tbl not in self._duckdb_synced_tables:
                continue
            if tbl not in by_table:
                by_table[tbl] = {"add": [], "delete": [], "delete_ids": []}
            if e["op"] == "row_delete":
                app_id = e.get("data")
                if app_id is not None:
                    by_table[tbl]["delete_ids"].append(app_id)
            elif e.get("row_id") is not None:
                by_table[tbl]["delete"].append(e["row_id"])
                by_table[tbl]["add"].append(e["row_id"])

        if not by_table:
            return

        # Resolve rowids to actual app-level ids for add entries
        by_table_ids: dict[str, dict[str, list[int | str]]] = {}
        for tbl, ops in by_table.items():
            info = self._duckdb_synced_tables.get(tbl, {})
            pk_col = info.get("pk_column") or self._get_pk_column(tbl)
            pk_is_int = info.get("pk_is_int", True)
            quoted_pk = self._duckdb_quote_identifier(pk_col)
            by_table_ids[tbl] = {"add": [], "delete": []}
            add_rowids = ops.get("add", [])
            if add_rowids:
                quoted_tbl = self._duckdb_quote_identifier(tbl)
                with self._connect() as cur:
                    placeholders = ",".join("?" * len(add_rowids))
                    rows = cur.execute(
                        f"SELECT rowid, {quoted_pk} FROM {tbl} WHERE rowid IN ({placeholders})",
                        add_rowids,
                    ).fetchall()
                rid_to_id = {r[0]: r[1] for r in rows}
                by_table_ids[tbl]["add"] = [rid_to_id[rid] for rid in add_rowids if rid in rid_to_id]
                by_table_ids[tbl]["delete"] = [rid_to_id[rid] for rid in ops["delete"] if rid in rid_to_id]
            # row_delete entries carry the app-level id (as str) in `data`
            for app_id in ops.get("delete_ids", []):
                try:
                    by_table_ids[tbl]["delete"].append(int(app_id) if pk_is_int else app_id)
                except (TypeError, ValueError):
                    by_table_ids[tbl]["delete"].append(app_id)

        if not any(v["add"] or v["delete"] for v in by_table_ids.values()):
            return

        with self._db_lock:
            dk = self._duckdb_conn
            dk.execute(f"ATTACH '{self._db_path}' AS src (TYPE sqlite)")
            try:
                for tbl, ops in by_table_ids.items():
                    info = self._duckdb_synced_tables.get(tbl, {})
                    pk_col = info.get("pk_column") or self._get_pk_column(tbl)
                    quoted_tbl = self._duckdb_quote_identifier(tbl)
                    quoted_pk = self._duckdb_quote_identifier(pk_col)
                    if ops["delete"]:
                        placeholders = ",".join("?" * len(ops["delete"]))
                        dk.execute(
                            f"DELETE FROM {quoted_tbl} WHERE {quoted_pk} IN ({placeholders})",
                            ops["delete"],
                        )
                    if ops["add"]:
                        placeholders = ",".join("?" * len(ops["add"]))
                        col_list = ", ".join(
                            self._duckdb_quote_identifier(name)
                            for name, _ in self._duckdb_table_layout(tbl)[0]
                        )
                        dk.execute(
                            f"INSERT INTO {quoted_tbl} ({col_list}) "
                            f"SELECT {col_list} FROM src.{quoted_tbl} "
                            f"WHERE {quoted_pk} IN ({placeholders})",
                            ops["add"],
                        )
                    count = dk.execute(f"SELECT count(*) FROM {quoted_tbl}").fetchone()[0]
                    dk.execute(
                        "UPDATE _duckdb_sync SET synced_count = ? WHERE table_name = ?",
                        (count, tbl),
                    )
                    self._duckdb_synced_tables[tbl]["count"] = count
            finally:
                try:
                    dk.execute("DETACH src")
                except Exception:
                    pass

    def sync_duckdb_table(self, table: str) -> None:
        _validate_identifier(table, "table")
        self._full_sync_duckdb_table(table)

