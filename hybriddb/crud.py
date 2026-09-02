from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from hybriddb.utils import (
    JOURNAL_CAP,
    _now_iso,
    _validate_identifier,
    _validate_order_by,
)

logger = logging.getLogger("hybriddb")

class CrudMixin:
    def _row_to_metadata(self, table: str, row: dict[str, Any], cur=None,
                         meta: dict[str, Any] | None = None) -> dict[str, Any]:
        # `meta` may be passed pre-fetched by batch loops (connection-hygiene:
        # avoid re-reading + re-parsing the schema json per row).
        if meta is None:
            meta = self._table_meta(table, cur=cur)
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
        with self._connect() as cur:
            meta = self._table_meta(table, cur=cur)
            if not meta:
                raise ValueError(f"Table '{table}' not found")
            filtered = {k: v for k, v in data.items() if k in meta["columns"]}
            # The implicit autoincrement PK is not part of _schema metadata;
            # honor an explicitly provided value (e.g. insert(id=100)) instead
            # of silently dropping it.
            pk_col = self._get_pk_column(table, cur=cur)
            if pk_col not in meta["columns"] and pk_col in data:
                filtered[pk_col] = data[pk_col]
            columns = list(filtered.keys())
            placeholders = ", ".join("?" * len(columns))
            col_types = meta["columns"]
            values = []
            for c in columns:
                v = filtered[c]
                if isinstance(v, (dict, list)) and col_types.get(c) == "JSON":
                    v = json.dumps(v, default=str)
                values.append(v)

            cur.execute(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})", values)
            internal_rowid = cur.lastrowid
            if pk_col in filtered:
                user_pk = filtered[pk_col]
            else:
                # No PK supplied: only valid when the PK aliases SQLite rowid
                # (INTEGER PRIMARY KEY), otherwise the row is unfindable.
                pk_type = self._pk_column_type(table, cur=cur)
                if pk_col != "rowid" and pk_type is not None and "INT" not in pk_type.upper():
                    raise ValueError(
                        f"Missing required primary key column '{pk_col}' for insert into '{table}'"
                    )
                user_pk = internal_rowid
            fetched = cur.execute(
                f"SELECT * FROM {table} WHERE rowid = ?", (internal_rowid,)
            ).fetchone()
            if fetched is None:
                raise RuntimeError(
                    f"Inserted row not found in '{table}' (rowid {internal_rowid})"
                )
            row = dict(fetched)

            metadata = self._row_to_metadata(table, row, cur=cur)
            now = _now_iso()
            versioned, hash_chain = self._versioned_state(cur, table)
            for col in self._get_longtext_columns(table, cur=cur):
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
            if versioned:
                self._capture_history(cur, table, "insert", user_pk, row, hash_chain=hash_chain)
        if sync:
            self._process_journal()
        return user_pk if user_pk is not None else 0

    def upsert(self, table: str, data: dict, sync: bool = True) -> int | str:
        """Insert the row if its primary key is absent, else update it.

        On versioned tables the prior state is captured in history
        automatically (insert records the new row; update records the
        new state as a post-image).

        Returns the primary key value.
        """
        _validate_identifier(table, "table")
        with self._connect() as cur:
            pk_col = self._get_pk_column(table, cur=cur)
        pk_val = data.get(pk_col)
        if pk_val is None:
            raise ValueError(
                f"upsert requires the primary key column '{pk_col}' in data"
            )
        if self.get(table, pk_val) is None:
            self.insert(table, data, sync=sync)
        else:
            self.update(table, pk_val, data, sync=sync)
        return pk_val

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
            pk_col = self._get_pk_column(table, cur=cur)
            lt_cols = self._get_longtext_columns(table, cur=cur)
            versioned, hash_chain = self._versioned_state(cur, table)
            history_prev: str | None = None  # chain head, maintained in-memory
            for data in rows:
                filtered = {k: v for k, v in data.items() if k in meta["columns"]}
                if pk_col not in meta["columns"] and pk_col in data:
                    filtered[pk_col] = data[pk_col]
                columns = list(filtered.keys())
                placeholders = ", ".join("?" * len(columns))
                col_types = meta["columns"]
                values = []
                for c in columns:
                    v = filtered[c]
                    if isinstance(v, (dict, list)) and col_types.get(c) == "JSON":
                        v = json.dumps(v, default=str)
                    values.append(v)
                cur.execute(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})", values)
                internal_rowid = cur.lastrowid
                if pk_col in filtered:
                    user_pk = filtered[pk_col]
                else:
                    pk_type = self._pk_column_type(table, cur=cur)
                    if pk_col != "rowid" and pk_type is not None and "INT" not in pk_type.upper():
                        raise ValueError(
                            f"Missing required primary key column '{pk_col}' for insert into '{table}'"
                        )
                    user_pk = internal_rowid
                fetched = cur.execute(
                    f"SELECT * FROM {table} WHERE rowid = ?", (internal_rowid,)
                ).fetchone()
                if fetched is None:
                    raise RuntimeError(
                        f"Inserted row not found in '{table}' (rowid {internal_rowid})"
                    )
                row = dict(fetched)
                ids.append(user_pk)
                metadata = self._row_to_metadata(table, row, cur=cur)
                for col in lt_cols:
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
                if versioned:
                    history_prev = self._capture_history(
                        cur, table, "insert", user_pk, row,
                        hash_chain=hash_chain, prev_hash=history_prev,
                    )
        if sync:
            # sync=True promises the indexes are current on return; the
            # journal processes in bounded batches, so drain fully.
            while self._journal_count(table) > 0:
                self._process_journal()
        return ids

    def update(self, table: str, row_id: int | str, data: dict, sync: bool = True) -> bool:
        _validate_identifier(table, "table")
        meta = self._table_meta(table)
        if not meta:
            raise ValueError(f"Table '{table}' not found")
        filtered = {k: v for k, v in data.items() if k in meta["columns"]}
        pk_col = self._get_pk_column(table)
        if pk_col not in meta["columns"] and pk_col in data:
            filtered[pk_col] = data[pk_col]
        if not filtered:
            return False

        col_types = meta["columns"]
        values = []
        for c in filtered:
            v = filtered[c]
            if isinstance(v, (dict, list)) and col_types.get(c) == "JSON":
                v = json.dumps(v, default=str)
            values.append(v)

        with self._connect() as cur:
            pk_col = self._get_pk_column(table, cur=cur)
            internal_rowid = self._resolve_internal_rowid(cur, table, row_id, pk_col)
            if internal_rowid is None:
                return False
            set_clause = ", ".join(f"{k} = ?" for k in filtered)
            cur.execute(f"UPDATE {table} SET {set_clause} WHERE {pk_col} = ?", values + [row_id])
            if cur.rowcount == 0:
                return False
            new_pk = filtered.get(pk_col)
            pk_changed = new_pk is not None and new_pk != row_id
            # Fetch by the new PK when the PK changed (the rowid itself moves
            # for INTEGER PRIMARY KEY tables, since they alias rowid), falling
            # back to the immutable rowid otherwise.
            row = None
            new_rowid = internal_rowid
            if pk_changed:
                fetched = cur.execute(
                    f"SELECT * FROM {table} WHERE {pk_col} = ?", (new_pk,)
                ).fetchone()
                if fetched is not None:
                    row = dict(fetched)
                r = cur.execute(
                    f"SELECT rowid FROM {table} WHERE {pk_col} = ?", (new_pk,)
                ).fetchone()
                if r is not None:
                    new_rowid = r[0]
            if row is None:
                fetched = cur.execute(
                    f"SELECT * FROM {table} WHERE rowid = ?", (internal_rowid,)
                ).fetchone()
                if fetched is None:
                    # The row vanished mid-update (e.g. a trigger). Roll back
                    # instead of committing the change without journal entries,
                    # which would silently desync Chroma/DuckDB.
                    raise RuntimeError(
                        f"Row disappeared during update of '{table}' (rowid {internal_rowid})"
                    )
                row = dict(fetched)
            metadata = self._row_to_metadata(table, row, cur=cur)
            lt_cols = self._get_longtext_columns(table, cur=cur)
            versioned, hash_chain = self._versioned_state(cur, table)
            if pk_changed:
                # The row's Chroma key is the rowid (or the PK itself for
                # INTEGER PRIMARY KEYs). Moving the PK means deleting the old
                # key and re-adding the row under its new key; DuckDB also
                # needs the old app id deleted explicitly.
                for col in lt_cols:
                    now = _now_iso()
                    cur.execute(
                        "INSERT INTO _journal (app_table, row_id, column_name, op, data, created_at) "
                        "VALUES (?, ?, ?, 'delete', ?, ?)",
                        (table, internal_rowid, col, str(row_id), now),
                    )
                    cur.execute(
                        "INSERT INTO _journal (app_table, row_id, column_name, op, data, metadata, created_at) "
                        "VALUES (?, ?, ?, 'add', ?, ?, ?)",
                        (table, new_rowid, col, row.get(col, ""), json.dumps(metadata), now),
                    )
                now = _now_iso()
                cur.execute(
                    "INSERT INTO _journal (app_table, row_id, op, created_at, data) "
                    "VALUES (?, ?, 'row_delete', ?, ?)",
                    (table, internal_rowid, now, str(row_id)),
                )
                cur.execute(
                    "INSERT INTO _journal (app_table, row_id, op, data, created_at) "
                    "VALUES (?, ?, 'row_add', ?, ?)",
                    (table, new_rowid, json.dumps(dict(row), default=str), now),
                )
            else:
                for col in lt_cols:
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
            if versioned:
                # post-image under the row's current pk; a pk change also
                # tombstones the old pk so as_of/history stay correct
                if pk_changed:
                    self._capture_history(
                        cur, table, "delete", row_id, {**row, pk_col: row_id},
                        hash_chain=hash_chain,
                    )
                self._capture_history(
                    cur, table, "update", row[pk_col], row, hash_chain=hash_chain,
                )
        if sync:
            self._process_journal()
        return True

    def delete(self, table: str, row_id: int | str, sync: bool = True) -> bool:
        _validate_identifier(table, "table")
        with self._connect() as cur:
            pk_col = self._get_pk_column(table, cur=cur)
            internal_rowid = self._resolve_internal_rowid(cur, table, row_id, pk_col)
            if internal_rowid is None:
                return False
            versioned, hash_chain = self._versioned_state(cur, table)
            deleted_row = None
            if versioned:
                fetched = cur.execute(
                    f"SELECT * FROM {table} WHERE {pk_col} = ?", (row_id,)
                ).fetchone()
                deleted_row = dict(fetched) if fetched is not None else None
            cur.execute(f"DELETE FROM {table} WHERE {pk_col} = ?", (row_id,))
            if cur.rowcount == 0:
                return False
            for col in self._get_longtext_columns(table, cur=cur):
                now = _now_iso()
                cur.execute(
                    "INSERT INTO _journal (app_table, row_id, column_name, op, data, created_at) "
                    "VALUES (?, ?, ?, 'delete', ?, ?)",
                    (table, internal_rowid, col, str(row_id), now),
                )
            now = _now_iso()
            cur.execute(
                "INSERT INTO _journal (app_table, row_id, op, created_at, data) "
                "VALUES (?, ?, 'row_delete', ?, ?)",
                (table, internal_rowid, now, str(row_id)),
            )
            if versioned and deleted_row is not None:
                # tombstone carrying the last known row state
                self._capture_history(
                    cur, table, "delete", deleted_row.get(pk_col, row_id),
                    deleted_row, hash_chain=hash_chain,
                )
        if sync:
            self._process_journal()
        return True

    def get(self, table: str, row_id: int | str) -> dict | None:
        _validate_identifier(table, "table")
        with self._connect() as cur:
            pk_col = self._get_pk_column(table, cur=cur)
            cur.execute(f"SELECT * FROM {table} WHERE {pk_col} = ?", (row_id,))
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
        # The first-token check is not enough: ``WITH x AS (SELECT 1) DELETE
        # FROM t`` starts with WITH but writes. Enforce read-only at the
        # SQLite level with an authorizer. Note: the authorizer receives the
        # C API action codes, which differ from the (incomplete) constants
        # exposed by Python's sqlite3 module, so use the raw values.
        # Allowed: PRAGMA(19), READ(20), SELECT(21), FUNCTION(31), RECURSIVE(33).
        _READ_ACTIONS = {19, 20, 21, 31, 33}

        conn = sqlite3.connect(self._db_path)
        try:
            def _deny_writes(action, arg1, arg2, dbname, source):
                return sqlite3.SQLITE_OK if action in _READ_ACTIONS else sqlite3.SQLITE_DENY

            conn.set_authorizer(_deny_writes)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def count(self, table: str, where: str = "", params: tuple = ()) -> int:
        _validate_identifier(table, "table")
        sql = f"SELECT COUNT(*) FROM {table}"
        if where:
            sql += f" WHERE {where}"
        with self._connect() as cur:
            result = cur.execute(sql, params).fetchone()
        return result[0] if result else 0

