"""Versioned tables: git-like primitives with a tamper-evident hash chain.

Design (docs/specs/2026-08-26-versioned-tables-v0.6.0.md):

- A versioned table has a shadow history table ``{table}__history`` recording
  a **post-image** of the row after every insert/update, and a tombstone
  (with the last known row) for deletes. Post-images make ``as_of``/``diff``
  single queries: the state at seq N is the latest post-image per pk with
  ``_seq <= N``, excluding tombstones.
- ``event_hash = SHA256(prev_hash | op | pk | row_json)`` — computed by the
  engine on write. Any modification or deletion of a history row breaks the
  chain and is detected by ``verify_chain()``.
- Rollback re-applies historical state as *new* versions — the chain never
  rewinds, so the audit trail stays complete.
- ``prune`` deletes a *contiguous prefix* of history and records a chain
  anchor (last pruned seq + hash) so ``verify_chain`` stays valid.
- History tables are engine-managed: not in ``_schema``, never listed,
  never mirrored to DuckDB, never graph-synced, never FTS/Chroma indexed.

Fork is deferred (materialized copies are expensive and no consumer needs
parallel versions yet); checkpoints cover the rewind workflow.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from hybriddb.utils import _now_iso, _validate_identifier

logger = logging.getLogger("hybriddb")

GENESIS_HASH = "0" * 64
HISTORY_SUFFIX = "__history"


def _history_table(table: str) -> str:
    return f"{table}{HISTORY_SUFFIX}"


def _event_hash(prev_hash: str, op: str, pk: str, row_json: str) -> str:
    payload = f"{prev_hash}|{op}|{pk}|{row_json}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _canon(row: dict) -> str:
    return json.dumps(row, sort_keys=True, default=str)


class VersioningMixin:
    def _init_versioning_tables(self) -> None:
        with self._connect() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS _versioned_tables (
                    table_name TEXT PRIMARY KEY,
                    hash_chain INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS _version_checkpoints (
                    table_name TEXT NOT NULL,
                    label TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (table_name, label)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS _chain_anchors (
                    table_name TEXT NOT NULL,
                    anchor_seq INTEGER NOT NULL,
                    anchor_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (table_name, anchor_seq)
                )
            """)

    # ── registry ───────────────────────────────────────────────────────────

    def is_versioned(self, table: str) -> bool:
        _validate_identifier(table, "table")
        with self._connect() as cur:
            row = cur.execute(
                "SELECT 1 FROM _versioned_tables WHERE table_name = ?", (table,)
            ).fetchone()
        return row is not None

    def _versioned_state(self, cur, table: str) -> tuple[bool, bool]:
        """(is_versioned, hash_chain) using the caller's cursor."""
        row = cur.execute(
            "SELECT hash_chain FROM _versioned_tables WHERE table_name = ?", (table,)
        ).fetchone()
        if row is None:
            return False, False
        return True, bool(row[0])

    def _versioned_state_read(self, table: str) -> tuple[bool, bool]:
        """(is_versioned, hash_chain) without a caller cursor."""
        with self._connect() as cur:
            return self._versioned_state(cur, table)

    def _create_history_table(self, cur, table: str) -> None:
        hname = _history_table(table)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {hname} (
                _seq INTEGER PRIMARY KEY AUTOINCREMENT,
                _op TEXT NOT NULL,
                _ts TEXT NOT NULL,
                _author TEXT,
                pk TEXT NOT NULL,
                row_json TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL
            )
        """)
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{hname}_pk ON {hname}(pk, _seq)"
        )

    def _capture_history(
        self, cur, table: str, op: str, pk: Any, row: dict[str, Any],
        hash_chain: bool = True, prev_hash: str | None = None,
    ) -> str:
        """Append a post-image event; returns the new chain head hash."""
        hname = _history_table(table)
        if prev_hash is None:
            r = cur.execute(
                f"SELECT event_hash FROM {hname} ORDER BY _seq DESC LIMIT 1"
            ).fetchone()
            prev_hash = r[0] if r else GENESIS_HASH
        rj = json.dumps(row, sort_keys=True, default=str)
        if hash_chain:
            event_hash = _event_hash(prev_hash, op, str(pk), rj)
        else:
            event_hash = ""
        cur.execute(
            f"INSERT INTO {hname} (_op, _ts, _author, pk, row_json, prev_hash, event_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (op, _now_iso(), getattr(self, "author", None), str(pk), rj, prev_hash, event_hash),
        )
        return event_hash

    # ── reads ──────────────────────────────────────────────────────────────

    def log(self, table: str, limit: int = 100) -> list[dict]:
        """Change log for a versioned table, newest first."""
        _validate_identifier(table, "table")
        self._require_versioned(table)
        rows = self.raw_query(
            f"SELECT _seq, _op, pk, _ts, _author, event_hash FROM {_history_table(table)} "
            "ORDER BY _seq DESC LIMIT ?",
            (limit,),
        )
        return [
            {
                "seq": r["_seq"], "op": r["_op"], "pk": r["pk"], "ts": r["_ts"],
                "author": r["_author"], "hash": r["event_hash"],
            }
            for r in rows
        ]

    def history(self, table: str, key: Any) -> list[dict]:
        """Every version of a row, oldest to newest, with hashes."""
        _validate_identifier(table, "table")
        self._require_versioned(table)
        rows = self.raw_query(
            f"SELECT _seq, _op, _ts, _author, row_json, event_hash, prev_hash "
            f"FROM {_history_table(table)} WHERE pk = ? ORDER BY _seq",
            (str(key),),
        )
        return [
            {
                "seq": r["_seq"], "op": r["_op"], "ts": r["_ts"], "author": r["_author"],
                "data": json.loads(r["row_json"]), "hash": r["event_hash"],
                "prev_hash": r["prev_hash"],
            }
            for r in rows
        ]

    def _require_versioned(self, table: str) -> None:
        if not self.is_versioned(table):
            raise ValueError(f"Table '{table}' is not versioned (create it with versioned=True)")

    def _state_at_raw(self, table: str, seq: int) -> dict[str, tuple[str, str]]:
        """Like `_state_at` but without json parsing: {pk: (row_json, op)},
        including tombstones. Used by `rollback` so rows that are not
        restored are never json-decoded."""
        hname = _history_table(table)
        rows = self.raw_query(
            f"SELECT pk, row_json, _op FROM ("
            f"  SELECT pk, row_json, _op, ROW_NUMBER() OVER (PARTITION BY pk ORDER BY _seq DESC) AS rn"
            f"  FROM {hname} WHERE _seq <= ?"
            f") WHERE rn = 1",
            (seq,),
        )
        return {r["pk"]: (r["row_json"], r["_op"]) for r in rows}

    def _state_at(self, table: str, seq: int) -> dict[str, dict[str, Any]]:
        """Reconstruct {pk_str: row} for the table state at a log position."""
        hname = _history_table(table)
        rows = self.raw_query(
            f"SELECT pk, row_json FROM ("
            f"  SELECT pk, row_json, _op, ROW_NUMBER() OVER (PARTITION BY pk ORDER BY _seq DESC) AS rn"
            f"  FROM {hname} WHERE _seq <= ?"
            f") WHERE rn = 1 AND _op != 'delete'",
            (seq,),
        )
        return {r["pk"]: json.loads(r["row_json"]) for r in rows}

    def as_of(self, table: str, seq: int) -> list[dict]:
        """Point-in-time read: the table's rows as of a log position."""
        _validate_identifier(table, "table")
        self._require_versioned(table)
        return list(self._state_at(table, seq).values())

    def diff(self, table: str, from_seq: int, to_seq: int) -> dict:
        """Rows added/removed/changed between two log positions."""
        _validate_identifier(table, "table")
        self._require_versioned(table)
        a = self._state_at(table, from_seq)
        b = self._state_at(table, to_seq)
        added = [b[k] for k in b if k not in a]
        removed = [a[k] for k in a if k not in b]
        changed = [
            {"before": a[k], "after": b[k]}
            for k in b if k in a and _canon(a[k]) != _canon(b[k])
        ]
        return {"added": added, "removed": removed, "changed": changed}

    # ── checkpoint / rollback ──────────────────────────────────────────────

    def checkpoint(self, table: str, label: str) -> dict:
        """Tag the current log position as a named restore point."""
        _validate_identifier(table, "table")
        self._require_versioned(table)
        if not label or not isinstance(label, str):
            raise ValueError("checkpoint label must be a non-empty string")
        with self._connect() as cur:
            row = cur.execute(
                f"SELECT COALESCE(MAX(_seq), 0) FROM {_history_table(table)}"
            ).fetchone()
            seq = row[0]
            cur.execute(
                "INSERT OR REPLACE INTO _version_checkpoints "
                "(table_name, label, seq, created_at) VALUES (?, ?, ?, ?)",
                (table, label, seq, _now_iso()),
            )
        return {"table": table, "label": label, "seq": seq}

    def _checkpoint_seq(self, table: str, label: str) -> int:
        with self._connect() as cur:
            row = cur.execute(
                "SELECT seq FROM _version_checkpoints WHERE table_name = ? AND label = ?",
                (table, label),
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown checkpoint '{label}' for table '{table}'")
        return row[0]

    def rollback(self, table: str, checkpoint: str | None = None,
                 at_seq: int | None = None, sync: bool = True) -> dict:
        """Re-apply the state at a checkpoint/seq as new versions.

        The chain never rewinds: the rollback itself is recorded as new
        versions, and everything discarded stays verifiable in history.
        Removals are applied in one batched transaction (set-based delete,
        bulk tombstone + journal writes, hash chain computed in memory).
        Restored LONGTEXT rows are re-embedded into Chroma, so rollback
        cost is O(removed + changed rows) — cheap for append-heavy tables.
        """
        _validate_identifier(table, "table")
        self._require_versioned(table)
        if checkpoint is not None:
            seq = self._checkpoint_seq(table, checkpoint)
        elif at_seq is not None:
            seq = at_seq
        else:
            raise ValueError("rollback requires checkpoint= or at_seq=")

        hname = _history_table(table)

        # O(1) early exit: no events logged since the target position means
        # current state provably matches it — nothing to do.
        with self._connect() as cur:
            head = cur.execute(f"SELECT COALESCE(MAX(_seq), 0) FROM {hname}").fetchone()[0]
        if head <= seq:
            return {"table": table, "seq": seq, "changes": 0}

        pk_col = self._get_pk_column(table)
        target_raw = self._state_at_raw(table, seq)
        meta = self._table_meta(table)
        versioned, hash_chain = self._versioned_state_read(table)
        lt_cols = self._get_longtext_columns(table)
        target_pks = {pk for pk, (rj, op) in target_raw.items() if op != "delete"}

        current: dict[str, tuple[int, dict[str, Any]]] = {}
        for r in self.raw_query(f"SELECT rowid AS _rid, * FROM {table}"):
            row = {k: v for k, v in dict(r).items() if k != "_rid"}
            current[str(r[pk_col])] = (r["_rid"], row)

        changes = 0

        # ── phase 1: batched removals — one transaction ──
        # Rows present now but absent from the target state. Applied before
        # any restores; pks sorted so the tombstone chain is deterministic.
        removal_pks = sorted(pk for pk in current if pk not in target_pks)
        if removal_pks:
            with self._connect() as cur:
                versioned, hash_chain = self._versioned_state(cur, table)
                lt_cols = self._get_longtext_columns(table, cur=cur)
                head_row = cur.execute(
                    f"SELECT event_hash FROM {hname} ORDER BY _seq DESC LIMIT 1"
                ).fetchone()
                prev_hash = head_row[0] if head_row else GENESIS_HASH
                now = _now_iso()
                author = getattr(self, "author", None)
                for start in range(0, len(removal_pks), 500):
                    chunk = removal_pks[start : start + 500]
                    ph = ",".join("?" * len(chunk))
                    fetched = cur.execute(
                        f"SELECT rowid AS _rid, * FROM {table} WHERE {pk_col} IN ({ph})",
                        chunk,
                    ).fetchall()
                    by_pk = {str(r[pk_col]): r for r in fetched}
                    cur.execute(f"DELETE FROM {table} WHERE {pk_col} IN ({ph})", chunk)
                    journal_col = [
                        (table, by_pk[pk]["_rid"], col, now)
                        for pk in chunk for col in lt_cols
                    ]
                    journal_row = [(table, by_pk[pk]["_rid"], now, pk) for pk in chunk]
                    tombstones = []
                    for pk in chunk:
                        tomb = {
                            k: v for k, v in dict(by_pk[pk]).items() if k != "_rid"
                        }
                        rj = json.dumps(tomb, sort_keys=True, default=str)
                        event_hash = (
                            _event_hash(prev_hash, "delete", pk, rj)
                            if hash_chain else ""
                        )
                        tombstones.append(
                            ("delete", now, author, pk, rj, prev_hash, event_hash)
                        )
                        if hash_chain:
                            prev_hash = event_hash
                    if journal_col:
                        cur.executemany(
                            "INSERT INTO _journal (app_table, row_id, column_name, op, created_at) "
                            "VALUES (?, ?, ?, 'delete', ?)",
                            journal_col,
                        )
                    if journal_row:
                        cur.executemany(
                            "INSERT INTO _journal (app_table, row_id, op, created_at, data) "
                            "VALUES (?, ?, 'row_delete', ?, ?)",
                            journal_row,
                        )
                    cur.executemany(
                        f"INSERT INTO {hname} (_op, _ts, _author, pk, row_json, prev_hash, event_hash) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        tombstones,
                    )
                    changes += len(chunk)

        # ── phase 2: restores / reverts — batched; one transaction ──
        # Split into insert-restores (missing pks) and update-restores
        # (content differs). Only restored rows are json-decoded; post-images
        # contain every column, so a full-row INSERT/SET is equivalent to the
        # prior per-row upsert semantics. Event order: insert-restores
        # (sorted pks) then update-restores (sorted pks) — the hash chain is
        # computed in memory and threaded through every batch.
        insert_pks: list[str] = []
        update_pks: list[str] = []
        for pk_str in sorted(target_raw):
            rj, op = target_raw[pk_str]
            if op == "delete":
                continue  # deleted in the target state — removal only applies
            cur_entry = current.get(pk_str)
            if cur_entry is None:
                insert_pks.append(pk_str)
            elif rj != _canon(cur_entry[1]):
                update_pks.append(pk_str)

        if insert_pks or update_pks:
            with self._connect() as cur:
                meta_cols = list(meta["columns"].keys())
                insert_cols = (
                    ([pk_col] if pk_col not in meta_cols else []) + meta_cols
                )
                non_pk_cols = [c for c in meta_cols if c != pk_col]
                head_row = cur.execute(
                    f"SELECT event_hash FROM {hname} ORDER BY _seq DESC LIMIT 1"
                ).fetchone()
                prev_hash = head_row[0] if head_row else GENESIS_HASH
                now = _now_iso()
                author = getattr(self, "author", None)

                # ── 2a: insert-restores (missing pks), chunked ──
                for ci in range(0, len(insert_pks), 500):
                    chunk = insert_pks[ci : ci + 500]
                    parsed_rows = []
                    ins_vals = []
                    for pk_str in chunk:
                        row = json.loads(target_raw[pk_str][0])
                        parsed_rows.append((pk_str, row))
                        ins_vals.append(tuple(row.get(c) for c in insert_cols))
                    ph = ", ".join("?" * len(insert_cols))
                    cur.executemany(
                        f"INSERT INTO {table} ({', '.join(insert_cols)}) VALUES ({ph})",
                        ins_vals,
                    )
                    ph2 = ",".join("?" * len(chunk))
                    fresh = cur.execute(
                        f"SELECT rowid AS _rid, {pk_col} FROM {table} "
                        f"WHERE {pk_col} IN ({ph2})",
                        chunk,
                    ).fetchall()
                    rid_by_pk = {str(r[pk_col]): r["_rid"] for r in fresh}
                    journal_col = []
                    journal_row = []
                    events = []
                    for pk_str, row in parsed_rows:
                        rid = rid_by_pk[pk_str]
                        md = json.dumps(
                            self._row_to_metadata(table, row, cur=cur, meta=meta)
                        )
                        for col in lt_cols:
                            journal_col.append(
                                (table, rid, col, row.get(col, "") or "", md, now)
                            )
                        journal_row.append(
                            (table, rid_by_pk[pk_str], now,
                             json.dumps(row, default=str))
                        )
                        ev_hash = (
                            _event_hash(prev_hash, "insert", pk_str, target_raw[pk_str][0])
                            if hash_chain else ""
                        )
                        events.append(
                            ("insert", now, author, pk_str,
                             target_raw[pk_str][0], prev_hash, ev_hash)
                        )
                        if hash_chain:
                            prev_hash = ev_hash
                    if journal_col:
                        cur.executemany(
                            "INSERT INTO _journal (app_table, row_id, column_name, op, data, metadata, created_at) "
                            "VALUES (?, ?, ?, 'add', ?, ?, ?)",
                            journal_col,
                        )
                    if journal_row:
                        cur.executemany(
                            "INSERT INTO _journal (app_table, row_id, op, data, created_at) "
                            "VALUES (?, ?, 'row_add', ?, ?)",
                            journal_row,
                        )
                    if events:
                        cur.executemany(
                            f"INSERT INTO {hname} (_op, _ts, _author, pk, row_json, prev_hash, event_hash) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            events,
                        )
                    changes += len(chunk)

                # ── 2b: update-restores — rows present but content differs ──
                # Full-row SET is equivalent to the prior per-row upsert
                # semantics: post-images contain every column.
                non_pk_cols = [c for c in meta_cols if c != pk_col]
                set_clause = ", ".join(f"{c} = ?" for c in non_pk_cols)
                for ci in range(0, len(update_pks), 500):
                    chunk = update_pks[ci : ci + 500]
                    updates = []
                    journal_col = []
                    journal_row = []
                    events = []
                    for pk_str in chunk:
                        row = json.loads(target_raw[pk_str][0])
                        vals = [row.get(c) for c in non_pk_cols] + [pk_str]
                        updates.append(tuple(vals))
                        rid = current[pk_str][0]
                        md = json.dumps(
                            self._row_to_metadata(table, row, cur=cur, meta=meta)
                        )
                        for col in lt_cols:
                            journal_col.append(
                                (table, rid, col, row.get(col, ""), md, now)
                            )
                        journal_row.append(
                            (table, rid, now, json.dumps(dict(row), default=str))
                        )
                        rj = target_raw[pk_str][0]
                        ev_hash = (
                            _event_hash(prev_hash, "update", pk_str, rj)
                            if hash_chain else ""
                        )
                        events.append(
                            ("update", now, author, pk_str, rj, prev_hash, ev_hash)
                        )
                        if hash_chain:
                            prev_hash = ev_hash
                    if updates:
                        cur.executemany(
                            f"UPDATE {table} SET {set_clause} WHERE {pk_col} = ?",
                            updates,
                        )
                    if journal_col:
                        cur.executemany(
                            "INSERT INTO _journal (app_table, row_id, column_name, op, data, metadata, created_at) "
                            "VALUES (?, ?, ?, 'update', ?, ?, ?)",
                            journal_col,
                        )
                    if journal_row:
                        cur.executemany(
                            "INSERT INTO _journal (app_table, row_id, op, data, created_at) "
                            "VALUES (?, ?, 'row_update', ?, ?)",
                            journal_row,
                        )
                    if events:
                        cur.executemany(
                            f"INSERT INTO {hname} (_op, _ts, _author, pk, row_json, prev_hash, event_hash) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            events,
                        )
                    changes += len(chunk)

        if sync:
            while self._journal_count(table) > 0:
                self._process_journal()
        return {"table": table, "seq": seq, "changes": changes}

    # ── chain security ─────────────────────────────────────────────────────

    def verify_chain(self, table: str) -> dict:
        """Recompute the hash chain; report the first broken link."""
        _validate_identifier(table, "table")
        self._require_versioned(table)
        hname = _history_table(table)
        with self._connect() as cur:
            row = cur.execute(
                "SELECT hash_chain FROM _versioned_tables WHERE table_name = ?", (table,)
            ).fetchone()
        if row is None:
            raise ValueError(f"Table '{table}' is not versioned")
        if not row[0]:
            raise ValueError(f"hash_chain is not enabled for table '{table}'")
        anchors = {
            (r["anchor_seq"], r["anchor_hash"])
            for r in self.raw_query(
                "SELECT anchor_seq, anchor_hash FROM _chain_anchors WHERE table_name = ?",
                (table,),
            )
        }
        prev = GENESIS_HASH
        checked = 0
        for r in self.raw_query(
            f"SELECT _seq, _op, pk, row_json, prev_hash, event_hash "
            f"FROM {hname} ORDER BY _seq"
        ):
            if r["prev_hash"] != prev:
                # allowed only at a prune anchor boundary
                if (r["_seq"] - 1, r["prev_hash"]) not in anchors:
                    return {"valid": False, "checked": checked, "first_broken_seq": r["_seq"]}
            expected = _event_hash(r["prev_hash"], r["_op"], r["pk"], r["row_json"] or "")
            if r["event_hash"] != expected:
                return {"valid": False, "checked": checked, "first_broken_seq": r["_seq"]}
            prev = r["event_hash"]
            checked += 1
        return {"valid": True, "checked": checked, "first_broken_seq": None}

    # ── archive / prune ────────────────────────────────────────────────────

    def archive(self, table: str, path: str | Path, format: str = "jsonl") -> dict:
        """Export current + history rows for retention.

        Formats: ``jsonl`` (dependency-free) or ``parquet`` (via DuckDB).
        Archive before pruning — prune deletes history permanently.
        """
        _validate_identifier(table, "table")
        self._require_versioned(table)
        if format not in ("jsonl", "parquet"):
            raise ValueError(f"Unknown archive format: {format!r} (jsonl or parquet)")
        dest = Path(path).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        current_rows = self.raw_query(f"SELECT * FROM {table}")
        history_rows = self.raw_query(f"SELECT * FROM {_history_table(table)}")

        current_path = dest / f"{table}.jsonl"
        history_path = dest / f"{table}__history.jsonl"
        if format == "parquet":
            if self._duckdb_conn is None:
                raise RuntimeError("parquet archive requires duckdb (pip install duckdb)")
            current_path = dest / f"{table}.parquet"
            history_path = dest / f"{table}__history.parquet"
            with self._db_lock:
                dk = self._duckdb_conn
                dk.execute(f"ATTACH '{self._db_path}' AS src (TYPE sqlite)")
                try:
                    dk.execute(
                        f"COPY (SELECT * FROM src.\"{table}\") TO '{current_path}' (FORMAT PARQUET)"
                    )
                    dk.execute(
                        f"COPY (SELECT * FROM src.\"{_history_table(table)}\") "
                        f"TO '{history_path}' (FORMAT PARQUET)"
                    )
                finally:
                    try:
                        dk.execute("DETACH src")
                    except Exception:
                        pass
        else:
            with open(current_path, "w") as f:
                for r in current_rows:
                    f.write(json.dumps(dict(r), default=str) + "\n")
            with open(history_path, "w") as f:
                for r in history_rows:
                    f.write(json.dumps(dict(r), default=str) + "\n")
        return {
            "path": str(dest), "format": format,
            "current": str(current_path), "history": str(history_path),
            "rows": len(current_rows), "history_rows": len(history_rows),
        }

    def prune(self, table: str, before_seq: int | None = None,
              checkpoint: str | None = None, keep_checkpoints: bool = True) -> dict:
        """Delete a contiguous prefix of history, keeping chain validity.

        Records a chain anchor at the prune boundary so ``verify_chain``
        remains valid for the retained tail. Only rows *older* than the
        boundary are removed; everything from the boundary onward is kept,
        so the retained history stays contiguous and verifiable.
        """
        _validate_identifier(table, "table")
        self._require_versioned(table)
        if checkpoint is not None:
            before_seq = self._checkpoint_seq(table, checkpoint) + 1
        elif before_seq is None:
            raise ValueError("prune requires before_seq= or checkpoint=")
        if keep_checkpoints:
            blocking = self.raw_query(
                "SELECT label FROM _version_checkpoints WHERE table_name = ? AND seq < ?",
                (table, before_seq),
            )
            if blocking:
                labels = ", ".join(b["label"] for b in blocking)
                raise ValueError(
                    f"prune would remove checkpoint(s) {labels} — pass keep_checkpoints=False "
                    f"to override, or prune to a later boundary"
                )
        hname = _history_table(table)
        with self._connect() as cur:
            last = cur.execute(
                f"SELECT _seq, event_hash FROM {hname} WHERE _seq < ? ORDER BY _seq DESC LIMIT 1",
                (before_seq,),
            ).fetchone()
            if last is None:
                return {"pruned": 0, "anchor_seq": None}
            cur.execute(f"DELETE FROM {hname} WHERE _seq < ?", (before_seq,))
            cur.execute(
                "INSERT OR REPLACE INTO _chain_anchors "
                "(table_name, anchor_seq, anchor_hash, created_at) VALUES (?, ?, ?, ?)",
                (table, last["_seq"], last["event_hash"], _now_iso()),
            )
            pruned = cur.execute("SELECT changes()").fetchone()[0]
        return {"pruned": pruned, "anchor_seq": last["_seq"]}