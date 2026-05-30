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

class JournalMixin:
    def _journal_count(self, table: str | None = None) -> int:
        with self._connect() as cur:
            if table:
                cur.execute(
                    "SELECT COUNT(*) FROM _journal WHERE status = 'pending' AND app_table = ?",
                    (table,),
                )
            else:
                cur.execute("SELECT COUNT(*) FROM _journal WHERE status = 'pending'")
            result = cur.fetchone()
        return result[0] if result else 0

    def _process_journal(self, batch_limit: int = 5000) -> int:
        table_caps: dict[str, int] = {}
        with self._connect() as cur:
            cur.execute(
                "SELECT app_table, COUNT(*) FROM _journal WHERE status = 'pending' GROUP BY app_table"
            )
            for row in cur.fetchall():
                table_caps[row[0]] = row[1]

        for tbl, count in table_caps.items():
            if count > JOURNAL_CAP:
                logger.warning("journal.overflow table=%s pending=%d", tbl, count)
                self._hybrid_disabled[tbl] = True

        with self._connect() as cur:
            cur.execute(
                "SELECT * FROM _journal WHERE status = 'pending' ORDER BY id LIMIT ?",
                (batch_limit,),
            )
            entries = [dict(r) for r in cur.fetchall()]

        if not entries:
            return 0

        chroma_entries = [e for e in entries if e["op"] in ("add", "update", "delete", "meta_update")]
        row_entries = [e for e in entries if e["op"].startswith("row_")]

        adds = [e for e in chroma_entries if e["op"] == "add"]
        updates = [e for e in chroma_entries if e["op"] == "update"]
        deletes = [e for e in chroma_entries if e["op"] == "delete"]
        meta_updates = [e for e in chroma_entries if e["op"] == "meta_update"]

        by_collection: dict[str, dict[str, list]] = defaultdict(
            lambda: {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
        )
        for entry in adds:
            collection_name = f"{entry['app_table']}_{entry['column_name']}"
            doc = entry["data"] or ""
            embedding = self._get_embedding(doc)
            metadata = json.loads(entry["metadata"]) if entry["metadata"] else {}
            if not metadata:
                metadata = None
            by_collection[collection_name]["ids"].append(str(entry["row_id"]))
            by_collection[collection_name]["embeddings"].append(embedding)
            by_collection[collection_name]["documents"].append(doc)
            by_collection[collection_name]["metadatas"].append(metadata)

        for coll_name, batch in by_collection.items():
            collection = self._get_collection(coll_name)
            if collection is None:
                continue
            for i in range(0, len(batch["ids"]), CHROMA_BATCH):
                kwargs: dict[str, Any] = {
                    "ids": batch["ids"][i:i + CHROMA_BATCH],
                    "embeddings": batch["embeddings"][i:i + CHROMA_BATCH],
                    "documents": batch["documents"][i:i + CHROMA_BATCH],
                }
                metas = batch["metadatas"][i:i + CHROMA_BATCH]
                if any(m is not None for m in metas):
                    kwargs["metadatas"] = metas
                collection.upsert(**kwargs)

        by_collection_del: dict[str, list[str]] = defaultdict(list)
        for entry in deletes:
            collection_name = f"{entry['app_table']}_{entry['column_name']}"
            by_collection_del[collection_name].append(str(entry["row_id"]))

        for coll_name, ids in by_collection_del.items():
            collection = self._get_collection(coll_name)
            if collection is None:
                continue
            for i in range(0, len(ids), CHROMA_BATCH):
                collection.delete(ids=ids[i:i + CHROMA_BATCH])

        for entry in updates:
            collection_name = f"{entry['app_table']}_{entry['column_name']}"
            try:
                collection = self._get_collection(collection_name)
                metadata = json.loads(entry["metadata"]) if entry["metadata"] else {}
                doc = entry["data"] or ""
                embedding = self._get_embedding(doc)
                update_kwargs: dict[str, Any] = {
                    "ids": [str(entry["row_id"])], "embeddings": [embedding],
                    "documents": [doc],
                }
                if metadata:
                    update_kwargs["metadatas"] = [metadata]
                collection.update(**update_kwargs)
            except Exception as e:
                logger.warning("journal.update_failed entry_id=%s error=%s", entry["id"], e)

        for entry in meta_updates:
            try:
                self._process_meta_update(entry)
            except Exception as e:
                logger.warning("journal.meta_update_failed entry_id=%s error=%s", entry["id"], e)

        if row_entries:
            try:
                self._sync_duckdb_from_journal(row_entries)
            except Exception as e:
                logger.warning("duckdb.sync_failed error=%s", e)

        done_ids = [e["id"] for e in entries]
        with self._connect() as cur:
            placeholders = ",".join("?" * len(done_ids))
            cur.execute(f"DELETE FROM _journal WHERE id IN ({placeholders})", done_ids)

        for tbl in table_caps:
            if not self._hybrid_disabled.get(tbl):
                continue
            remaining = self._journal_count(tbl)
            if remaining <= JOURNAL_CAP:
                self._hybrid_disabled.pop(tbl, None)
                logger.info("hybrid_search_recovered table=%s remaining=%d", tbl, remaining)

        return len(done_ids)

    def _process_meta_update(self, entry: dict) -> None:
        pass

    def journal_status(self, table: str | None = None) -> dict:
        if table:
            _validate_identifier(table, "table")
        with self._connect() as cur:
            if table:
                cur.execute(
                    "SELECT status, COUNT(*) FROM _journal WHERE app_table = ? GROUP BY status",
                    (table,),
                )
            else:
                cur.execute("SELECT status, COUNT(*) FROM _journal GROUP BY status")
            rows = cur.fetchall()
        result = {"pending": 0, "failed": 0, "done": 0}
        for row in rows:
            if row[0] in result:
                result[row[0]] = row[1]
        return result

    def process_journal(self, limit: int = 5000) -> int:
        return self._process_journal(batch_limit=limit)

