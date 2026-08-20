from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from hybriddb.utils import (
    CHROMA_BATCH,
    JOURNAL_CAP,
    _validate_identifier,
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
        entries: list[dict] = []
        with self._connect() as cur:
            cur.execute(
                "SELECT app_table, COUNT(*) FROM _journal WHERE status = 'pending' GROUP BY app_table"
            )
            for row in cur.fetchall():
                table_caps[row[0]] = row[1]
            cur.execute(
                "SELECT * FROM _journal WHERE status = 'pending' ORDER BY id LIMIT ?",
                (batch_limit,),
            )
            entries = [dict(r) for r in cur.fetchall()]

            for tbl, count in table_caps.items():
                if count > JOURNAL_CAP:
                    logger.warning("journal.overflow table=%s pending=%d", tbl, count)
                    self._hybrid_disabled[tbl] = True

            if not entries:
                return 0

            self._apply_chroma_entries(entries)

            row_entries = [e for e in entries if e["op"].startswith("row_")]
            if row_entries:
                try:
                    self._sync_duckdb_from_journal(row_entries)
                except Exception as e:
                    logger.warning("duckdb.sync_failed error=%s", e)

            done_ids = [e["id"] for e in entries]
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

    def _apply_chroma_entries(self, entries: list[dict]) -> None:
        """Apply chroma journal entries chronologically (last-op-wins per row).

        Grouping by op type (all adds, then all deletes) broke the journal
        order: e.g. delete-then-reinsert of a TEXT-PK row (rowid reused)
        produced duplicate ids in one upsert call (DuplicateIDError, permanent
        journal wedge) or left Chroma with the wrong final state.
        """
        chroma_entries = [e for e in entries if e["op"] in ("add", "update", "delete")]
        final_by_collection: dict[str, dict[str, dict]] = defaultdict(dict)
        for entry in chroma_entries:
            collection_name = f"{entry['app_table']}_{entry['column_name']}"
            final_by_collection[collection_name][str(entry["row_id"])] = entry

        for coll_name, by_id in final_by_collection.items():
            collection = self._get_collection(coll_name)
            if collection is None:
                continue
            ids = list(by_id.keys())
            for i in range(0, len(ids), CHROMA_BATCH):
                chunk = ids[i:i + CHROMA_BATCH]
                upsert_ids: list[str] = []
                embeddings: list[list[float]] = []
                documents: list[str] = []
                metadatas: list[dict | None] = []
                delete_ids: list[str] = []
                for rid in chunk:
                    entry = by_id[rid]
                    if entry["op"] == "delete":
                        delete_ids.append(rid)
                        continue
                    doc = entry["data"] or ""
                    upsert_ids.append(rid)
                    embeddings.append(self._get_embedding(doc))
                    documents.append(doc)
                    metadata = json.loads(entry["metadata"]) if entry["metadata"] else {}
                    metadatas.append(metadata or None)
                if delete_ids:
                    collection.delete(ids=delete_ids)
                if upsert_ids:
                    kwargs: dict[str, Any] = {
                        "ids": upsert_ids,
                        "embeddings": embeddings,
                        "documents": documents,
                    }
                    if any(m is not None for m in metadatas):
                        kwargs["metadatas"] = metadatas
                    collection.upsert(**kwargs)

        for entry in [e for e in entries if e["op"] == "meta_update"]:
            try:
                self._process_meta_update(entry)
            except Exception as e:
                logger.warning("journal.meta_update_failed entry_id=%s error=%s", entry["id"], e)

    def _process_meta_update(self, entry: dict) -> None:
        """Refresh Chroma metadata for a table/column after a schema change.

        ``drop_column``/``rename_column`` write ``meta_update`` journal
        entries so stored metadata stays in sync with the new schema
        (e.g. renamed keys, removed columns). Chroma merges metadata on
        update/upsert and rejects empty dicts, so stale keys can only be
        removed by deleting and re-adding the records with fresh metadata.
        """
        table = entry["app_table"]
        col = entry["column_name"]
        if not col:
            return
        collection = self._get_collection(f"{table}_{col}")
        if collection is None:
            return
        rows = self.raw_query(f"SELECT *, rowid AS _row FROM {table}")
        if not rows:
            return
        rows_by_id = {str(r["_row"]): dict(r) for r in rows}
        meta_by_id = {
            rid: self._row_to_metadata(table, row) for rid, row in rows_by_id.items()
        }
        ids = list(meta_by_id.keys())
        for i in range(0, len(ids), CHROMA_BATCH):
            chunk = ids[i:i + CHROMA_BATCH]
            existing = collection.get(ids=chunk, include=["embeddings", "documents"])
            existing_ids = existing.get("ids", [])
            if existing_ids:
                embeddings = existing.get("embeddings")
                documents = existing.get("documents")
                if embeddings is None or documents is None \
                        or len(embeddings) != len(existing_ids) or len(documents) != len(existing_ids):
                    logger.warning(
                        "journal.meta_update_missing_data table=%s column=%s", table, col
                    )
                    continue
                if any(e is None for e in embeddings) or any(doc is None for doc in documents):
                    logger.warning(
                        "journal.meta_update_missing_data table=%s column=%s", table, col
                    )
                    continue
                collection.delete(ids=existing_ids)
                collection.add(
                    ids=existing_ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=[meta_by_id[i] for i in existing_ids],
                )
            # Self-heal: ids absent from the collection (e.g. a failed
            # rename copy) are re-embedded from the SQLite documents.
            missing_ids = [rid for rid in chunk if rid not in set(existing_ids)]
            if missing_ids:
                docs = [rows_by_id[rid][col] or "" for rid in missing_ids]
                collection.add(
                    ids=missing_ids,
                    embeddings=[self._get_embedding(doc) for doc in docs],
                    documents=docs,
                    metadatas=[meta_by_id[rid] for rid in missing_ids],
                )

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

