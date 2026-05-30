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

class MaintenanceMixin:
    def _check_index_health(self, auto_rebuild: bool = False) -> None:
        max_bytes = self._max_chroma_index_gb * 1024**3
        warn_bytes = int(max_bytes * _CHROMA_INDEX_WARN_FACTOR)

        vector_dir = Path(self._vector_path)
        if not vector_dir.exists():
            return
        for seg_dir in vector_dir.iterdir():
            if not seg_dir.is_dir():
                continue
            link_file = seg_dir / "link_lists.bin"
            header_file = seg_dir / "header.bin"
            if not link_file.exists() or not header_file.exists():
                continue

            size_bytes = link_file.stat().st_size
            size_gb = size_bytes / (1024**3)
            header_corrupt = self._is_hnsw_header_corrupt(str(header_file), str(link_file))

            if size_bytes >= warn_bytes or header_corrupt:
                logger.warning(
                    "chromadb.index_bloated path=%s size_gb=%.2f max_gb=%d header_corrupt=%s",
                    str(link_file), size_gb, self._max_chroma_index_gb, header_corrupt,
                )

            if size_bytes > max_bytes or header_corrupt:
                logger.error(
                    "chromadb.index_needs_rebuild path=%s size_gb=%.2f reason=%s",
                    str(link_file), size_gb,
                    "header_corrupt" if header_corrupt else "size_exceeded",
                )
                if auto_rebuild:
                    logger.info("chromadb.auto_rebuilding path=%s", str(link_file))
                    self._rebuild_chroma_index()
                    return

    @staticmethod
    def _is_hnsw_header_corrupt(header_path: str, link_path: str) -> bool:
        try:
            with open(header_path, "rb") as f:
                data = f.read()
            if len(data) < 68:
                return True
            max_elements = struct.unpack_from("Q", data, 8)[0]
            size_data_per_element = struct.unpack_from("Q", data, 24)[0]
            max_m0 = struct.unpack_from("I", data, 56)[0]
            if max_elements == 0 or max_elements > _CHROMA_INDEX_MAX_ELEMENTS:
                return True
            if size_data_per_element != EMBEDDING_DIM * 4:
                return True
            if max_m0 > _CHROMA_INDEX_MAX_M0:
                return True
            return False
        except Exception:
            return True

    def _rebuild_chroma_index(self) -> None:
        if self._chroma is None:
            return
        old_path = Path(self._vector_path)
        temp_root = Path(tempfile.mkdtemp(dir=old_path.parent, prefix="chroma_rebuild_"))
        temp_vectors = temp_root / "vectors"
        try:
            new_client = chromadb.PersistentClient(
                path=str(temp_vectors),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            collection_names = self._chroma.list_collections()
            for coll_name in collection_names:
                coll_name_str = coll_name.name if hasattr(coll_name, "name") else str(coll_name)
                old_col = self._chroma.get_collection(coll_name_str)
                metadata = old_col.metadata
                new_col = new_client.create_collection(name=coll_name_str, metadata=metadata)
                offset = 0
                while True:
                    result = old_col.get(
                        limit=_CHROMA_REBUILD_BATCH,
                        offset=offset,
                        include=["embeddings", "documents", "metadatas"],
                    )
                    ids = result.get("ids", [])
                    if not ids:
                        break
                    emb = result.get("embeddings", [])
                    docs = result.get("documents", [])
                    metas = result.get("metadatas", [])
                    if None in emb or None in docs:
                        raise RuntimeError(
                            f"Corrupt vector data in collection '{coll_name_str}' "
                            f"at offset {offset} — manual intervention required"
                        )
                    new_col.add(ids=ids, embeddings=emb, documents=docs, metadatas=metas)
                    offset += _CHROMA_REBUILD_BATCH

            backup_path = old_path.with_suffix(
                old_path.suffix + ".backup_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
            )
            shutil.move(str(old_path), str(backup_path))
            try:
                shutil.move(str(temp_vectors), str(old_path))
            except Exception:
                shutil.move(str(backup_path), str(old_path))
                raise
            shutil.rmtree(str(temp_root), ignore_errors=True)
            self._chroma = chromadb.PersistentClient(
                path=str(old_path), settings=ChromaSettings(anonymized_telemetry=False),
            )
            key = os.fspath(old_path)
            with _chroma_pool_lock:
                _chroma_client_pool[key] = self._chroma
            logger.info("chromadb.index_rebuilt old_backup=%s new_path=%s", str(backup_path), str(old_path))
        except Exception:
            shutil.rmtree(str(temp_root), ignore_errors=True)
            raise

    def force_rebuild_chroma_index(self) -> dict:
        if self._chroma is None:
            return {"status": "unavailable", "error": "ChromaDB not initialized"}
        self._rebuild_chroma_index()
        total = sum(
            self._chroma.get_collection(c.name if hasattr(c, "name") else str(c)).count()
            for c in self._chroma.list_collections()
        )
        return {"status": "rebuilt", "vectors_copied": total}

    def _get_embedding(self, text: str) -> list[float]:
        if not text:
            return [0.0] * EMBEDDING_DIM
        return self._embedding_fn(text)

    def _get_collection(self, name: str):
        if self._chroma is None:
            return None
        return self._chroma.get_or_create_collection(name=name)

    def reconcile(self, table: str) -> dict:
        _validate_identifier(table, "table")
        result = {"ghosts_deleted": 0, "missing_added": 0, "metadata_updated": 0}
        lt_cols = self._get_longtext_columns(table)

        for col in lt_cols:
            collection_name = f"{table}_{col}"
            try:
                collection = self._get_collection(collection_name)
                chroma_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()

                with self._connect() as cur:
                    cur.execute(f"SELECT rowid as _row, id, {col} FROM {table}")
                    sql_rows = cur.fetchall()

                sql_ids = {str(r["_row"]) for r in sql_rows}
                id_to_row = {str(r["_row"]): dict(r) for r in sql_rows}

                ghosts = chroma_ids - sql_ids
                if ghosts:
                    collection.delete(ids=list(ghosts))
                    result["ghosts_deleted"] += len(ghosts)

                missing = sql_ids - chroma_ids
                if missing:
                    with self._connect() as cur:
                        cur.execute(
                            f"SELECT *, rowid as _rowid FROM {table} "
                            f"WHERE rowid IN ({','.join('?' * len(missing))})",
                            tuple(int(mid) for mid in missing),
                        )
                        full_rows = cur.fetchall()
                    full_row_by_rowid = {str(r["_rowid"]): dict(r) for r in full_rows}

                    ids_batch, embeddings_batch, docs_batch, metas_batch = [], [], [], []
                    for mid in missing:
                        row = id_to_row.get(mid)
                        full_row = full_row_by_rowid.get(mid)
                        if row:
                            doc = row[col] or ""
                            ids_batch.append(mid)
                            embeddings_batch.append(self._get_embedding(doc))
                            docs_batch.append(doc)
                            metas_batch.append(self._row_to_metadata(table, full_row) if full_row else {})
                            if len(ids_batch) >= CHROMA_BATCH:
                                collection.upsert(
                                    ids=ids_batch, embeddings=embeddings_batch,
                                    documents=docs_batch, metadatas=metas_batch,
                                )
                                result["missing_added"] += len(ids_batch)
                                ids_batch, embeddings_batch, docs_batch, metas_batch = [], [], [], []
                    if ids_batch:
                        collection.upsert(
                            ids=ids_batch, embeddings=embeddings_batch,
                            documents=docs_batch, metadatas=metas_batch,
                        )
                        result["missing_added"] += len(ids_batch)

                self._hybrid_disabled.pop(table, None)
            except Exception as e:
                logger.warning("reconcile_failed table=%s column=%s error=%s", table, col, e)

        self._auto_sync_graph_nodes()
        self._auto_sync_graph_edges()
        self.decay_edges()
        self.raw_query("DELETE FROM _graph_edges WHERE weight < 0.05")
        return result

    def health(self, table: str) -> dict:
        _validate_identifier(table, "table")
        with self._connect() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            sql_count = cur.fetchone()[0]

        chroma_docs: dict[str, int] = {}
        status = "ok"

        for col in self._get_longtext_columns(table):
            collection_name = f"{table}_{col}"
            try:
                collection = self._get_collection(collection_name)
                chroma_docs[collection_name] = collection.count()
                if chroma_docs[collection_name] != sql_count:
                    status = "drift"
            except Exception:
                chroma_docs[collection_name] = -1
                status = "broken"

        pending = self._journal_count(table)
        if pending > 0 and status == "ok":
            status = "drift"

        return {
            "sqlite_rows": sql_count, "chroma_docs": chroma_docs,
            "pending_journal": pending, "status": status,
        }

    def close(self) -> None:
        if self._duckdb_conn is not None:
            try:
                self._duckdb_conn.close()
            except Exception:
                pass

