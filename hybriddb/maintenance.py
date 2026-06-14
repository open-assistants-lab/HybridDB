from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import struct
import tempfile

import chromadb
from chromadb.config import Settings as ChromaSettings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hybriddb.embedding import EMBEDDING_DIM
from hybriddb.utils import (
    CHROMA_BATCH,
    _CHROMA_INDEX_MAX_ELEMENTS,
    _CHROMA_INDEX_MAX_M0,
    _CHROMA_INDEX_WARN_FACTOR,
    _CHROMA_REBUILD_BATCH,
    _validate_identifier,
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
            header_corrupt = self._is_hnsw_header_corrupt(str(header_file))

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
    def _is_hnsw_header_corrupt(header_path: str) -> bool:
        try:
            with open(header_path, "rb") as f:
                data = f.read()
            # header.bin layout: 4-byte version prefix, then struct fields
            # See chroma-hnswlib hnswalg.h persistHeader()
            if len(data) < 100:
                return True
            version = struct.unpack_from("i", data, 0)[0]
            if version != 1:
                return True
            max_elements = struct.unpack_from("Q", data, 12)[0]
            size_data_per_element = struct.unpack_from("Q", data, 28)[0]
            max_m0 = struct.unpack_from("Q", data, 68)[0]
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
        from hybriddb.db import _chroma_client_pool, _chroma_pool_lock
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

    # ── Import/Export & SQL Utilities ──────────────────────────────────────

    def backup(self, path: str | Path) -> None:
        """Copy the entire database directory atomically.

        Flushes the journal and checkpoints the WAL before copying,
        then copies all files (SQLite, ChromaDB vectors, DuckDB analytics).

        Args:
            path: Destination directory. Must not exist or must be empty.
        """
        dest = Path(path).resolve()
        dest_exists = dest.exists()
        if dest_exists and any(dest.iterdir()):
            raise FileExistsError(f"Destination must not exist or be empty: {dest}")
        if dest_exists:
            shutil.rmtree(str(dest))

        with self._db_lock:
            self._process_journal()
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                conn.close()
            shutil.copytree(str(self.path), str(dest))

    def restore(self, path: str | Path) -> None:
        """Replace the current database with a backup directory.

        The current database is moved to a ``.old`` directory before
        the restore. ChromaDB client pool is invalidated and reinitialized.

        Args:
            path: Source directory. Must exist and contain ``app.db``.
                Must not be inside the current database directory.
        """
        src = Path(path).resolve()
        if not src.exists() or not (src / "app.db").exists():
            raise FileNotFoundError(f"Not a valid HybridDB directory: {src}")

        if src == self.path.resolve() or str(self.path.resolve()) in str(src):
            raise ValueError(
                f"Restore source must be outside the current database directory. "
                f"Got src={src} inside db={self.path.resolve()}"
            )

        with self._db_lock:
            self._process_journal()

            old = self.path.with_suffix(self.path.suffix + ".old")
            if old.exists():
                ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                old = self.path.with_suffix(self.path.suffix + f".old_{ts}")

            shutil.move(str(self.path), str(old))
            shutil.copytree(str(src), str(self.path), dirs_exist_ok=True)

            from hybriddb.db import _chroma_client_pool, _chroma_pool_lock
            with _chroma_pool_lock:
                _chroma_client_pool.pop(str(self._vector_path), None)
            self._chroma = None
            self._init_chroma(force=True)

            if self._duckdb_conn is not None:
                try:
                    self._duckdb_conn.close()
                except Exception:
                    pass
                self._init_duckdb()
                self._auto_register_duckdb_tables()

    def vacuum(self) -> int:
        """Reclaim disk space by rebuilding the SQLite database file.

        Returns the number of bytes freed (before minus after file size).

        ChromaDB and DuckDB files are not affected.
        """
        before = Path(self._db_path).stat().st_size
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()
        after = Path(self._db_path).stat().st_size
        return max(0, before - after)

    def check_integrity(self) -> dict:
        """Run diagnostic checks across SQLite, ChromaDB, and DuckDB.

        Returns a structured report with ``overall`` status:
        ``"ok"``, ``"degraded"`` (recoverable), or ``"corrupt"`` (data-loss risk).
        """
        result: dict[str, Any] = {
            "sqlite_integrity": "ok",
            "chromadb_collections": 0,
            "chromadb_errors": [],
            "duckdb_tables_synced": 0,
            "duckdb_errors": [],
            "overall": "ok",
        }

        # SQLite integrity
        conn = sqlite3.connect(self._db_path)
        try:
            row = conn.execute("PRAGMA integrity_check").fetchone()
            result["sqlite_integrity"] = row[0] if row else "unknown"
        except Exception as e:
            result["sqlite_integrity"] = str(e)
        finally:
            conn.close()

        if result["sqlite_integrity"] != "ok":
            result["overall"] = "corrupt"

        # ChromaDB collections
        if self._chroma is not None:
            try:
                collections = self._chroma.list_collections()
                result["chromadb_collections"] = len(collections)
                for c in collections:
                    name = c.name if hasattr(c, "name") else str(c)
                    try:
                        self._chroma.get_collection(name).count()
                    except Exception as exc:
                        result["chromadb_errors"].append(f"{name}: {exc}")
            except Exception as e:
                result["chromadb_errors"].append(str(e))

        if result["chromadb_errors"] and result["overall"] != "corrupt":
            result["overall"] = "degraded"

        # DuckDB sync
        if self._duckdb_conn is not None and self._duckdb_path:
            try:
                for tname in sorted(self._duckdb_synced_tables):
                    try:
                        dk = self._duckdb_conn
                        quoted = self._duckdb_quote_identifier(tname)
                        dk.execute(f"SELECT 1 FROM {quoted} LIMIT 1")
                        result["duckdb_tables_synced"] += 1
                    except Exception as e:
                        result["duckdb_errors"].append(f"{tname}: {e}")
            except Exception as e:
                result["duckdb_errors"].append(str(e))

        if result["duckdb_errors"] and result["overall"] == "ok":
            result["overall"] = "degraded"

        return result

    def stats(self) -> dict:
        """Return size and count statistics for all storage layers."""
        result: dict[str, Any] = {
            "sqlite_size_bytes": 0,
            "chromadb_size_bytes": 0,
            "duckdb_size_bytes": 0,
            "total_size_bytes": 0,
            "tables": {},
        }

        if Path(self._db_path).exists():
            result["sqlite_size_bytes"] = Path(self._db_path).stat().st_size
        if Path(self._vector_path).exists():
            total = 0
            for f in Path(self._vector_path).rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
            result["chromadb_size_bytes"] = total
        if self._duckdb_path and Path(self._duckdb_path).exists():
            result["duckdb_size_bytes"] = Path(self._duckdb_path).stat().st_size

        result["total_size_bytes"] = (
            result["sqlite_size_bytes"]
            + result["chromadb_size_bytes"]
            + result["duckdb_size_bytes"]
        )

        for table in self.list_tables():
            meta = self._table_meta(table)
            if not meta:
                continue
            lt_cols = self._get_longtext_columns(table)
            text_cols = self._get_text_columns(table)
            tbl_stats: dict[str, Any] = {
                "rows": self.count(table),
                "fts_indexes": len(text_cols),
                "chromadb_collections": len(lt_cols),
                "chromadb_vectors": 0,
                "duckdb_synced": table in self._duckdb_synced_tables,
                "duckdb_rows": (
                    self._duckdb_synced_tables[table].get("count", 0)
                    if table in self._duckdb_synced_tables
                    else 0
                ),
            }
            if lt_cols and self._chroma is not None:
                total_vectors = 0
                for col in lt_cols:
                    try:
                        collection = self._chroma.get_collection(f"{table}_{col}")
                        total_vectors += collection.count()
                    except Exception:
                        pass
                tbl_stats["chromadb_vectors"] = total_vectors
            result["tables"][table] = tbl_stats

        return result

    def reindex(self, table: str | None = None) -> None:
        """Rebuild ChromaDB, FTS5, and DuckDB indexes from SQLite data.

        Automatically called by :meth:`import_sql`. Use for recovery when
        ChromaDB or FTS5 indexes become corrupt.

        Args:
            table: Table name to reindex, or ``None`` to reindex all user tables.
        """
        tables = [table] if table else self.list_tables()

        for tbl in tables:
            lt_cols = self._get_longtext_columns(tbl)

            # Rebuild FTS5
            with self._connect() as cur:
                self._rebuild_all_fts5(cur, tbl)

            # Rebuild ChromaDB
            if lt_cols and self._chroma is not None:
                for col in lt_cols:
                    collection_name = f"{tbl}_{col}"
                    try:
                        self._chroma.delete_collection(collection_name)
                    except Exception:
                        pass
                    collection = self._chroma.get_or_create_collection(
                        name=collection_name
                    )

                    offset = 0
                    while True:
                        rows = self.raw_query(
                            f"SELECT rowid as _row, {col} FROM {tbl} "
                            f"ORDER BY rowid LIMIT {CHROMA_BATCH} OFFSET {offset}"
                        )
                        if not rows:
                            break
                        ids_batch = [str(r["_row"]) for r in rows]
                        docs_batch = [r[col] or "" for r in rows]
                        emb_batch = [self._get_embedding(d) for d in docs_batch]
                        collection.upsert(
                            ids=ids_batch,
                            embeddings=emb_batch,
                            documents=docs_batch,
                        )
                        offset += CHROMA_BATCH

            # Rebuild DuckDB
            self._full_sync_duckdb_table(tbl)

