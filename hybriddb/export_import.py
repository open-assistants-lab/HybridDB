"""SQL dump / restore with automatic index rebuild."""

from __future__ import annotations

import logging
import re
import shutil
import sqlite3
from pathlib import Path

logger = logging.getLogger("hybriddb")

# FTS5 shadow tables, virtual tables, and triggers are rebuilt by
# import_sql → reindex. Exclude all statements that reference _fts_
# tables (CREATE, INSERT, CREATE VIRTUAL TABLE, CREATE TRIGGER, etc).
_FTS5_RE = re.compile(r"(?:_fts_|\bFTS5\b|\bCREATE\s+TRIGGER\b)")


class ExportImportMixin:
    """SQL export/import backed by SQLite iterdump + executescript."""

    def export_sql(self, path: str | Path) -> None:
        """Export the entire database as a portable SQL text file.

        Uses SQLite ``iterdump()`` with FTS5 virtual tables and triggers
        excluded — they are rebuilt on import. ChromaDB vectors and
        DuckDB analytics are also not included.

        Args:
            path: Output file path for the SQL dump.
        """
        output = Path(path)
        with self._db_lock:
            self._process_journal()
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                with open(output, "w") as f:
                    for line in conn.iterdump():
                        if _FTS5_RE.search(line):
                            continue
                        f.write(line + "\n")
            finally:
                conn.close()

    def import_sql(self, path: str | Path) -> None:
        """Load an SQL dump and rebuild all indexes.

        Destructive: replaces the entire database. A backup at
        ``.import_backup`` is created automatically as a safety net.

        After import, search works immediately with full keyword + semantic
        + DuckDB analytics.

        Args:
            path: Input SQL dump file path.
        """
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Dump file not found: {input_path}")

        with self._db_lock:
            # Create safety backup
            import_backup = self.path.with_suffix(self.path.suffix + ".import_backup")
            self._process_journal()
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                conn.close()
            if import_backup.exists():
                shutil.rmtree(str(import_backup), ignore_errors=True)
            shutil.copytree(str(self.path), str(import_backup))

            # Close DuckDB before deleting the DB file
            if self._duckdb_conn is not None:
                try:
                    self._duckdb_conn.close()
                except Exception:
                    pass
                self._duckdb_conn = None

            # Delete old ChromaDB collections (new ones rebuild via reindex)
            if self._chroma is not None:
                try:
                    for c in self._chroma.list_collections():
                        try:
                            name = c.name if hasattr(c, "name") else str(c)
                            self._chroma.delete_collection(name)
                        except Exception:
                            pass
                except Exception:
                    pass

            # Delete the SQLite database file and WAL/SHM
            db_file = Path(self._db_path)
            for suffix in ("", "-wal", "-shm"):
                f = db_file.with_name(db_file.name + suffix)
                if f.exists():
                    f.unlink()

            # Execute the dump into a fresh database
            sql_text = input_path.read_text()
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(sql_text)
                conn.commit()
            finally:
                conn.close()

            # Reinitialize ChromaDB (collections may have changed)
            if self._max_chroma_index_gb > 0:
                self._init_chroma(force=True)

            # Reinitialize DuckDB
            if self._duckdb_path:
                self._init_duckdb()
                self._auto_register_duckdb_tables()

            # Rebuild all indexes from the imported data
            self.reindex()
