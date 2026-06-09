# Changelog

## [0.4.3] — 2026-06-09

### Fixed
- `_sync_duckdb_from_journal` in `analytics.py` now wraps the sync loop in `try/finally` so `DETACH src` always executes, even if an exception occurs mid-sync. Previously a leaked DuckDB attachment would cause all subsequent syncs to fail with `"src" already attached`.
- `search(table, column, query=None)` in `search.py` no longer passes the column name as the query parameter to `search_all()`. Now returns the most recent rows via `query()` with no search filtering.

## [0.4.2] — 2026-05-31

### Changed
- Capped all dependency upper bounds for supply chain safety: `chromadb>=1.5.0,<2.0`, `duckdb>=1.0.0,<2.0`, `networkx>=3.0,<4.0`, `sentence-transformers>=3.0.0,<6.0`.

## [0.4.1] — 2026-05-31

### Fixed
- `_rebuild_chroma_index` in `maintenance.py` now imports `_chroma_client_pool` and `_chroma_pool_lock` from `hybriddb.db` (late import) instead of using stale local copies. Fixes broken ChromaDB client pooling on index rebuild.

---

## [0.4.0] — 2026-05-31

### Added
- `export_sql(path)` — Export entire database as portable SQL dump (FTS5 excluded, rebuilt on import).
- `import_sql(path)` — Load SQL dump and rebuild ChromaDB, FTS5, and DuckDB indexes.
- `backup(path)` — Copy entire database directory atomically (SQLite + ChromaDB + DuckDB).
- `restore(path)` — Replace current database from a backup directory.
- `vacuum()` — Reclaim disk space by defragmenting the SQLite file.
- `check_integrity()` — Run diagnostic checks across SQLite, ChromaDB, and DuckDB.
- `stats()` — Return size and row count statistics for all storage layers.
- `reindex(table)` — Rebuild ChromaDB, FTS5, and DuckDB indexes from SQLite data.
- Recall@K benchmarks (`test_recall_keyword`, `test_recall_semantic`, `test_recall_hybrid`).
- Cold-start search benchmark (`test_cold_start_search`).

### Fixed
- `run_benchmarks.sh` — Fixed `$1` mode arg forwarding to pytest.
- `run_benchmarks.sh` — Added `--run-benchmarks` flag so benchmarks aren't skipped.

---

## [0.3.0] — 2026-05-31

### Changed
- **Architecture split**: Monolithic `db.py` (~2,400 lines) split into focused mixins: schema, CRUD, search, journal, graph, analytics, maintenance, async.
- Default embedding model switched to **ChromaDB's bundled MiniLM** (`all-MiniLM-L6-v2`, ~80MB). Previously required manual `sentence-transformers` install.
- Published to PyPI at `pip install hybriddb`.

### Added
- **Benchmark suite** — 27 benchmarks covering search (keyword/vector/hybrid), storage growth, concurrent ops, graph algorithms, analytics, and ChromaDB health.
- `compare_results.py` — Benchmark comparison tool with markdown output.
- `run_benchmarks.sh` — One-command smoke/full/e2e benchmark runner.

---

## [0.2.0] — 2026-05-28

### Added
- Full feature parity with in-repo (EA) version.
- Agent-first positioning in README and docs.
- Embedded, local-first, OSS differentiators.

---

## [0.1.0] — 2026-05-27

### Added
- Initial release: SQLite + FTS5 + ChromaDB + DuckDB + NetworkX graph in a single local-first Python package.
- `HybridDB` class with create, insert, update, delete, query, search, analytics, graph operations.
- Self-healing operation journal for cross-engine consistency.
- Concurrent write safety via threading.RLock.
- WAL mode for SQLite.
