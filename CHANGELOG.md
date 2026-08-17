# Changelog

## [0.5.5] — 2026-08-16

### Breaking
- Synced graph node IDs are now namespaced by table: `{table}:{pk}` (e.g. `docs:1`, `items:a1`). Previously two registered tables with the same primary key values silently overwrote each other's nodes. `search_graph` / `search_graph_ppr` now return namespaced `node_id` values, and manual edges between synced nodes must use them. (#bug)

### Added
- `_auto_sync_graph_nodes()` now renders every `{column}` placeholder in label templates from the row (previously only the id column was substituted, leaving labels like `docs: {title}` as literal text) and refreshes labels on re-sync when rows change. (#bug)
- `_auto_sync_graph_nodes()` now removes ghost nodes — auto-synced nodes whose table row was deleted — instead of accumulating them forever. Result dict gains `nodes_updated` and `nodes_removed`. (#bug)

### Fixed
- `search_graph_ppr` now runs PageRank on the undirected subgraph, consistent with its `direction="both"` traversal. Directed PageRank flowed seed mass away from connected nodes, so auto-synced edges (child→parent) gave connected documents ~0 score and the "graph brings indirect match" behavior silently did nothing. (#bug)
- `_auto_sync_graph_edges` no longer hardcodes `s.id`/`t.id`. It uses each table's actual primary key column, so edge rules work with custom PK names (e.g. `uid TEXT PRIMARY KEY`) instead of raising `no such column`. The target column defaults to the target table's real PK, not the literal `id`. (#bug)
- `traverse()` with a `type` filter bound parameters in the wrong order (the type value landed in the final `WHERE node_id != ?`), returning the start node instead of typed neighbors. (#bug)
- `_find_seed_nodes` now logs swallowed per-collection search exceptions at debug level instead of failing silently (this is how the 0.5.4 rowid bug hid).

## [0.5.4] — 2026-08-15

### Fixed
- `search_graph` and `search_graph_ppr` returned `[]` for tables using the default `id INTEGER PRIMARY KEY AUTOINCREMENT`. `_find_seed_nodes` ran `SELECT rowid, id`, but SQLite names both result columns `id` when the PK is the rowid alias, so `dict(row)` dropped `rowid` and the rowid→PK mapping raised `KeyError`, which was silently swallowed — zero seeds, empty results. Now aliases the rowid (`SELECT rowid AS _rid`) so the mapping works for every PK type. (#bug)

## [0.5.3] — 2026-07-31

### Added
- `search_graph_ppr(k_seeds)` — separate control over number of seed nodes from vector search vs. number of final results. `k_seeds=None` (default) uses `limit` for both, preserving backward compatibility. Pass `k_seeds=20` with `limit=5` to spread from 20 seeds but return top-5.
- `sync_graph_nodes()` — public API wrapper for `_auto_sync_graph_nodes()`. Syncs registered table rows into graph nodes without requiring callers to use a private method.

## [0.5.2] — 2026-07-31

### Fixed
- Added `scipy` to the `graph` optional dependency. NetworkX's `pagerank` requires scipy at runtime but it wasn't listed.

## [0.5.1] — 2026-07-31

### Fixed
- `search_graph_ppr` default `min_similarity` changed from 0.1 to 0.0. The ChromaDB default embedding model (MiniLM) produces cosine distances around 1.0 for short texts, yielding `1.0 - distance ≈ 0`. A 0.1 threshold filtered out all seeds, returning empty results. Callers who need filtering should pass `min_similarity` explicitly.
- Added `scipy` to the `graph` optional dependency. NetworkX's `pagerank` requires scipy at runtime but it wasn't listed as a dependency.

## [0.5.0] — 2026-07-31

### Added
- `pagerank(personalization, alpha)` — Personalized PageRank support. The `personalization` dict biases the random walk toward seed nodes (query-relevant importance instead of global importance). `alpha` controls damping (lower = more concentrated near seeds). Backward compatible — `pagerank()` with no args returns standard PageRank as before.
- `search_graph_ppr(query, hop_expansion, limit, alpha, min_similarity)` — Graph-aware semantic retrieval via Personalized PageRank. Pipeline: vector search → filter by `min_similarity` → expand subgraph via `traverse()` → PPR on subgraph with seeds as personalization → return ranked by PPR score. Finds nodes that are semantically relevant AND structurally close to seeds.
- `_find_seed_nodes(query, limit, min_similarity)` — Extracted shared seed-finding helper used by both `search_graph` and `search_graph_ppr`. Fixed latent bug: ChromaDB returns rowids, not the table's primary key column values. Now maps rowids back to the actual PK via SQL lookup.

### Fixed
- `search_graph` and `_find_seed_nodes` now return the table's primary key value as `node_id` instead of the ChromaDB rowid. Previously, `search_graph` returned rowids which didn't match graph node IDs when tables use a custom-named `TEXT PRIMARY KEY` column.

## [0.4.5] — 2026-06-15

### Fixed
- `drop_column` no longer leaves stale FTS5 triggers for the dropped column, preventing "no such column" errors on subsequent inserts. Root cause: `_table_meta()` opened a new connection inside the transaction and couldn't see uncommitted `_save_table_meta` changes. Added optional `cur` parameter to `_table_meta`, `_get_text_columns`, `_get_longtext_columns`, and `_has_autoincrement_id` so `_rebuild_all_fts5` reads from the same connection. (#bug)
- `insert`/`update`/`insert_batch` now `json.dumps` dict/list values for `JSON`-typed columns instead of crashing with "Error binding parameter: type 'dict' is not supported". (#bug)
- `search_all`/`search_columns` now actually apply the `where` parameter (equality post-filter) instead of silently ignoring it. Added `_matches_where` helper. (#bug)
- `_is_hnsw_header_corrupt` now uses correct binary offsets for Chroma's `header.bin` (which has a 4-byte `PERSISTENCE_VERSION` prefix). Previously every header was flagged corrupt, triggering unnecessary index rebuilds. Also removed unused `link_path` parameter. (#bug)
- `create_table` now detects PRIMARY KEY on any column name (not just `id`), preventing duplicate PK definitions with custom-named primary keys. (#bug)
- Custom-named `INTEGER PRIMARY KEY AUTOINCREMENT` columns (e.g., `my_pk`) are now fully supported across insert, update, delete, get, batch insert, FTS search, and drop_column. Previously all downstream code hardcoded `WHERE id = ?` and `new.id`/`old.id` in FTS triggers. Added `_get_pk_column()` (for WHERE clauses) and `_get_rowid_ref()` (for FTS rowid references — only INTEGER PKs alias rowid). (#bug)
- `TEXT PRIMARY KEY` columns now correctly use `rowid` for FTS triggers and joins, avoiding "datatype mismatch" errors. (#bug)

## [0.4.4] — 2026-06-11

### Fixed
- `_sync_duckdb_from_journal` in `analytics.py` now resolves SQLite rowid back to the actual app-level id column when syncing add/update journal entries. Required for TEXT/UUID primary keys where rowid does not equal app id.
- `crud.delete()` now stores the app-level id in the journal's `data` column so `_sync_duckdb_from_journal` can sync row deletions (the row is already gone from SQLite at sync time).
- `_sync_duckdb_from_journal` now inspects the DuckDB id column type (BIGINT vs TEXT) to properly quote/unquote SQL values, preventing UUID quoting failures.
- `_sync_duckdb_from_journal` wraps the sync loop in `try/finally` so `DETACH src` always executes even if an exception occurs mid-sync. Previously a leaked DuckDB attachment would cause all subsequent syncs to fail.
- `search(table, column, query=None)` no longer passes the column name as the query parameter to `search_all()`. Now returns the most recent rows via `query()` with no search filtering.

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
