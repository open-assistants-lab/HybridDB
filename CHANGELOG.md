# Changelog

## [0.6.0] — 2026-08-26

### Added

**Versioned tables — git-like primitives with a tamper-evident hash chain**
(docs/specs/2026-08-26-versioned-tables-v0.6.0.md):

- `create_table(..., versioned=True, hash_chain=True)` — every
  insert/update/delete appends a post-image (or delete tombstone) to a
  shadow `__history` table with `SHA256(prev_hash | op | pk | row_json)`
  chain links computed by the engine. FTS5/Chroma keep indexing current
  data only — hybrid search is unchanged.
- `upsert(table, data)` — insert-or-update; history captures the event
  automatically on versioned tables.
- `log(table)` / `history(table, key)` — change log and per-row version
  history.
- `as_of(table, seq)` — point-in-time read; `diff(table, from_seq, to_seq)`
  — added/removed/changed rows between two log positions.
- `checkpoint(table, label)` / `rollback(table, checkpoint=|at_seq=)` —
  named restore points; rollback re-applies historical state as *new*
  versions (the chain never rewinds, so the audit trail stays complete),
  and is cheap for append-heavy tables (Chroma deletions, not re-embeds).
- `verify_chain(table)` — recomputes the chain and reports the first
  broken link; detects direct tampering with the history store.
- `archive(table, path, format="jsonl"|"parquet")` and
  `prune(table, before_seq=|checkpoint=)` — retention with **chain
  anchors** so pruning keeps `verify_chain` valid for the retained tail.
- `db.author` — optional author recorded per history event.
- Guardrails: schema changes (`add/drop/rename_column`) are rejected on
  versioned tables; `__history` table names are reserved; history tables
  are excluded from `list_tables`, DuckDB mirroring, and graph sync.
- Creating a table with an `id` column that lacks `PRIMARY KEY` now raises
  a clear error (it previously collided with the implicit auto-increment
  `id`, surfacing as a cryptic `duplicate column name` from SQLite).
- `scripts/publish.sh` — token-prompting release upload helper.
- Measured write overhead for versioned tables: ~13%
  (20.6k → 18.0k rows/s at 100k rows).
- `fork` is deferred to a later release (checkpoint/rollback covers the
  agent-memory rewind workflow; fork is a materialized copy with full
  re-embedding).

## [0.5.8] — 2026-08-25

### Fixed
- **Packaging**: `dependencies` and `keywords` were misplaced in
  `pyproject.toml` (parsed as URL metadata), so published wheels/sdists
  carried **no dependency declarations** — a fresh `pip install hybriddb`
  did not pull in chromadb. Both relocated into `[project]`.
- Version string consistency (`__version__` matches package metadata).

### Added
- PyPI badges and project URLs in README.

## [0.5.7] — 2026-08-25

### Performance (from the v0.5.7 performance study, docs/PERFORMANCE.md)

- **`insert_batch` is 13× faster** (1,860 → 24,500 rows/s at 100k rows):
  profiling showed the write bottleneck was opening a fresh SQLite
  connection per row (~200k connections per 100k rows), fixed by threading
  the batch cursor through the CRUD hot paths and using one connection
  end-to-end in journal processing.
- **`insert_batch(sync=True)` now actually means synced**: batches larger
  than the journal's per-call limit (5,000 entries) previously left the
  journal backlogged, so every subsequent search silently paid a 35–40s
  journal-flush embedding cost until drained. `sync=True` now drains fully
  before returning.
- DuckDB mirror stores `REAL` columns as `DOUBLE` (was float32, silently
  losing precision vs SQLite's float64 — caught by a correctness-checked
  benchmark).
- Analytics: incremental journal sync no longer fails when `src` is already
  attached (idempotent ATTACH).

### Added

- **BEIR search-accuracy evaluation** (`tests/benchmarks/test_accuracy.py`):
  graded nDCG/recall/precision/MRR on NFCorpus + SciFact, for
  keyword/semantic/hybrid, with fusion-weight and embedding-model
  sensitivity. Findings: hybrid fusion beats both single modes in both
  domains; the hash-embedding fallback is a 5.3× accuracy cliff.
- Deep-dive benchmarks: DuckDB-vs-SQLite per query shape with correctness
  assertions, mirror sync overhead, graph retrieval (`search_graph`,
  `search_graph_ppr`), traversal, NetworkX build/cache, and
  `scripts/analytics_sweep.py` — DuckDB wins 6–350× vs SQLite, growing with
  scale, with zero write overhead from the mirror.
- `docs/PERFORMANCE.md` — full performance study.

### Fixed

- Benchmark fixtures that broke at FULL scale: storage/graph tests inserted
  string ids into the implicit INTEGER pk; recall fixtures had 10k-doc
  clusters, capping recall@10 at 0.001; `test_sync_overhead` assumed one
  `process_journal()` drains a 1M-entry backlog. All now valid at any scale.

## [0.5.6] — 2026-08-20

### Fixed
- FTS5 indexes are now backfilled after every rebuild (`reindex`, `drop_column`, `rename_column`, `import_sql`). External-content FTS5 tables start empty and triggers only index future writes, so keyword search silently returned zero results for existing rows after any of these operations. `_create_fts5` now runs the `'rebuild'` command on creation. (#bug)
- `add_node`/`add_nodes` no longer delete a node's edges when re-adding an existing id. `INSERT OR REPLACE` triggered the `_graph_edges` FK `ON DELETE CASCADE` (REPLACE = DELETE + INSERT), silently wiping all edges of the node. Now an `ON CONFLICT(id) DO UPDATE` upsert that preserves edges and `created_at`. (#bug)
- Journal processing is now chronological with last-op-wins per row instead of grouped by op type. Grouped processing broke the journal order: delete-then-reinsert of a TEXT-PK row (rowid reused, no AUTOINCREMENT) produced duplicate ids in one Chroma upsert call (`DuplicateIDError`), wedging the journal permanently — every subsequent `sync=True` operation and even `search()` raised. Updates are applied as upserts so a row whose add fell outside the batch still lands in Chroma. (#bug)
- DuckDB analytics now supports custom-PK tables. The mirror no longer injects a synthetic `id BIGINT` column for tables whose PK is not named `id` (positional `INSERT INTO t SELECT * FROM src.t` mismatched columns, so `HybridDB.__init__` crashed with a `BinderException` on reopen, and incremental sync raised `no such column: id`). Full and incremental syncs now use the real PK column with parameterized values. Stale broken mirrors from older versions are detected at init and rebuilt automatically. (#bug)
- `reconcile()` no longer hardcodes an `id` column — it queried `SELECT rowid, id, ...`, which failed with `no such column` for custom-PK tables and silently skipped self-healing. (#bug)
- `insert`/`update` fetch rows by immutable SQLite `rowid` instead of by primary key, so updating the PK column works instead of crashing with `TypeError: 'NoneType' object is not iterable`. `insert` into a TEXT-PK table without providing the key now raises a clear `ValueError` instead of the same crash. (#bug)
- `GraphAPI` facade now exposes `search_graph_ppr` and `sync_graph_nodes` (both raised `AttributeError` since v0.5.0/v0.5.3). (#bug)
- `hybriddb.__version__` is back in sync with the package version (0.5.5). (#bug)
- `_is_hnsw_header_corrupt` no longer flags every healthy index as corrupt. Chroma-hnswlib stores 4 bytes per dimension plus a fixed per-element overhead (140 bytes for the vendored build), so the old `size_data_per_element == EMBEDDING_DIM * 4` comparison always failed — with `auto_rebuild_chroma=True` the index was rebuilt on every startup. The expected size is now derived from the collection's real dimension in Chroma's catalog. (#bug)
- `reindex()` (and therefore `import_sql`) no longer wipes Chroma metadata — the rebuild upsert now passes row metadata like `reconcile()` does. (#bug)
- `insert()` returns the real primary key value (e.g. an empty string for a TEXT PK) instead of coercing falsy keys to `0`. (#bug)
- DuckDB attach/detach cleanup is exception-safe: a failed `ATTACH` no longer masks the original error with a spurious `DETACH` failure, and broken mirror entries are dropped at init so they can be re-registered. (#bug)
- `db.graph.add_node` now mirrors the mixin signature — the first positional argument is the node id (`db.graph.add_node("n1", label="A")`). Previously the facade treated it as the label, so mixin-style calls silently created nodes with random ids; pass only `label=` to auto-generate an id as before. (#bug)
- Graph sync rules are validated at registration (`register_entity_node` rejects an unknown `id_column`) and stale rules no longer crash `sync_graph_nodes`/`search_graph`/`reconcile` — a rule referencing a dropped or renamed column is skipped with a warning instead of raising. (#bug)
- `_process_meta_update` now actually refreshes Chroma metadata after `drop_column`/`rename_column` (it was a no-op). Chroma merges metadata on update/upsert, so records are re-added with fresh metadata to drop stale keys. (#bug)
- `insert`/`insert_batch`/`update` now honor explicitly provided primary key values on default tables (e.g. `insert(id=100)`), and `update` correctly moves the row across Chroma and DuckDB when the PK changes — including INTEGER primary keys, where the rowid itself moves. Previously explicit ids were silently dropped and PK changes returned `False`/crashed. (#bug)
- `_process_meta_update` self-heals: rows missing from a Chroma collection (e.g. after a failed rename copy) are re-embedded from the SQLite documents instead of being skipped, so semantic search recovers without a manual `reconcile`. (#bug)
- `pagerank(personalization=...)` now raises a clear `ValueError` when personalization references nodes not in the graph, instead of a cryptic `ZeroDivisionError` from NetworkX. (#bug)
- Empty/NULL longtext values are embedded by the configured embedding function (falling back to a zero vector only if it errors), so custom embedding dimensions no longer wedge the journal with a 384-dim zero vector. (#bug)
- `insert`/`insert_batch`/`update` raise a clear `RuntimeError` (rolling back) if the row cannot be re-fetched after the write, instead of committing the change without journal entries (silent Chroma/DuckDB desync) or crashing with `TypeError: 'NoneType' object is not iterable`. (#bug)
- `read_query` is now enforced read-only at the SQLite level via an authorizer. The first-token check could be bypassed with `WITH x AS (SELECT 1) DELETE FROM t`, which executed the write. Note: the authorizer receives the C API action codes, which differ from the incomplete constants exposed by Python's `sqlite3` module. (#bug)
- `drop_column` now rejects dropping the primary key column with a clear `ValueError` — previously a custom TEXT PK was silently coerced to `INTEGER PRIMARY KEY AUTOINCREMENT`, corrupting the key values. (#bug)
- The FTS5 fallback LIKE search now uses `ESCAPE '\'`, so queries containing `%` or `_` match the literal characters instead of nothing (the backslash escapes were previously ineffective). (#bug)

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
