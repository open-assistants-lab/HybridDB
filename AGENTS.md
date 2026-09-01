# AGENTS.md — notes for AI coding agents working on HybridDB

Embedded hybrid-search DB: SQLite + FTS5 + ChromaDB + DuckDB + graph, with a
self-healing journal and opt-in versioned tables. Python ≥3.11, `uv` for
everything.

## Commands

```bash
uv run python -m pytest -q                     # full suite (fast, ~20s)
uv run --with ruff ruff check hybriddb tests   # lint (config: pyflakes rules only)
uv run python -m pytest tests/benchmarks -q --run-benchmarks --benchmark-disable   # benchmark smoke
uv run python -m pytest tests/benchmarks/test_accuracy.py -q --run-benchmarks --benchmark-disable  # BEIR accuracy (downloads datasets on first run)
```

Release: `docs/RELEASE.md` — build with `uv build`, publish with
`scripts/publish.sh` (prompts for the PyPI token; it is also in `.env`).
Bump version in **both** `pyproject.toml` and `hybriddb/__init__.py`, and add
a CHANGELOG entry — `__version__` and pyproject have drifted apart before.

## Layout

- `hybriddb/db.py` — `HybridDB` class composes the mixins; `_connect()` is
  the single SQLite entry point (RLock-serialized, WAL, FK on)
- mixins: `schema.py`, `crud.py`, `search.py`, `journal.py`, `versioning.py`,
  `graph.py`, `analytics.py`, `maintenance.py`, `export_import.py`,
  `async_api.py`, plus `facades.py` (`db.graph`/`db.olap`), `embedding.py`
  and `chunking.py` (long-doc splitting helper)
- versioning spec + amendments: `docs/specs/2026-08-26-versioned-tables-v0.6.0.md`
- performance study: `docs/PERFORMANCE.md`; benchmarks:
  `tests/benchmarks/` + `docs/BENCHMARKS.md`

## Hard-won invariants — do not regress these

**Connection hygiene (the #1 historical bottleneck).** Opening a SQLite
connection costs ~0.5ms. Never call `_table_meta()`/`_get_*()` without a
`cur` inside a per-row loop — thread the caller's `cur` through (this was a
13× slowdown once). Hoist loop-invariant lookups out of batch loops.

**Journal semantics.**
- `_process_journal()` processes ≤5,000 entries per call. `sync=True` write
  paths must drain (`while self._journal_count(table) > 0:`) — a large
  `sync=False` batch + one `process_journal()` call leaves a backlog, and
  every search then pays the flush cost.
- Journal processing applies Chroma ops **chronologically, last-op-wins per
  (collection, rowid)** — never regroup by op type (caused a
  `DuplicateIDError` wedge on TEXT-PK delete+reinsert; TEXT-PK tables reuse
  rowids, default tables don't because of AUTOINCREMENT).
- `row_delete` journal entries carry the app-level pk in `data` (the row is
  already gone from SQLite when DuckDB syncs).

**FTS5.** External-content tables start empty; triggers only index future
writes. Any path that recreates an FTS table (create/reindex/drop_column/
rename_column/import_sql) must run `INSERT INTO fts(fts) VALUES('rebuild')`
(`_create_fts5` does this — don't remove it). The LIKE fallback in
`_fts_search` needs `ESCAPE '\'` for `%`/`_`.

**DuckDB mirror.**
- REAL columns map to `DOUBLE` (float32 loses precision — caught by a
  correctness-checked benchmark).
- DuckDB mirrors are registered **lazily on first OLAP query** — do not
  auto-register at open; journal entries for unregistered tables are
  already skipped.
- `_versioned`/`__history`-style engine tables must stay out of
  `_schema`/`list_tables()`/DuckDB auto-registry/graph sync.
- DuckDB **merges** metadata on update/upsert; removing keys requires
  delete+re-add.
- `ATTACH` doesn't take bind params, but params *do* work through the
  sqlite scanner for SELECTs — prefer parameters over string-built literals.
- `ATTACH`/`DETACH` are wrapped in try/except — a failed ATTACH must not be
  masked by a DETACH error in `finally`.

**Versioned tables.** History rows are **post-images** (state after the op;
delete = tombstone with the last known row) — `as_of` is a latest-per-pk
query, not a replay. `prune` deletes a contiguous prefix and records a chain
anchor; without the anchor `verify_chain` breaks. Rollback re-applies state
as *new* versions (chain never rewinds) and is itself audited. Schema
changes on versioned tables raise. `__history` suffix and `_fts_` substrings
are reserved in table names. History tables must never be registered in
`_schema`, mirrored to DuckDB, or FTS/Chroma indexed.

**Rollback batching.** Both rollback phases are set-based — removals (chunked
`DELETE … pk IN`, 500 pks per statement) and restores (missing → batched
`INSERT`, changed → batched full-row `UPDATE`; post-images contain every
column, so full-row SET is equivalent). The SHA chain is computed in memory
across all three phases (removals → insert-restores → update-restores, pks
sorted, chain head threaded through `executemany` — the chain forces
*ordering*, not per-row transactions). Never reintroduce per-row
`delete()`/`upsert()` loops here (they measured 41× slower). `prune` deletes a contiguous prefix and records a chain
anchor; without the anchor `verify_chain` breaks. Rollback re-applies state
as *new* versions (chain never rewinds) and is itself audited. Schema
changes on versioned tables raise. `__history` suffix and `_fts_` substrings
are reserved in table names. History tables must never be registered in
`_schema`, mirrored to DuckDB, or FTS/Chroma indexed.

**read_query** is authorizer-enforced read-only. Note: the authorizer
receives the **C API action codes**, which differ from the constants in
Python's `sqlite3` module (`SQLITE_CREATE_TABLE=2` vs C's 5;
`SQLITE_REPLACE` doesn't exist) — use the raw values. A raising callback is
treated as DENY.

**Chroma quirks.** Query results include numpy arrays — never use
`arr in [...]`-style truthiness on them. Chroma metadata **merges** on
update/upsert and rejects empty dicts; removing a metadata key requires
delete+re-add. Empty docs: `_get_embedding("")` calls the configured fn so
custom dims stay consistent. Metadata scalar columns are mirrored for
`where=` pre-filtering; operator forms (`$gte`…) are enforced by Chroma for
semantic/hybrid but must ALSO be evaluated in `_matches_where` for keyword
mode (no Chroma query happens there). Chroma metadata merges on
update/upsert — removing keys requires delete+re-add.

**Long documents** are chunked above the engine (`hybriddb.chunking` +
chunks-as-rows with a parent link) — one embedding per LONGTEXT cell is the
right granularity for messages/memory, not for multi-page documents.

## Testing conventions

- Bug fixes get a failing test first, in `tests/test_regressions.py` (or a
  dedicated file), named after the behavior not the bug.
- Benchmark functions must be **idempotent** — pytest-benchmark calls them
  repeatedly; state them with a `DELETE FROM ...` first or use
  `pedantic(rounds=1)`.
- Benchmarks with correctness assertions are the best regression net we
  have (they caught the float32 mirror bug) — assert result equality, not
  just timing.
- Data with string ids (`generate_graph_data`, `generate_docs`) needs TEXT
  PK tables — explicit ids are honored on insert, and a string id into an
  implicit `INTEGER PRIMARY KEY` raises `IntegrityError` (by design).
- Recall-style fixtures: **fixed relevant set (10/cluster) + scale-amount
  distractors** — with 10k-doc clusters, recall@10 is mathematically capped
  at 0.001.

## Performance notes

- Smoke-scale micro-benchmarks have a ±50–180% noise floor; only trust
  interleaved A/B runs or the 100k/1M-row sweeps
  (`scripts/analytics_sweep.py`). Full-scale runs take hours — fixture
  re-embedding dominates (each test rebuilds its DB).
- Measured reference points (Apple Silicon): non-versioned `insert_batch`
  ~20–24k rows/s, versioned ~18k (+13%); DuckDB mirror 6–350× faster than
  SQLite for analytics, growing with scale; point lookups only 2.4× (mirror
  has no index — use `get()`/`query()`); hybrid fusion beats keyword and
  semantic on BEIR (NFCorpus + SciFact); hash-embedding fallback is a 5.3×
  accuracy cliff.