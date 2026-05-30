# HybridDB Architecture Split Design

## Context

HybridDB is being prepared for its first public PyPI release. The current `0.3.0` work has already made the public API developer-friendly enough for release:

- ChromaDB bundled MiniLM is the default embedding path.
- Public constants exist: `TEXT`, `LONGTEXT`, `INTEGER`, `REAL`, `BOOLEAN`, `JSON`.
- Typed schema helper exists: `Column`.
- Search accepts friendly modes: `mode="hybrid"`, `mode=HYBRID`, and `mode=SearchMode.HYBRID`.
- `db.search("docs", "query")` searches all searchable text columns.
- `cursor()`, `connect()`, `read_query()`, and async wrappers exist.
- `db.graph.*` and `db.olap.*` facades exist.
- Default tests pass quickly with benchmark tests skipped by default.
- Benchmark smoke tests are explicit and pass with `--run-benchmarks --benchmark-disable`.
- Documentation now includes `docs/API.md`, `docs/BENCHMARKS.md`, and `docs/RELEASE.md`.

The main remaining issue is internal architecture. Most behavior still lives in one large `hybriddb/db.py` file. That file now contains constructor logic, SQLite connection management, ChromaDB setup, DuckDB setup, schema management, CRUD, search, journal processing, graph operations, analytics, maintenance, async wrappers, public constants, type helpers, embedding helpers, and facades.

Before PyPI release is the right time to split internal architecture because:

- There are no external consumers relying on internal module paths yet.
- Public API behavior is already covered by tests.
- The release should not ship a hard-to-maintain 2,400-line core file if we can split it safely now.
- CoreMem and Executive Assistant depend on HybridDB behavior, so preserving the public API matters more than changing implementation style.
- Executive Assistant currently imports HybridDB's private embedding helper through `hybriddb.db._default_embedding_fn` from `src/sdk/hybrid_db.py`. The split must keep that compatibility path until EA is updated to a public import.

## Goal

Split HybridDB internals into focused modules while preserving the exact public API documented in `docs/API.md`.

The user-facing API must remain:

```python
from hybriddb import HybridDB, LONGTEXT, TEXT, HYBRID

db = HybridDB("./data")
db.create_table("docs", {"title": TEXT, "body": LONGTEXT})
db.insert("docs", {"title": "Hello", "body": "Hybrid search memory"})
db.search("docs", "memory")
db.search("docs", "body", "memory", mode=HYBRID)
await db.asearch("docs", "memory")
db.graph.add_node("Alice")
db.olap.query("SELECT COUNT(*) AS total FROM docs")
```

## Non-Goals

This spec does not include:

- Native async SQLite/ChromaDB clients.
- A new `AsyncHybridDB` class.
- Storage format changes.
- Journal schema changes.
- ChromaDB collection naming changes.
- DuckDB schema changes.
- Graph schema changes.
- Public method renames or removals.
- Behavior changes to search ranking, FTS, vector search, hybrid fusion, graph algorithms, or OLAP.

The async API will remain `asyncio.to_thread` wrappers around the sync API.

## Public API Compatibility Requirements

All of the following must continue to work after the split:

### Imports

```python
from hybriddb import (
    BOOLEAN,
    HYBRID,
    INTEGER,
    JSON,
    KEYWORD,
    LONGTEXT,
    REAL,
    SEMANTIC,
    TEXT,
    Column,
    default_embedding_fn,
    EmbeddingModelError,
    HybridDB,
    SearchMode,
)
```

`default_embedding_fn` must be exported publicly from `hybriddb`. It should be the stable public name for ChromaDB's bundled MiniLM embedding with hash fallback.

Compatibility requirement: `from hybriddb.db import _default_embedding_fn` must continue to work for one release cycle because Executive Assistant currently uses that private import. It may be documented as deprecated after EA is migrated to `from hybriddb import default_embedding_fn`.

### Core Methods

- `create_table`
- `add_column`
- `drop_column`
- `rename_column`
- `get_schema`
- `list_tables`
- `insert`
- `insert_batch`
- `update`
- `delete`
- `get`
- `query`
- `raw_query`
- `read_query`
- `cursor`
- `connect`
- `count`
- `search`
- `search_all`
- `search_columns`
- `health`
- `reconcile`
- `journal_status`
- `process_journal`
- `close`

### Async Methods

- `acreate_table`
- `aadd_column`
- `adrop_column`
- `arename_column`
- `ainsert`
- `ainsert_batch`
- `aupdate`
- `adelete`
- `aget`
- `aquery`
- `araw_query`
- `aread_query`
- `acount`
- `asearch`
- `asearch_all`
- `ahealth`
- `areconcile`
- `aprocess_journal`
- `aclose`

### Advanced Facades

- `db.graph.*` must continue to delegate to graph methods.
- `db.olap.*` must continue to delegate to analytics methods.
- Direct graph and analytics methods on `db` must remain available.

## Proposed Module Layout

```text
hybriddb/
├── __init__.py
├── db.py
├── types.py
├── embedding.py
├── utils.py
├── schema.py
├── crud.py
├── search.py
├── journal.py
├── graph.py
├── analytics.py
├── async_api.py
├── maintenance.py
└── facades.py
```

### `hybriddb/types.py`

Responsibility:

- `SearchMode`
- `KEYWORD`, `SEMANTIC`, `HYBRID`
- `Column`
- `TEXT`, `LONGTEXT`, `INTEGER`, `REAL`, `BOOLEAN`, `JSON`
- `EmbeddingModelError`

This module has no dependency on ChromaDB, SQLite, DuckDB, or NetworkX.

### `hybriddb/embedding.py`

Responsibility:

- `EMBEDDING_DIM`
- ChromaDB default embedding lazy initialization.
- Hash fallback embedding.
- `_default_embedding_fn(text)`.
- `default_embedding_fn(text)`.
- Compatibility alias: `_default_embedding_fn = default_embedding_fn`.
- Thread-safe embedding function cache.

This module may import ChromaDB lazily inside the default embedding loader.

### `hybriddb/utils.py`

Responsibility:

- `_now_iso()`
- `_is_safe_identifier()`
- `_validate_identifier()`
- `_validate_order_by()`
- `_coerce_search_mode()`
- `_column_spec()`
- `_sanitize_fts_query()`
- Small constants shared across modules, such as identifier regexes and system table names.

This module should stay small and should not import `HybridDB`.

### `hybriddb/facades.py`

Responsibility:

- `GraphAPI`
- `AnalyticsAPI`

These classes are thin delegating facades. They should not contain graph algorithms or DuckDB implementation details.

### `hybriddb/db.py`

Responsibility:

- Define `HybridDB`.
- Store constructor and shared initialization.
- Compose mixins.
- Own shared connection state:
  - SQLite paths.
  - ChromaDB client.
  - DuckDB connection.
  - locks.
  - graph/olap facades.
- Keep `_connect()` private, while exposing `cursor()` and `connect()` as public wrappers.

Expected shape:

```python
class HybridDB(
    SchemaMixin,
    CrudMixin,
    SearchMixin,
    JournalMixin,
    GraphMixin,
    AnalyticsMixin,
    MaintenanceMixin,
    AsyncMixin,
):
    ...
```

### `hybriddb/schema.py`

Responsibility:

- `SchemaMixin`
- `create_table`
- `add_column`
- `drop_column`
- `rename_column`
- `get_schema`
- `list_tables`
- FTS table/trigger creation helpers if they are schema-only.

Schema methods may call journal/search helpers through `self` but should not implement journal processing.

### `hybriddb/crud.py`

Responsibility:

- `CrudMixin`
- `insert`
- `insert_batch`
- `update`
- `delete`
- `get`
- `query`
- `raw_query`
- `read_query`
- `count`
- row metadata helpers.

CRUD methods may enqueue journal entries but should not process journal internals directly beyond calling `self._process_journal()`.

### `hybriddb/search.py`

Responsibility:

- `SearchMixin`
- `search`
- `search_all`
- `search_columns`
- `_fts_search`
- `_vector_search`
- `_fuse_hybrid`
- `_fetch_rows_by_ids`
- recency scoring.

Search must preserve existing behavior:

- `TEXT` supports keyword search.
- `LONGTEXT` supports keyword, semantic, and hybrid search.
- `search(table, query)` searches all searchable text columns.
- `search(table, column, query)` searches one column.

### `hybriddb/journal.py`

Responsibility:

- `JournalMixin`
- `_process_journal`
- `process_journal`
- `journal_status`
- `_journal_count`
- vector add/update/delete journal application.
- DuckDB row journal handoff.

Journal behavior is central to data integrity. It should move carefully and remain covered by existing journal tests.

### `hybriddb/graph.py`

Responsibility:

- `GraphMixin`
- Graph schema initialization helpers.
- Graph sync rules.
- Node and edge CRUD.
- Traversal and NetworkX-backed algorithms.
- `get_neighbors` alias.
- `add_edge(..., edge_type=...)` compatibility.

Direct methods remain on `HybridDB`; `db.graph.*` delegates through `GraphAPI`.

### `hybriddb/analytics.py`

Responsibility:

- `AnalyticsMixin`
- DuckDB initialization.
- DuckDB table registration.
- DuckDB sync.
- `analytics(sql)`.

Direct methods remain on `HybridDB`; `db.olap.*` delegates through `AnalyticsAPI`.

### `hybriddb/maintenance.py`

Responsibility:

- `MaintenanceMixin`
- `health`
- `reconcile`
- `close`
- Chroma pool cleanup if needed.

### `hybriddb/async_api.py`

Responsibility:

- `AsyncMixin`
- Async wrappers using `asyncio.to_thread`.

This module must not introduce `aiosqlite`, native async ChromaDB, or a separate async storage path.

## Migration Strategy

Use a mechanical split, not a rewrite.

Release order:

1. Refactor standalone HybridDB first.
2. Preserve the compatibility alias `hybriddb.db._default_embedding_fn` during the split.
3. Build and install the refactored standalone package locally.
4. Update Executive Assistant to import the public helper with `from hybriddb import default_embedding_fn`.
5. Run EA's embedding-path smoke checks and at least the 20Q LongMemEval smoke before removing any compatibility alias in a future release.

Do not refactor EA's in-repo `src/sdk/hybrid_db.py` first. EA should consume the stable public package API after standalone HybridDB defines it.

Recommended order:

1. Move import-free definitions first:
   - `types.py`
   - `embedding.py`
   - `utils.py`
   - `facades.py`
2. Update `__init__.py` exports.
3. Keep a compatibility import in `db.py`: `from hybriddb.embedding import default_embedding_fn as _default_embedding_fn`.
4. Move async wrappers to `async_api.py`.
5. Move schema methods to `schema.py`.
6. Move CRUD methods to `crud.py`.
7. Move search methods to `search.py`.
8. Move journal methods to `journal.py`.
9. Move analytics methods to `analytics.py`.
10. Move graph methods to `graph.py`.
11. Move maintenance methods to `maintenance.py`.
12. Reduce `db.py` to constructor, shared initialization, private shared connection helpers, compatibility imports, and mixin composition.
13. Update Executive Assistant to use `from hybriddb import default_embedding_fn` instead of `from hybriddb.db import _default_embedding_fn`.

After each step, run focused tests for the moved area.

## Testing Requirements

Before any refactor step:

```bash
uv run python -m pytest tests/test_db.py -q
```

After moving each module:

```bash
uv run python -m pytest tests/test_db.py -q
```

After the full split:

```bash
# 1. Quick import-cycle check before full suite
uv run python -c "from hybriddb import HybridDB; print('imports ok')"

# 2. Verify MRO resolves as expected
uv run python -c "
from hybriddb import HybridDB
mro = [c.__name__ for c in HybridDB.__mro__]
# Expect the same order used in the class definition:
# ['HybridDB', 'SchemaMixin', 'CrudMixin', 'SearchMixin', 'JournalMixin',
#  'GraphMixin', 'AnalyticsMixin', 'MaintenanceMixin', 'AsyncMixin', 'object']
print('mro:', ' -> '.join(mro))
"

# 2b. Verify embedding compatibility paths
uv run python -c "
from hybriddb import default_embedding_fn
from hybriddb.db import _default_embedding_fn
assert default_embedding_fn is _default_embedding_fn
assert len(default_embedding_fn('hello')) == 384
print('embedding compatibility ok')
"

# 3. Lint, test, benchmarks
uv run --with ruff ruff check hybriddb tests
uv run python -m pytest -q
uv run python -m pytest tests/benchmarks -q --run-benchmarks --benchmark-disable

# 4. Build + wheel smoke
rm -rf dist && uv build
```

Wheel smoke test:

```bash
uv run --no-project --isolated --no-cache \
  --with /Users/eddy/Developer/Python/HybridDB/dist/hybriddb-0.3.0-py3-none-any.whl \
  --with duckdb \
  python - <<'PY'
import asyncio
from tempfile import TemporaryDirectory
from hybriddb import HYBRID, LONGTEXT, TEXT, Column, HybridDB

async def main():
    with TemporaryDirectory() as tmp:
        db = HybridDB(tmp)
        await db.acreate_table('docs', {'title': Column(TEXT), 'body': LONGTEXT})
        await db.ainsert('docs', {'title': 'Hello', 'body': 'Hybrid search memory'})
        rows = await db.asearch('docs', 'memory', mode=HYBRID)
        assert rows and rows[0]['title'] == 'Hello'
        assert await db.aread_query('SELECT title FROM docs') == [{'title': 'Hello'}]
        with db.cursor() as cur:
            cur.execute('SELECT COUNT(*) FROM docs')
            assert cur.fetchone()[0] == 1
        node_id = db.graph.add_node('Alice', type='person')
        assert db.graph.get_node(node_id)['label'] == 'Alice'
        assert db.olap.query('SELECT COUNT(*) AS total FROM docs')[0]['total'] == 1
        await db.aclose()
    print('wheel smoke ok')

asyncio.run(main())
PY
```

Expected results:

- Core tests pass.
- Default suite reports `105 passed, 23 skipped` or a higher pass count if tests are added.
- Benchmark smoke reports `23 passed`.
- Ruff passes.
- Wheel smoke prints `wheel smoke ok`.

## Risk Management

Main risks:

- Import cycles between mixin modules and `db.py`.
- Moving private helpers into modules that need access to shared state.
- Accidentally changing method resolution order.
- Breaking optional dependencies: DuckDB and NetworkX should remain optional.
- Accidentally changing public exports from `hybriddb.__init__`.
- Breaking EA's private `hybriddb.db._default_embedding_fn` import before EA migrates to the public `hybriddb.default_embedding_fn` import.

Mitigations:

- Keep mixin modules dependent on `self`, not on concrete `HybridDB` imports.
- Use `TYPE_CHECKING` for type-only imports when needed. Every mixin module must use `from __future__ import annotations` or `TYPE_CHECKING` guards for any reference to `HybridDB` to prevent circular imports.
- No mixin should define `__init__`. All shared state is initialized in `HybridDB.__init__` and accessed via `self` in mixin methods.
- Verify `MRO` is correct: `HybridDB.__mro__` must resolve methods predictably. Add a once-off smoke assertion after the split.
- Keep ChromaDB, DuckDB, NetworkX imports lazy where they are currently lazy.
- Run tests after every module move.
- Do not change method names or signatures during the split.
- Preserve all existing private method names if tests or other modules still use them.
- Preserve `_default_embedding_fn` in `hybriddb.db` as a compatibility alias during this release. Do not remove it as part of the architecture split.

## Acceptance Criteria

The refactor is complete only when all of these are true:

- Public imports from `hybriddb` are unchanged.
- `from hybriddb import default_embedding_fn` works.
- `from hybriddb.db import _default_embedding_fn` still works and points to the same callable as `default_embedding_fn`.
- Public examples in `README.md` and `docs/API.md` still run.
- `HybridDB` is still the single public database class.
- `db.py` is reduced to constructor/shared initialization and mixin composition.
- Core tests pass.
- Default test suite passes with benchmark tests skipped.
- Benchmark smoke suite passes explicitly.
- Ruff passes.
- Wheel builds.
- Wheel smoke test passes in an isolated environment.
- Executive Assistant is updated to use `hybriddb.default_embedding_fn` after the standalone split is verified locally.

## Decision

Proceed with a conservative internal module split before PyPI release.

Do not implement native async, storage migrations, or public API changes as part of this refactor.
