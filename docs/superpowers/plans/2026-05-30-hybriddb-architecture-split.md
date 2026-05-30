# HybridDB Architecture Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split HybridDB internals into focused modules while preserving the public `HybridDB` API and local release behavior.

**Architecture:** Keep `HybridDB` as the single public class. Move behavior into mixins by responsibility, with `db.py` owning constructor/shared state and composing the mixins. Keep a compatibility alias for `hybriddb.db._default_embedding_fn` until Executive Assistant is migrated to the public `hybriddb.default_embedding_fn` import.

**Tech Stack:** Python 3.11+, SQLite, ChromaDB, optional DuckDB, optional NetworkX, pytest, ruff, hatchling/uv.

---

## File Map

- Modify: `hybriddb/__init__.py` — keep public exports stable and add `default_embedding_fn`.
- Modify: `hybriddb/db.py` — reduce to constructor/shared init, compatibility imports, and mixin composition.
- Create: `hybriddb/types.py` — constants, `SearchMode`, `Column`, `EmbeddingModelError`.
- Create: `hybriddb/embedding.py` — ChromaDB MiniLM default embedding and hash fallback.
- Create: `hybriddb/utils.py` — identifier validation, time helpers, search-mode coercion, FTS sanitization.
- Create: `hybriddb/facades.py` — `GraphAPI`, `AnalyticsAPI`.
- Create: `hybriddb/schema.py` — `SchemaMixin`.
- Create: `hybriddb/crud.py` — `CrudMixin`.
- Create: `hybriddb/search.py` — `SearchMixin`.
- Create: `hybriddb/journal.py` — `JournalMixin`.
- Create: `hybriddb/analytics.py` — `AnalyticsMixin`.
- Create: `hybriddb/graph.py` — `GraphMixin`.
- Create: `hybriddb/maintenance.py` — `MaintenanceMixin`.
- Create: `hybriddb/async_api.py` — `AsyncMixin`.
- Modify after standalone verification: `/Users/eddy/Developer/Python/executive-assistant/src/sdk/hybrid_db.py` — use public `from hybriddb import default_embedding_fn`.

---

### Task 1: Baseline Verification

**Files:** none

- [ ] **Step 1: Run core tests before moving code**

Run:

```bash
uv run python -m pytest tests/test_db.py -q
```

Expected: all core tests pass.

- [ ] **Step 2: Run default suite before moving code**

Run:

```bash
uv run python -m pytest -q
```

Expected: `105 passed, 23 skipped` or higher pass count if tests were added.

---

### Task 2: Move Types And Embedding

**Files:**
- Create: `hybriddb/types.py`
- Create: `hybriddb/embedding.py`
- Modify: `hybriddb/db.py`
- Modify: `hybriddb/__init__.py`

- [ ] **Step 1: Move public types**

Move these definitions from `hybriddb/db.py` to `hybriddb/types.py`:

```python
class SearchMode(Enum): ...
KEYWORD = SearchMode.KEYWORD
SEMANTIC = SearchMode.SEMANTIC
HYBRID = SearchMode.HYBRID
@dataclass(frozen=True)
class Column: ...
class EmbeddingModelError(Exception): ...
TEXT = "TEXT"
LONGTEXT = "LONGTEXT"
INTEGER = "INTEGER"
REAL = "REAL"
BOOLEAN = "BOOLEAN"
JSON = "JSON"
```

- [ ] **Step 2: Move embedding helpers**

Move these definitions from `hybriddb/db.py` to `hybriddb/embedding.py`:

```python
EMBEDDING_DIM = 384
_hash_embedding(text: str) -> list[float]
_get_default_ef()
default_embedding_fn(text: str) -> list[float]
_default_embedding_fn = default_embedding_fn
```

- [ ] **Step 3: Add compatibility imports in `db.py`**

`hybriddb/db.py` must keep:

```python
from hybriddb.embedding import default_embedding_fn, default_embedding_fn as _default_embedding_fn
from hybriddb.types import Column, EmbeddingModelError, SearchMode
```

- [ ] **Step 4: Update package exports**

`hybriddb/__init__.py` must export `default_embedding_fn` and all existing public constants/classes.

- [ ] **Step 5: Verify compatibility imports**

Run:

```bash
uv run python -c "from hybriddb import default_embedding_fn; from hybriddb.db import _default_embedding_fn; assert default_embedding_fn is _default_embedding_fn; assert len(default_embedding_fn('hello')) == 384; print('embedding compatibility ok')"
```

Expected: `embedding compatibility ok`.

- [ ] **Step 6: Run core tests**

Run:

```bash
uv run python -m pytest tests/test_db.py -q
```

Expected: pass.

---

### Task 3: Move Utilities And Facades

**Files:**
- Create: `hybriddb/utils.py`
- Create: `hybriddb/facades.py`
- Modify: `hybriddb/db.py`

- [ ] **Step 1: Move utility helpers**

Move these definitions to `hybriddb/utils.py`:

```python
_SYSTEM_TABLES
_SAFE_IDENTIFIER_RE
_STOPWORDS
_now_iso
_is_safe_identifier
_validate_identifier
_validate_order_by
_coerce_search_mode
_column_spec
_sanitize_fts_query
```

- [ ] **Step 2: Move facades**

Move `GraphAPI` and `AnalyticsAPI` to `hybriddb/facades.py`.

- [ ] **Step 3: Keep imports lazy where needed**

Do not import DuckDB or NetworkX in `facades.py`.

- [ ] **Step 4: Run core tests**

Run:

```bash
uv run python -m pytest tests/test_db.py -q
```

Expected: pass.

---

### Task 4: Move Async API

**Files:**
- Create: `hybriddb/async_api.py`
- Modify: `hybriddb/db.py`

- [ ] **Step 1: Move async wrappers into `AsyncMixin`**

Move all `a*` methods into:

```python
class AsyncMixin:
    async def acreate_table(...): ...
    async def ainsert(...): ...
    async def asearch(...): ...
```

Do not introduce native async clients.

- [ ] **Step 2: Compose `AsyncMixin` into `HybridDB`**

Temporarily set:

```python
class HybridDB(AsyncMixin):
    ...
```

Additional mixins will be added in later tasks.

- [ ] **Step 3: Run async tests twice**

Run:

```bash
uv run python -m pytest tests/test_db.py::TestAsyncApi -q
uv run python -m pytest tests/test_db.py::TestAsyncApi -q
```

Expected: both runs pass.

---

### Task 5: Move Schema, CRUD, Search, And Journal

**Files:**
- Create: `hybriddb/schema.py`
- Create: `hybriddb/crud.py`
- Create: `hybriddb/search.py`
- Create: `hybriddb/journal.py`
- Modify: `hybriddb/db.py`

- [ ] **Step 1: Move schema methods into `SchemaMixin`**

Move schema methods and schema-only helpers. Run:

```bash
uv run python -m pytest tests/test_db.py::TestCreateTable tests/test_db.py::TestSchemaOperations -q
```

Expected: pass.

- [ ] **Step 2: Move CRUD methods into `CrudMixin`**

Move CRUD/query methods. Run:

```bash
uv run python -m pytest tests/test_db.py::TestCRUD tests/test_db.py::TestInsertSync tests/test_db.py::TestAutoIncrement -q
```

Expected: pass.

- [ ] **Step 3: Move search methods into `SearchMixin`**

Move `search`, `search_all`, `search_columns`, FTS/vector helpers, fusion, and row fetch helpers. Run:

```bash
uv run python -m pytest tests/test_db.py::TestSearch -q
```

Expected: pass.

- [ ] **Step 4: Move journal methods into `JournalMixin`**

Move `_process_journal`, `process_journal`, `journal_status`, and journal helpers. Run:

```bash
uv run python -m pytest tests/test_db.py::TestJournal -q
```

Expected: pass.

---

### Task 6: Move Analytics, Graph, And Maintenance

**Files:**
- Create: `hybriddb/analytics.py`
- Create: `hybriddb/graph.py`
- Create: `hybriddb/maintenance.py`
- Modify: `hybriddb/db.py`

- [ ] **Step 1: Move analytics into `AnalyticsMixin`**

Run:

```bash
uv run python -m pytest tests/test_db.py -q -k "analytics or olap"
```

Expected: pass.

- [ ] **Step 2: Move graph into `GraphMixin`**

Run:

```bash
uv run python -m pytest tests/test_db.py -q -k "graph or node or edge or neighbor or pagerank"
```

Expected: pass.

- [ ] **Step 3: Move maintenance into `MaintenanceMixin`**

Run:

```bash
uv run python -m pytest tests/test_db.py -q -k "health or reconcile or journal_status or close"
```

Expected: pass.

---

### Task 7: Finalize `db.py` Composition

**Files:**
- Modify: `hybriddb/db.py`

- [ ] **Step 1: Compose final class**

`HybridDB` should look like:

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

- [ ] **Step 2: Verify MRO**

Run:

```bash
uv run python -c "from hybriddb import HybridDB; print(' -> '.join(c.__name__ for c in HybridDB.__mro__))"
```

Expected order begins with:

```text
HybridDB -> SchemaMixin -> CrudMixin -> SearchMixin -> JournalMixin -> GraphMixin -> AnalyticsMixin -> MaintenanceMixin -> AsyncMixin
```

- [ ] **Step 3: Run core tests**

Run:

```bash
uv run python -m pytest tests/test_db.py -q
```

Expected: pass.

---

### Task 8: Full Verification And Build

**Files:**
- No additional files unless verification finds issues.

- [ ] **Step 1: Lint**

Run:

```bash
uv run --with ruff ruff check hybriddb tests
```

Expected: `All checks passed!`

- [ ] **Step 2: Default tests**

Run:

```bash
uv run python -m pytest -q
```

Expected: `105 passed, 23 skipped` or better.

- [ ] **Step 3: Benchmark smoke**

Run:

```bash
uv run python -m pytest tests/benchmarks -q --run-benchmarks --benchmark-disable
```

Expected: `23 passed`.

- [ ] **Step 4: Build**

Run:

```bash
rm -rf dist && uv build
```

Expected: wheel and sdist build.

- [ ] **Step 5: Wheel smoke**

Run the wheel smoke command from `docs/RELEASE.md`.

Expected: `wheel smoke ok`.

---

### Task 9: Update Executive Assistant Embedding Import

**Files:**
- Modify: `/Users/eddy/Developer/Python/executive-assistant/src/sdk/hybrid_db.py`

- [ ] **Step 1: Replace private embedding import**

Change:

```python
from hybriddb.db import _default_embedding_fn as hybriddb_default
```

to:

```python
from hybriddb import default_embedding_fn as hybriddb_default
```

- [ ] **Step 2: Verify EA embedding path**

Run from `/Users/eddy/Developer/Python/executive-assistant`:

```bash
uv run python -c "from src.sdk.hybrid_db import _default_embedding_fn; r=_default_embedding_fn('hello'); assert len(r)==384; print('ea embedding ok')"
```

Expected: `ea embedding ok`.

- [ ] **Step 3: Verify MessageStore/CoreMem alignment**

Run a small MessageStore + CoreMem round-trip from EA and confirm search returns at least one result.

- [ ] **Step 4: Optional LongMemEval smoke**

Run:

```bash
PYTHONPATH="src/coremem/src:." uv run python tests/evaluation/longmemeval_retrieval.py --limit 20
```

Expected: no broad regression.

---

## Self-Review

- Spec coverage: The plan covers standalone-first refactor, compatibility alias, public embedding helper, MRO check, tests, build, and EA migration.
- Placeholder scan: No placeholders remain.
- Type consistency: `default_embedding_fn`, `_default_embedding_fn`, `SearchMode`, `Column`, mixin names, and public method names match the architecture spec.
