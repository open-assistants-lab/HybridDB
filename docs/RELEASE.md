# HybridDB Release Guide

This guide documents local release steps for HybridDB. PyPI upload requires a token or trusted publishing environment.

## Pre-Release Checklist

Run from `/Users/eddy/Developer/Python/HybridDB`.

```bash
uv run --with ruff ruff check hybriddb tests
uv run python -m pytest -q
uv run python -m pytest tests/benchmarks -q --run-benchmarks --benchmark-disable
```

Expected current results:

```text
ruff: All checks passed
pytest: 292 passed, 48 skipped
benchmark smoke: 48 passed
```

## Interpreter / SQLite Gate (required)

SQLite is **not** a pip dependency — it ships inside the interpreter. So the
SQLite version a user gets is whatever their Python bundles, and a newer Python
means a newer SQLite we never tested. This is the same exposure that let 0.8.1
and 0.8.2 ship broken against the chromadb floor (issue #1), except we do not
control it at all.

Current state: development runs Python 3.13 / SQLite 3.50.4, while Python 3.14
bundles **SQLite 3.53.4**. Anyone on 3.14 was therefore ahead of our test
matrix.

Run the gate before every release:

```bash
uv run python scripts/check_interpreter_matrix.py            # 3.13 + 3.14
uv run python scripts/check_interpreter_matrix.py --quick    # skip perf gates
uv run python scripts/check_interpreter_matrix.py -- -k TestSQLiteCapabilities
```

It runs the suite in isolated environments, prints the resolved Python /
SQLite / chromadb versions so a failure can be attributed, and exits non-zero if
any interpreter fails. The *chromadb floor* check above covers the bottom of the
supported chromadb range; this covers the newest interpreter/SQLite end.

Two consequences worth remembering:

- **FTS5 is a compile-time SQLite option**, not a given. `hybriddb` probes for
  it and raises `FTS5UnavailableError` with an actionable message rather than
  leaking `no such module: fts5` from the DDL. Do not add a code path that
  creates an FTS5 virtual table without going through `_create_fts5()`.
- The perf gates in `tests/test_versioning.py` compare two wall-clock
  measurements, so a single sample sits on the noise floor. The update-heavy
  gate's ingest allowance is **3x**, not 2x: at 2x it missed by 4% on 3.14 in
  the full suite while passing in isolation, and the gate exists to catch a
  ~17x regression, which 3x still catches by a wide margin. **Do not** make the
  gates more stable by repeating the churn/rollback cycle to take a median — each
  round triples the Chroma write volume, which provokes a hard segfault in
  chromadb's native Rust bindings. Widen the allowance instead.

## Engine Accuracy Gate (required when the default embedding engine changes)

The default engine is the bundled uint8 MiniLM. If that engine, its
quantization, or its tokenizer changes, the BEIR delta must be re-measured and
`docs/PERFORMANCE.md` updated before release — a silent quality regression is
exactly what the hash fallback shipped once.

```bash
uv run python -u scripts/beir_bundle_eval.py      # bundled uint8 vs fp32, BEIR
```

Acceptance: bundled-uint8 nDCG/Recall deltas stay within ±0.02 of fp32 on both
datasets (measured 0.10.0: −0.0038 worst case, SciFact hybrid +0.0031). A larger
delta means the engine must not ship as default.

## Declared-Floor Check (required)

Private chromadb internals are reached by the client-invalidation path, so the
**declared floor** must be exercised before every release — not only whichever
chromadb the dev environment happens to have. `SharedSystemClient` gained
`_identifier_to_refcount` in 1.5.2; 1.5.0/1.5.1 have no refcount table, and
unguarded access there broke `force_rebuild_chroma_index()` and `restore()`
(0.8.1/0.8.2, issue #1).

```bash
for CV in 1.5.0 1.5.1; do
  uv run --no-project --isolated --no-cache --refresh \
    --with "chromadb==$CV" \
    --with /Users/eddy/Developer/Python/HybridDB/dist/hybriddb-0.10.0-py3-none-any.whl \
    python -c "
import chromadb, tempfile
from chromadb.api.shared_system_client import SharedSystemClient
from hybriddb import HybridDB, LONGTEXT
def mock(t): return [0.0]*384 if not t else [0.1]*384
with tempfile.TemporaryDirectory() as tmp:
    db = HybridDB(tmp, embedding_fn=mock)
    db.create_table('t', {'id': 'TEXT PRIMARY KEY', 'body': LONGTEXT})
    for i in range(3): db.insert('t', {'id': f'm{i}', 'body': f'msg {i} alpha'})
    assert db.force_rebuild_chroma_index()['status'] == 'rebuilt'
    db.create_table('later', {'id': 'TEXT PRIMARY KEY', 'body': LONGTEXT})
    db.insert('later', {'id': 'x', 'body': 'post rebuild'})
    assert db.count('later') == 1 and db.search('t', 'body', 'alpha')
    db.close()
print('floor ok', chromadb.__version__, hasattr(SharedSystemClient,'_identifier_to_refcount'))
"
done
```

Expected:

```text
floor ok 1.5.0 False
floor ok 1.5.1 False
```

Any new access to a `chromadb` private attribute must be feature-detected with
`getattr` and covered by a test — see the floor test in
`tests/test_regressions.py::TestChromaDirectorySwap`.

## Build

```bash
rm -rf dist
uv build
```

Expected files for version `0.10.0` (the wheel now carries the bundled 23 MB
MiniLM engine, so it is ~16 MB rather than the historical 60–400 KB):

```text
dist/hybriddb-0.10.0.tar.gz          (~17 MB)
dist/hybriddb-0.10.0-py3-none-any.whl (~16 MB)
```

The wheel smoke below deletes `~/.cache/chroma/onnx_models` first — the
bundled engine must work with **no** cached model, or the offline promise is
not being tested.

> Keep `~/.cache/chroma/onnx_models` clean or removed for these checks. If a
> cached model is present, verify the default label is
> `hybriddb:all-MiniLM-L6-v2-uint8` — if resolution silently picked a
> downloaded model instead, bundling has broken.

The wheel smoke test should also use the **default engine** (no `embedding_fn`)
at least once, and exercise a numeric-looking `TEXT` primary key in semantic
mode — both were found defective in the past (#5 exposure and #6).

## Wheel Smoke Test

Run an isolated install test from outside the repo:

```bash
uv run --no-project --isolated --no-cache \
  --with /Users/eddy/Developer/Python/HybridDB/dist/hybriddb-0.10.0-py3-none-any.whl \
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
        node_id = db.graph.add_node(label='Alice', type='person')
        assert db.graph.get_node(node_id)['label'] == 'Alice'
        assert db.olap.query('SELECT COUNT(*) AS total FROM docs')[0]['total'] == 1
        await db.aclose()
    print('wheel smoke ok')

asyncio.run(main())
PY
```

Expected output:

```text
wheel smoke ok
```

## Publish Dry Run

```bash
uv publish --dry-run dist/*
```

Without credentials, this currently reports an OIDC/token error but still checks the package files.

## Publish To PyPI

With a PyPI token:

```bash
UV_PUBLISH_TOKEN=pypi-... uv publish dist/*
```

Or configure trusted publishing in PyPI and run the same command from the trusted CI environment.

## Post-Release Smoke Test

After PyPI release:

```bash
uv run --no-project --isolated --no-cache --with hybriddb==0.10.0 python - <<'PY'
from tempfile import TemporaryDirectory
from hybriddb import HybridDB, LONGTEXT

with TemporaryDirectory() as tmp:
    db = HybridDB(tmp)
    db.create_table('docs', {'body': LONGTEXT})
    db.insert('docs', {'body': 'hello memory'})
    assert db.search('docs', 'body', 'memory')
    print('pypi smoke ok')
PY
```

## Versioning Notes

- `0.3.0` adds the developer-friendly API surface: constants, typed columns, string modes, all-column search shorthand, public cursor, read-only query, async wrappers, graph facade, and OLAP facade.
- Keep `0.x` while the API is still marked alpha.
- Bump minor versions for new public API and patch versions for bug fixes.
