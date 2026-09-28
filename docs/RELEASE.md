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
pytest: 277 passed, 48 skipped
benchmark smoke: 48 passed
```

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
    --with /Users/eddy/Developer/Python/HybridDB/dist/hybriddb-0.8.4-py3-none-any.whl \
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

Expected files for version `0.8.4`:

```text
dist/hybriddb-0.8.4.tar.gz
dist/hybriddb-0.8.4-py3-none-any.whl
```

## Wheel Smoke Test

Run an isolated install test from outside the repo:

```bash
uv run --no-project --isolated --no-cache \
  --with /Users/eddy/Developer/Python/HybridDB/dist/hybriddb-0.8.4-py3-none-any.whl \
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
uv run --no-project --isolated --no-cache --with hybriddb==0.8.4 python - <<'PY'
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
