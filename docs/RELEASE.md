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
pytest: 262 passed, 48 skipped
benchmark smoke: 48 passed
```

## Build

```bash
rm -rf dist
uv build
```

Expected files for version `0.5.6`:

```text
dist/hybriddb-0.5.6.tar.gz
dist/hybriddb-0.5.6-py3-none-any.whl
```

## Wheel Smoke Test

Run an isolated install test from outside the repo:

```bash
uv run --no-project --isolated --no-cache \
  --with /Users/eddy/Developer/Python/HybridDB/dist/hybriddb-0.5.6-py3-none-any.whl \
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
uv run --no-project --isolated --no-cache --with hybriddb==0.5.6 python - <<'PY'
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
