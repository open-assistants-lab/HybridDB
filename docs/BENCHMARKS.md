# HybridDB Benchmarks

HybridDB keeps benchmarks separate from the default test run.

Default `pytest` skips benchmark tests so the package has a fast, reliable CI-style suite. Benchmark smoke tests and full benchmark runs are opt-in.

## Default Test Suite

```bash
uv run python -m pytest -q
```

Expected shape:

```text
105 passed, 23 skipped
```

The skipped tests are benchmark tests under `tests/benchmarks/`.

## Benchmark Smoke Tests

Run functional benchmark smoke tests without repeated timing loops:

```bash
uv run python -m pytest tests/benchmarks -q --run-benchmarks --benchmark-disable
```

Expected shape:

```text
23 passed
```

Smoke scale is intentionally small:

- `n_docs=50`
- `n_graph_nodes=50`
- `n_graph_edges=150`
- `n_analytics_rows=1000`
- `concurrent_duration_s=1`

This verifies benchmark code paths without spending minutes embedding thousands of documents.

## Split Benchmark Smoke Runs

If a single benchmark command is too slow on a machine, run these groups separately:

```bash
uv run python -m pytest tests/benchmarks/test_search.py -q --run-benchmarks --benchmark-disable
uv run python -m pytest tests/benchmarks/test_analytics.py tests/benchmarks/test_graph.py -q --run-benchmarks --benchmark-disable
uv run python -m pytest tests/benchmarks/test_storage.py tests/benchmarks/test_concurrent.py -q --run-benchmarks --benchmark-disable
```

These are the recommended pre-release benchmark smoke checks.

## Timed Benchmarks

Run timed smoke benchmarks:

```bash
uv run python -m pytest tests/benchmarks -q --run-benchmarks --benchmark-json results/smoke.json
```

Run full-scale benchmarks:

```bash
uv run python -m pytest tests/benchmarks -q --run-benchmarks --benchmark-full --benchmark-json results/full.json
```

Full scale uses:

- `n_docs=100000`
- `n_graph_nodes=10000`
- `n_graph_edges=50000`
- `n_analytics_rows=1000000`
- `concurrent_duration_s=30`

Full benchmark runs are not intended for normal CI.

## Result Archives

When `--benchmark-json` is provided, benchmark results are archived in `results/`:

- Timestamped copy: `results/YYYY-MM-DDTHHMMSS-<git-hash>.json`
- Latest copy: `results/latest.json`

## Notes

- `TEXT` columns benchmark keyword search only.
- `LONGTEXT` columns benchmark keyword, semantic, and hybrid search.
- Concurrent benchmark smoke uses `TEXT` so it tests SQLite/FTS concurrency without introducing ChromaDB embedding work in worker threads.
- Storage benchmarks measure both SQLite files and ChromaDB directories.
