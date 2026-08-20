"""DuckDB analytics benchmarks: aggregation, group-by, join, overhead."""

import pytest

from .helpers import generate_analytics_data

pytest.importorskip("duckdb")


@pytest.fixture
def analytics_db(db, scale):
    rows = generate_analytics_data(scale.n_analytics_rows)
    db.create_table("analytics", {
        "category": "TEXT",
        "region": "TEXT",
        "value": "REAL",
        "quantity": "INTEGER",
        "timestamp": "TEXT",
    })
    db.insert_batch("analytics", rows, sync=False)
    db.create_table("metadata", {"category": "TEXT", "label": "TEXT"})
    db.insert_batch(
        "metadata",
        [{"category": cat, "label": f"Category {cat}"} for cat in ["A", "B", "C", "D", "E"]],
        sync=False,
    )
    db.register_duckdb_table("analytics")
    db.register_duckdb_table("metadata")
    return db


def test_simple_aggregation(benchmark, analytics_db):
    def _agg():
        return analytics_db.analytics(
            "SELECT COUNT(*) as cnt, AVG(value) as avg_val, SUM(quantity) as total_qty FROM analytics"
        )

    result = benchmark(_agg)
    assert len(result) == 1
    assert result[0]["cnt"] > 0


def test_group_by(benchmark, analytics_db):
    def _gb():
        return analytics_db.analytics(
            "SELECT category, COUNT(*) as cnt, AVG(value) as avg_val "
            "FROM analytics GROUP BY category ORDER BY cnt DESC"
        )

    result = benchmark(_gb)
    assert len(result) > 0


def test_join(benchmark, analytics_db):
    def _join():
        return analytics_db.analytics(
            "SELECT a.category, m.label, COUNT(*) as cnt "
            "FROM analytics a JOIN metadata m ON a.category = m.category "
            "GROUP BY a.category, m.label ORDER BY cnt DESC"
        )

    result = benchmark(_join)
    assert len(result) > 0


def test_analytics_overhead(benchmark, analytics_db, scale):
    """Compare HybridDB.analytics() vs native DuckDB overhead ratio."""
    sql = "SELECT COUNT(*) FROM analytics WHERE value > 100"

    def _hybrid():
        return analytics_db.analytics(sql)

    result = benchmark(_hybrid)
    assert len(result) == 1


# ── DuckDB vs SQLite: the OLAP value proposition ─────────────────────────

QUERIES = {
    "full_scan_agg": "SELECT COUNT(*) AS cnt, AVG(value) AS avg_val, SUM(quantity) AS total_qty FROM analytics",
    "group_by": "SELECT category, COUNT(*) AS cnt, AVG(value) AS avg_val FROM analytics GROUP BY category ORDER BY cnt DESC",
    "filtered_agg": "SELECT COUNT(*) AS cnt FROM analytics WHERE value > 100 AND region = 'R1'",
    "join": "SELECT a.category, m.label, COUNT(*) AS cnt FROM analytics a JOIN metadata m ON a.category = m.category GROUP BY a.category, m.label ORDER BY cnt DESC",
    "point_lookup": "SELECT * FROM analytics WHERE id = 5",
}


@pytest.mark.parametrize("qname", list(QUERIES))
def test_duckdb_vs_sqlite(benchmark, analytics_db, qname):
    """Same query on SQLite (raw) vs the DuckDB mirror — speedup + correctness."""
    sql = QUERIES[qname]

    def _sqlite():
        return analytics_db.raw_query(sql)

    def _duckdb():
        return analytics_db.analytics(sql)

    sqlite_rows = _sqlite()
    duck_rows = _duckdb()
    # correctness: identical result sets (order-insensitive, float-tolerant)
    def _sort_key(row):
        return tuple(str(v) for v in row.values())

    sqlite_rows = sorted(sqlite_rows, key=_sort_key)
    duck_rows = sorted(duck_rows, key=_sort_key)
    assert len(sqlite_rows) == len(duck_rows)
    for a, b in zip(sqlite_rows, duck_rows):
        assert set(a) == set(b)
        for key in a:
            if isinstance(a[key], float):
                assert abs(a[key] - b[key]) < 1e-6
            else:
                assert a[key] == b[key]

    def _run(fn):
        fn()  # warm
        import time
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    t_sqlite = _run(_sqlite)
    t_duck = _run(_duckdb)
    ratio = t_sqlite / t_duck if t_duck > 0 else float("inf")
    print(f"  {qname:16s}: sqlite={t_sqlite*1000:.2f}ms  duckdb={t_duck*1000:.2f}ms  "
          f"speedup={ratio:.2f}x")
    benchmark.pedantic(_duckdb, rounds=5)


# ── Mirror sync overhead ──────────────────────────────────────────────────

def test_sync_overhead(benchmark, db, scale):
    """Write cost with vs without the DuckDB mirror (incremental journal sync)."""
    rows = generate_analytics_data(scale.n_analytics_rows)
    db.create_table("analytics", {
        "category": "TEXT", "region": "TEXT", "value": "REAL", "quantity": "INTEGER",
    })
    db.insert_batch("analytics", rows, sync=False)
    db.process_journal()
    # fresh rows with new ids for the incremental-write measurement
    counter = [0]

    def _fresh_rows():
        base = scale.n_analytics_rows + counter[0] * 100
        counter[0] += 1
        return [dict(r, id=base + i) for i, r in enumerate(rows[:100])]

    def _no_mirror():
        db.insert_batch("analytics", _fresh_rows(), sync=False)
        db.process_journal()

    db.register_duckdb_table("analytics")

    def _with_mirror():
        db.insert_batch("analytics", _fresh_rows(), sync=False)
        db.process_journal()

    _no_mirror()  # warm
    import time
    t0 = time.perf_counter(); _no_mirror(); t_no = time.perf_counter() - t0
    t0 = time.perf_counter(); _with_mirror(); t_with = time.perf_counter() - t0
    print(f"  insert 100 rows + journal: without mirror={t_no*1000:.2f}ms  "
          f"with mirror={t_with*1000:.2f}ms  overhead={max(0, t_with-t_no)*1000:.2f}ms")
    # mirror must stay correct after the writes
    assert db.olap.query("SELECT COUNT(*) c FROM analytics")[0]["c"] == db.count("analytics")
    benchmark(_with_mirror)


def test_full_sync_cost(benchmark, db, scale):
    """Cost of registering a table (full copy into the DuckDB mirror)."""
    rows = generate_analytics_data(scale.n_analytics_rows)
    db.create_table("analytics", {
        "category": "TEXT", "region": "TEXT", "value": "REAL", "quantity": "INTEGER",
    })
    db.insert_batch("analytics", rows, sync=False)
    db.process_journal()

    def _register():
        db.register_duckdb_table("analytics")

    _register()  # warm (also verifies it works)
    import time
    t0 = time.perf_counter(); _register(); t = time.perf_counter() - t0
    print(f"  full sync of {scale.n_analytics_rows} rows: {t*1000:.1f}ms  "
          f"({scale.n_analytics_rows / max(t, 1e-9) / 1e6:.1f}M rows/s)")
    assert db.olap.query("SELECT COUNT(*) c FROM analytics")[0]["c"] == scale.n_analytics_rows
    benchmark(_register)
