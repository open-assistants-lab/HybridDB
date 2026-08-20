#!/usr/bin/env python
"""Analytics performance sweep: DuckDB mirror vs raw SQLite.

Builds tables at increasing sizes and measures the same queries on both
engines, to find where the columnar DuckDB mirror starts paying off and
what it costs on writes.

Usage:
    uv run python scripts/analytics_sweep.py [--max-rows 1000000]
"""

import argparse
import statistics
import tempfile
import time

from hybriddb import HybridDB
from tests.benchmarks.helpers import generate_analytics_data

QUERIES = {
    "full_scan_agg": "SELECT COUNT(*) AS cnt, AVG(value) AS avg_val, SUM(quantity) AS total_qty FROM analytics",
    "group_by": "SELECT category, COUNT(*) AS cnt, AVG(value) AS avg_val FROM analytics GROUP BY category ORDER BY cnt DESC",
    "filtered_agg": "SELECT COUNT(*) AS cnt FROM analytics WHERE value > 100 AND region = 'R1'",
    "join": "SELECT a.category, m.label, COUNT(*) AS cnt FROM analytics a JOIN metadata m ON a.category = m.category GROUP BY a.category, m.label ORDER BY cnt DESC",
    "point_lookup": "SELECT * FROM analytics WHERE id = 5",
}

SIZES = [10_000, 100_000, 1_000_000]


def median_time(fn, repeats: int = 5) -> float:
    fn()  # warm
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def build_db(n: int) -> HybridDB:
    tmp = tempfile.mkdtemp(prefix="hdb_sweep_")
    db = HybridDB(tmp, max_chroma_index_gb=0)
    db.create_table("analytics", {
        "category": "TEXT", "region": "TEXT", "value": "REAL", "quantity": "INTEGER",
    })
    rows = generate_analytics_data(n)
    t0 = time.perf_counter()
    db.insert_batch("analytics", rows, sync=False)
    db.process_journal()
    t_insert = time.perf_counter() - t0
    db.create_table("metadata", {"category": "TEXT", "label": "TEXT"})
    db.insert_batch("metadata", [
        {"category": c, "label": f"Category {c}"} for c in ["A", "B", "C", "D", "E"]
    ], sync=False)
    t0 = time.perf_counter()
    db.register_duckdb_table("analytics")
    db.register_duckdb_table("metadata")
    t_sync = time.perf_counter() - t0
    return db, t_insert, t_sync


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    args = parser.parse_args()
    sizes = [s for s in SIZES if s <= args.max_rows]

    print(f"{'rows':>10} | {'query':<16} | {'sqlite':>10} | {'duckdb':>10} | {'speedup':>8}")
    print("-" * 66)
    for n in sizes:
        db, t_insert, t_sync = build_db(n)
        for qname, sql in QUERIES.items():
            t_sqlite = median_time(lambda: db.raw_query(sql))
            t_duck = median_time(lambda: db.analytics(sql))
            ratio = t_sqlite / t_duck if t_duck > 0 else float("inf")
            print(f"{n:>10,} | {qname:<16} | {t_sqlite*1000:>8.2f}ms | {t_duck*1000:>8.2f}ms | {ratio:>7.2f}x")
        print(f"{n:>10,} | {'insert_batch':<16} | {t_insert*1000:>8.1f}ms | {'':>10} | {'':>8}")
        print(f"{n:>10,} | {'full_sync':<16} | {'':>10} | {t_sync*1000:>8.1f}ms | {'':>8}")
        print("-" * 66)


if __name__ == "__main__":
    main()
