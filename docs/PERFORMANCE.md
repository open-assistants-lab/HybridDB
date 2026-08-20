# HybridDB Performance Study

Measured on 2026-08-20, commit `00a361c`, on a MacBook Pro (Apple Silicon,
CPython 3.13). Search accuracy uses real IR benchmarks (BEIR); speed numbers
are smoke-scale (50 docs / 50 graph nodes / 1,000 analytics rows) unless noted.
Full-scale runs (`--benchmark-full`) are available but were not part of this
study.

## 1. Search Accuracy (BEIR)

Evaluated on two BEIR datasets with real queries and relevance judgments:
**NFCorpus** (3,633 medical docs, 324 queries, graded 0-3) and **SciFact**
(5,183 scientific abstracts, 301 queries, binary). Metrics are averaged over
all queries. Embeddings: ChromaDB's bundled MiniLM.

| Dataset | Mode | nDCG@10 | Recall@10 | P@10 | MRR |
|---|---|---|---|---|---|
| NFCorpus | keyword | 0.3083 | 0.1489 | 0.2167 | 0.5128 |
| NFCorpus | semantic | 0.3145 | 0.1542 | 0.2418 | 0.5063 |
| NFCorpus | **hybrid** | **0.3429** | **0.1699** | **0.2467** | **0.5548** |
| SciFact | keyword | 0.6683 | 0.7956 | 0.0883 | 0.6348 |
| SciFact | semantic | 0.6484 | 0.7883 | 0.0890 | 0.6068 |
| SciFact | **hybrid** | **0.7031** | **0.8443** | **0.0943** | **0.6649** |

**Fusion-weight sensitivity** (hybrid nDCG@10):

| fts_weight | 0.2 | 0.5 (default) | 0.8 |
|---|---|---|---|
| NFCorpus | 0.3271 | **0.3429** | 0.3332 |
| SciFact | 0.6741 | **0.7031** | 0.6948 |

**Embedding model gap** (NFCorpus, semantic):

| Embedding | nDCG@10 | Recall@10 | MRR |
|---|---|---|---|
| MiniLM | 0.3145 | 0.1542 | 0.5063 |
| hash fallback | 0.0593 | 0.0292 | 0.1213 |

### Findings

1. **Hybrid fusion is the right default.** It beats both single modes on
   every metric in both domains (+11% nDCG over keyword on NFCorpus, +5% on
   SciFact). The RRF fusion captures the lexical and semantic regimes.
2. **`fts_weight=0.5` is the sweet spot and the curve is flat** (spread
   < 0.03) — no tuning needed.
3. **The hash embedding fallback is a 5.3× accuracy cliff** (nDCG 0.059 vs
   0.315, near-random). Semantic search without a real embedding model is not
   viable; the fallback exists for offline smoke tests, not production.

## 2. Analytics: DuckDB Mirror vs Raw SQLite

Same queries on the SQLite table (`raw_query`) and the DuckDB mirror
(`olap.query`), at three sizes. Median of 5 runs.

| Rows | Query | SQLite | DuckDB | Speedup |
|---|---|---|---|---|
| 10k | full-scan agg | 1.46ms | 0.24ms | 6.2× |
| 10k | group-by | 3.42ms | 0.47ms | 7.3× |
| 10k | filtered agg | 1.54ms | 0.24ms | 6.5× |
| 10k | join | 5.46ms | 1.17ms | 4.7× |
| 10k | point lookup | 0.79ms | 0.20ms | 4.0× |
| 100k | full-scan agg | 7.67ms | 0.38ms | 20.3× |
| 100k | group-by | 34.13ms | 1.37ms | 24.9× |
| 100k | filtered agg | 8.85ms | 0.19ms | 46.4× |
| 100k | join | 58.29ms | 4.17ms | 14.0× |
| 100k | point lookup | 0.90ms | 0.28ms | 3.2× |
| 1M | full-scan agg | 70.98ms | 1.23ms | 57.6× |
| 1M | group-by | 378.31ms | 2.83ms | 133.6× |
| 1M | filtered agg | 76.86ms | 0.22ms | **351.9×** |
| 1M | join | 640.17ms | 8.45ms | 75.8× |
| 1M | point lookup | 1.14ms | 0.47ms | 2.4× |

**Mirror maintenance costs:**

| Rows | Full sync (register) | Incremental sync overhead |
|---|---|---|
| 10k | 63ms | within noise |
| 100k | 92ms | within noise |
| 1M | 614ms (1.6M rows/s) | within noise |

**Write path** (insert_batch + journal, no embeddings): ~2,000 rows/s
(0.5ms/row) — dominated by per-row Python + journal overhead, not by the
mirror.

### Findings

1. **The DuckDB mirror pays off immediately and compounds with scale.**
   Even at 10k rows it is 4-7× faster; at 1M rows, 58-352×. There is no
   "SQLite wins at small scale" regime for analytical queries.
2. **Filtered aggregation is the columnar sweet spot** (352× at 1M rows) —
   vectorized scans over `WHERE` clauses.
3. **Point lookups are the mirror's weak spot** (2.4× at 1M) — the mirror
   has no index. Use `get()`/`query()` for point lookups; the mirror is for
   analytics.
4. **The mirror is cheap to maintain**: full sync copies 1M rows in ~0.6s,
   and incremental journal sync adds no measurable write overhead.
5. **The journal is the write bottleneck** (~2k rows/s), not SQLite or the
   mirror. Per-row overhead: dict filtering, a post-insert `SELECT`, journal
   INSERTs, and JSON serialization. Optimization candidates: batch journal
   writes, drop the per-row re-select (use `lastrowid` + known columns).
6. **Fidelity fix found by the benchmark**: the mirror stored `REAL` columns
   as DuckDB `REAL` (float32), losing precision vs SQLite's float64
   (111.19 → 111.19000244140625). Now mapped to `DOUBLE`.

### Write-path optimization (2026-08-20)

Profiling `insert_batch` revealed the real bottleneck was not the journal
writes but **opening a fresh SQLite connection per row**: `_table_meta` and
`_get_longtext_columns` were called per row without the batch's cursor,
each opening + closing a connection (~200k connections per 100k rows).

Fixes:
- Thread the batch cursor through `_row_to_metadata` / `_get_longtext_columns`
  in the CRUD hot paths; hoist per-row schema lookups out of the loop
- `insert()` now does its schema reads inside its single connection
- `_process_journal` uses one connection end-to-end (reads + apply + delete)
  instead of three

| Path | Before | After | Speedup |
|---|---|---|---|
| `insert_batch` 100k rows (no embeddings) | 1,860 rows/s | **24,500 rows/s** | 13.2× |
| single `insert` (sync=True) | 166 rows/s | **313 rows/s** | 1.9× |
| `insert_batch` + LONGTEXT + DuckDB mirror | — | 11,400 rows/s | embedding-bound |

### `sync=True` now actually means synced (2026-08-20)

A FULL-scale benchmark run exposed that `insert_batch(sync=True)` on a batch
larger than the journal's per-call limit (5,000 entries) left the rest
pending — so *every subsequent search silently paid a 35–40s journal-flush
embedding cost* until the backlog drained. `sync=True` now drains the
journal fully before returning (the honest cost is paid upfront, in the
write call); `sync=False` + `process_journal()` keeps the bounded-batch
progressive behavior for callers who want control.

### FULL-scale scaling validation (100k docs / 10k graph nodes / 1M analytics rows)

| Operation | Smoke | FULL | Scaling |
|---|---|---|---|
| keyword search (TEXT) | 7.0ms | 53ms | 7.6× for 2,000× data |
| vector search (TEXT) | 1.9ms | 38ms | ~sub-linear |
| hybrid search (TEXT) | 5.2ms | 52ms | ~sub-linear |
| analytics agg (1M rows) | 0.2ms | 1.7ms | vectorized |
| analytics group-by (1M) | 0.4ms | 2.4ms | vectorized |
| analytics join (1M) | 0.7ms | 8.3ms | vectorized |
| get_neighbors (10k nodes) | 0.9ms | 7.7ms | ✓ |
| pagerank (10k nodes) | 0.5ms | 51ms | ✓ |
| traverse depth 3 (50k edges) | 2.5ms | 0.8s | CTE breadth |
| to_networkx build (10k) | 2.7ms | 297ms | ✓ (cache ~0ms) |
| sync_graph_nodes (10k) | 43ms | 5.9s | 0.6ms/node |
| community_detect (10k) | 2.3ms | 1.3s | ✓ |
| betweenness (1k nodes) | 6.9ms | 4.5s | O(N·E) — cap size |
| search_graph (10k) | 52ms | 6.1s | seed+expand |
| search_graph_ppr (10k) | 52ms | 5.9s | seed+PPR |

Search latency stays sub-100ms at 100k docs; analytics stays single-digit ms
at 1M rows; the graph scales linearly-ish with nodes/edges, with semantic
retrieval (search_graph/PPR) and traversal as the dominant costs at 10k-node
scale.

## 3. Graph Performance (smoke scale: 50 nodes / 150 edges)

| Operation | Mean |
|---|---|
| add_nodes batch (50) | 37.1ms |
| add_edges batch (150) | 100.4ms |
| get_neighbors | 0.9ms |
| traverse (depth 3, both) | 2.5ms |
| shortest_path | 0.01ms |
| pagerank | 0.5ms |
| community_detect (louvain) | 2.3ms |
| betweenness_centrality (1k nodes) | 6.9ms |
| to_networkx build | 2.7ms |
| to_networkx cache hit | ~0ms |
| sync_graph_nodes (50) | 43.0ms |
| **search_graph** | **52.3ms** |
| **search_graph_ppr** | **52.4ms** |

### Findings

1. **Semantic graph retrieval (search_graph / PPR) is the dominant graph
   cost** (~52ms at smoke scale) — the vector seed search + hop expansion +
   PageRank pipeline. Everything else is sub-10ms.
2. **The NetworkX cache works**: cache hits are ~0ms vs 2.7ms builds.
3. **Graph sync is O(n) with per-node round trips** (43ms for 50 nodes ≈
   0.9ms/node) — the dominant cost at scale; a batch upsert would help.

## 4. Search Speed (smoke scale: 50 docs)

| Operation | Mean |
|---|---|
| keyword search (TEXT) | 7.0ms |
| vector search (TEXT) | 1.9ms |
| hybrid search (TEXT) | 5.2ms |
| keyword search (LONGTEXT) | 5.3ms |
| vector search (LONGTEXT) | 1.7ms |
| hybrid search (LONGTEXT) | 2.1ms |
| cold-start search (first query) | 25.9ms |

## 5. Recommendations

1. **Keep hybrid + fts_weight=0.5 as the default** — it is the accuracy
   sweet spot in both domains with flat sensitivity.
2. **Warn loudly when falling back to hash embeddings** — the accuracy cliff
   (5.3×) makes silent fallback a footgun.
3. **Route analytical queries to `olap.query`** — the mirror is 6-350×
   faster and free to maintain; keep point lookups on SQLite.
4. **Optimize the journal write path** (~2k rows/s ceiling): batch journal
   inserts and eliminate the per-row post-insert `SELECT`.
5. **Batch graph sync** (0.9ms/node) for large registrations.

## Reproducing

```bash
# accuracy (downloads BEIR datasets on first run)
uv run python -m pytest tests/benchmarks/test_accuracy.py -q --run-benchmarks --benchmark-disable

# analytics + graph + search speed (smoke)
uv run python -m pytest tests/benchmarks -q --run-benchmarks --benchmark-json results/smoke.json

# DuckDB vs SQLite crossover sweep
uv run python scripts/analytics_sweep.py
```
