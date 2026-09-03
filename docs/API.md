# HybridDB API Reference

This document describes the stable public API for HybridDB `0.8.x`.

HybridDB is one embedded database object that coordinates SQLite, FTS5, ChromaDB, a self-healing journal, optional DuckDB analytics, and optional graph helpers.

> Source of truth: the code in `hybriddb/`. This file is verified against the
> implementation and the changelog as of `0.8.0` (2026-09-03).

## Imports

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
    EmbeddingModelError,
    HybridDB,
    SearchMode,
    default_embedding_fn,
)
```

Constants: `TEXT` and `LONGTEXT` enable FTS5 (`LONGTEXT` also enables ChromaDB
semantic search); `INTEGER`, `REAL`, `BOOLEAN`, `JSON` are plain SQLite columns.
`KEYWORD`, `SEMANTIC`, `HYBRID` are `SearchMode` enum aliases. `Column(type,
constraints)` is a typed schema helper for `create_table()`.
`EmbeddingModelError` is raised when the configured embedding model cannot be
loaded; `default_embedding_fn` is ChromaDB's bundled local MiniLM embedding.

## Constructor

```python
db = HybridDB(
    path="./data",
    embedding_fn=None,
    embedding_model_name=None,
    force_model=False,
    max_chroma_index_gb=5,
    auto_rebuild_chroma=False,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | — | Directory for the SQLite file (`app.db`) and the vector store (`vectors/`). Created if missing. |
| `embedding_fn` | `callable \| None` | `None` | Custom embedding function. Defaults to ChromaDB's bundled local MiniLM. |
| `embedding_model_name` | `str \| None` | `None` | Label recorded for the embedding. Defaults to `chroma:all-MiniLM-L6-v2`, or `"custom"` when `embedding_fn` is provided. |
| `force_model` | `bool` | `False` | Skip the embedding-model mismatch check on init — use when you deliberately swapped `embedding_fn` for an existing store. |
| `max_chroma_index_gb` | `int` | `5` | Guardrail for local disk usage by the vector index. |
| `auto_rebuild_chroma` | `bool` | `False` | Rebuild the Chroma index on startup when a corruption check trips. |

A hash embedding is used only as a fallback if ChromaDB's default embedding
cannot load — it exists for offline smoke tests, not production (measured at a
5.3× accuracy cliff on BEIR; see `docs/PERFORMANCE.md`).

Custom embedding:

```python
db = HybridDB(
    "./data",
    embedding_fn=lambda text: my_model.encode(text),
    embedding_model_name="my-model",
)
```

## Schema

```python
db.create_table("docs", {"title": TEXT, "body": LONGTEXT, "tags": JSON})
db.create_table("typed_docs", {"title": Column(TEXT), "body": Column(LONGTEXT)})
db.create_table("memories", {"id": TEXT, "content": LONGTEXT},
                versioned=True, hash_chain=True)
```

Column types:

| Type | SQLite | FTS5 | ChromaDB | Use for |
|------|--------|------|----------|---------|
| `TEXT` | TEXT | yes | no | names, titles, short strings |
| `LONGTEXT` | TEXT | yes | yes | documents, messages, memory content |
| `INTEGER` | INTEGER | no | no | counts and IDs |
| `REAL` | REAL | no | no | scores, prices, measurements |
| `BOOLEAN` | INTEGER | no | no | flags |
| `JSON` | TEXT | no | no | metadata |

Schema methods:

```python
db.create_table("docs", {"title": TEXT, "body": LONGTEXT})
db.add_column("docs", "summary", LONGTEXT)
db.rename_column("docs", "summary", "abstract")
db.drop_column("docs", "abstract")
schema = db.get_schema("docs")
tables = db.list_tables()
db.is_versioned("docs")   # -> bool
```

Public methods validate table and column identifiers. Use simple Python
identifiers such as `docs`, `messages`, `content`, or `created_at`.

Notes:

- Schema changes (`add_column` / `drop_column` / `rename_column`) are
  **rejected on versioned tables**.
- Creating a table with an `id` column that lacks `PRIMARY KEY` raises a
  clear error (it collides with the implicit auto-increment `id`).
- Primary keys on any column name are supported (`my_pk INTEGER PRIMARY KEY`).

## CRUD

```python
row_id = db.insert("docs", {"title": "Hello", "body": "Hybrid search memory"})
rows = db.insert_batch("docs", [{"title": "A", "body": "..."}, {"title": "B", "body": "..."}])

row = db.get("docs", row_id)
ok = db.update("docs", row_id, {"title": "Updated"})
deleted = db.delete("docs", row_id)
total = db.count("docs")

pk = db.upsert("docs", {"id": 1, "title": "Hello", "body": "..."})
```

- `insert_batch()` returns `list[int | str]` — strings when the table uses
  `id TEXT PRIMARY KEY`. Batches larger than the journal cap (5,000 entries)
  log a warning; with `sync=True` the journal is fully drained before
  returning (since 0.5.7 large batches no longer leave a backlog).
- `upsert()` inserts the row if its primary key is absent, else updates it.
  It requires the primary key column in `data` (a missing pk raises a clear
  `ValueError`). On versioned tables the prior state is captured in history
  automatically.
- Explicitly provided primary key values are honored on insert/update,
  including moving the row across Chroma and DuckDB when the pk changes.

## Query

```python
rows = db.query(
    "docs",
    where="title LIKE ?",
    params=("%hello%",),
    order_by="title ASC",
    limit=100,
)
```

For custom read-only SQL, use `read_query()`:

```python
rows = db.read_query("SELECT title FROM docs WHERE title LIKE ?", ("%hello%",))
```

`read_query` is enforced read-only at the SQLite level via an authorizer
(WITH-clause write bypasses are closed).

For advanced migrations or custom writes, use `raw_query()` or the public
cursor context manager:

```python
with db.cursor() as cur:
    cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_title ON docs(title)")
```

`connect()` is an alias for `cursor()`.

## Search

Search one column:

```python
db.search("docs", "body", "how do I get started?")
```

Search every searchable text column:

```python
db.search_all("docs", "getting started")
db.search_columns("docs", "getting started")
```

Modes:

```python
db.search("docs", "body", "hello", mode="keyword",
          where={"user_id": "u2"})   # scalar-column pre-filter at the Chroma level
db.search("docs", "body", "how do I begin?", mode="semantic")
db.search("docs", "body", "getting started", mode="hybrid")

db.search("docs", "body", "hello", mode=SearchMode.KEYWORD)
db.search("docs", "docs", "hello", mode=KEYWORD)
db.search("docs", "body", "hello", mode=HYBRID)
```

Full signature:

```python
db.search(
    table, column, query=None,
    mode=SearchMode.HYBRID, limit=10,
    fts_weight=0.5, recency_weight=0.0, recency_column=None,
    query_embedding=None, where=None,
)
```

| Parameter | Default | Description |
|---|---|---|
| `mode` | `hybrid` | `keyword` (BM25 lexical), `semantic` (vector ANN), or `hybrid` (RRF fusion of both). |
| `limit` | `10` | Maximum number of results. |
| `fts_weight` | `0.5` | Keyword-vs-semantic weight in the RRF fusion. Measured sweet spot — the accuracy curve is flat within ±0.03 either side. |
| `recency_weight` / `recency_column` | `0.0` / `None` | Boost recent content over older content. |
| `query_embedding` | `None` | Pre-computed query embedding; skips the embedding call. |
| `where` | `None` | Scalar-column filters pushed into the Chroma scan **before** the vector query. See [Metadata pre-filtering](#metadata-pre-filtering-multi-tenant-scoping). |

Behavior:

- `TEXT` columns support keyword search.
- `LONGTEXT` columns support keyword, semantic, and hybrid search.
- Hybrid search fuses keyword and vector results using reciprocal-rank fusion.
- Empty queries return `[]`.
- `search(..., query=None)` skips search entirely and returns the latest rows
  ordered by primary key.
- Pending journal entries for the table are processed before searching.
- If the Chroma index is unavailable for a table, semantic and hybrid modes
  degrade to keyword automatically.

`search_all(table, query, where=None, limit=10, fts_weight=0.5)` and
`search_columns(...)` accept the same `where` / `fts_weight` / `limit`
parameters.

Recency scoring:

```python
results = db.search(
    "messages",
    "content",
    "project update",
    recency_weight=0.3,
    recency_column="created_at",
)
```

## Journal And Maintenance

HybridDB journals ChromaDB and DuckDB mutations in SQLite. By default, inserts process the journal immediately.

```python
db.insert_batch("docs", rows, sync=False)
pending = db.journal_status("docs")
processed = db.process_journal(limit=5000)
```

Batches larger than 5,000 rows log a warning recommending deferred sync.

Health and repair:

```python
health = db.health("docs")
# {"sqlite_rows": 5000, "chroma_docs": {"contacts_bio": 5000}, "status": "ok"}

result = db.reconcile("docs")
# {"ghosts_deleted": 0, "missing_added": 3, "metadata_updated": 0}
```

`reconcile()` repairs missing ChromaDB documents, removes ghosts, and
refreshes graph-derived state.

Maintenance and backup:

```python
db.backup(path)                    # copy the entire database directory atomically
db.restore(path)                   # replace the current database from a backup
db.vacuum()                        # reclaim disk space (rebuilds the SQLite file)
report = db.check_integrity()      # diagnostics across SQLite, ChromaDB, DuckDB
db.reindex(table=None)             # rebuild Chroma + FTS5 + DuckDB from SQLite data
db.force_rebuild_chroma_index()    # drop and rebuild the Chroma index
db.stats()                         # size and count statistics for all storage layers
db.close()                         # close handles
```

Export/import as portable SQL (FTS5 is excluded from dumps and rebuilt on
import):

```python
db.export_sql("dump.sql")
db.import_sql("dump.sql")
```

## Async API

Async methods are wrappers around the sync API using worker threads. They are useful in FastAPI and other async applications because they avoid blocking the event loop while SQLite/ChromaDB work runs.

```python
await db.acreate_table("messages", {"content": LONGTEXT})
row_id = await db.ainsert("messages", {"content": "hello from async"})
row = await db.aget("messages", row_id)
results = await db.asearch("messages", "content", "hello")
total = await db.acount("messages")
await db.aclose()
```

Available async methods:

- `acreate_table`, `aadd_column`, `adrop_column`, `arename_column`
- `ainsert`, `ainsert_batch`, `aupdate`, `adelete`, `aget`
- `aquery`, `aread_query`, `araw_query`, `acount`
- `asearch`, `asearch_all`
- `ahealth`, `areconcile`, `aprocess_journal`, `aclose`

Thread safety:

- HybridDB uses an internal `RLock` around SQLite and DuckDB access.
- ChromaDB calls are coordinated through the journal and per-instance operations.
- For high-write workloads, prefer `insert_batch(..., sync=False)` plus `process_journal()`.
- **One store per process.** A second process (dashboard, worker) should not open the same
  database read-write; it can read the SQLite file read-only instead (WAL allows concurrent
  readers), e.g. by attaching it from its own DuckDB instance. DuckDB mirrors are per-process
  and rebuilt cheaply on demand (see `docs/PERFORMANCE.md`).

## Graph API

Graph helpers are available directly and through the `db.graph` facade.
Synced node ids are **namespaced by table**: `{table}:{pk}` (e.g. `docs:1`,
`items:a1`) — manual edges between synced nodes must use namespaced ids.

```python
alice = db.graph.add_node(label="Alice", type="person")     # auto-generated id
bob   = db.graph.add_node("bob-1", label="Bob", type="person")
db.graph.add_edge(None, alice, bob, edge_type="knows", weight=0.9)

neighbors = db.graph.get_neighbors(alice, direction="both")
path = db.graph.shortest_path(alice, bob)
scores = db.graph.pagerank()                                 # standard PageRank
scores = db.graph.pagerank(personalization={"docs:1": 1.0}, alpha=0.85)

# semantic graph retrieval: vector-search seeds, then expand via PageRank
ppr = db.graph.search_graph_ppr("memory", hop_expansion=2, limit=5)
# spread from 20 seeds but return top-5
ppr = db.graph.search_graph_ppr("memory", k_seeds=20, limit=5)

# re-sync registered table rows into graph nodes (also removes ghost nodes)
synced = db.graph.sync_graph_nodes()
```

Node/edge inventory:

| Method | Purpose |
|---|---|
| `add_node(id=None, label="", **kw)` / `add_nodes(nodes)` | Create nodes (re-adding an existing id preserves its edges) |
| `get_node(id)` / `update_node(id, data)` / `delete_node(id)` | Node lifecycle |
| `list_nodes(...)` | Enumerate nodes |
| `add_edge(id=None, source, target, type="relates_to", weight=1.0, properties=None, valid_until=None)` / `add_edges(edges)` | Edges with optional expiry |
| `get_edge(id)` / `get_edges(source=, target=, type=, limit=)` / `update_edge(id, data)` / `delete_edge(id)` | Edge lifecycle |
| `neighbors(id, direction="both", type=None)` / `get_neighbors(...)` | Adjacency |
| `traverse(start_id, max_depth=3, direction="out", type=None, max_cost=3.0)` | Recursive CTE traversal with cost cap |
| `shortest_path(source, target)` | Weighted shortest path |
| `pagerank(personalization=None, alpha=0.85)` | Standard or personalized PageRank |
| `betweenness_centrality()` / `community_detect()` / `connected_components()` | NetworkX algorithms |
| `decay_edges()` | Age-based edge decay |
| `to_networkx(directed=True)` | Export for custom algorithms |

Graph-aware semantic retrieval:

```python
# vector-search seeds, then expand neighbors
db.search_graph("memory", hop_expansion=2, limit=10)

# vector seeds -> subgraph expansion -> Personalized PageRank
ppr = db.graph.search_graph_ppr(
    "memory",
    hop_expansion=2,       # traversal depth from seeds
    limit=10,              # final results
    alpha=0.15,            # damping: lower = more concentrated near seeds
    min_similarity=0.0,    # seed filter (keep 0.0 for MiniLM — distances hover near 1.0)
    k_seeds=None,          # separate seed count from result count
)
```

Auto-sync rules let table rows become graph nodes/edges automatically:

```python
db.graph.register_entity_node("docs", type="entity", id_column="id",
                              label_template="docs: {title}")
db.graph.register_edge_rule("messages", "docs", edge_type="mentions")
db.graph.sync_graph_nodes()   # refreshes labels, removes ghost nodes
```

- Synced ids are namespaced (`docs:1`); label templates render every
  `{column}` placeholder from the row.
- `search_graph_ppr` runs PageRank on the undirected subgraph (consistent
  with `direction="both"` traversal).
- Edge rules use each table's real primary key column.

## Versioned Tables

Opt-in per table. Versioned tables keep an append-only, hash-chained history
(`{table}__history`) of every insert/update/delete, while the main table
stays the current state — FTS5 and Chroma keep indexing current data only.

```python
db.create_table("docs", {"id": TEXT, "body": LONGTEXT}, versioned=True, hash_chain=True)
db.author = "agent-1"                       # optional, recorded per event

db.upsert("docs", {"id": 1, "body": "v1"})  # insert-or-update
db.upsert("docs", {"id": 1, "body": "v2"})

db.log("docs", limit=100)                   # change log, newest first
db.history("docs", key=1)                   # every version of a row
db.diff("docs", from_seq=1, to_seq=2)       # added/removed/changed
db.as_of("docs", seq=1)                     # point-in-time read

cp = db.checkpoint("docs", "before-edit")
db.rollback("docs", checkpoint="before-edit")   # state re-applied as new versions
db.verify_chain("docs")                     # -> {"valid": True, "checked": N, ...}
db.archive("docs", "exports/docs", format="parquet")  # or "jsonl"
db.prune("docs", before_seq=5)              # retention; keeps the chain verifiable
db.is_versioned("docs")                     # -> bool
```

Semantics:

- History is **append-only**: rollback records the restored state as new
  versions — nothing is erased, so the audit trail stays complete.
- `verify_chain()` detects any direct modification of the history store.
- Pruning records a chain anchor; the retained tail stays verifiable.
  Rewind depth is bounded by retention: you cannot roll back past a pruned
  boundary.
- Rollback cost depends on the workload: append-heavy tables pay only
  Chroma deletions (cheap); update-heavy tables re-embed restored
  LONGTEXT rows (O(changed rows)). Since 0.7.0 both paths are batched —
  removal-heavy rollbacks are ~20× faster, update-heavy restores ~37×.
- Schema changes (`add_column`/`drop_column`/`rename_column`) are rejected
  on versioned tables.
- History tables are engine-managed: excluded from `list_tables()`, DuckDB
  mirroring, graph sync, and FTS/Chroma indexing.
- Write overhead is ~13% (measured at 100k rows). `fork` is planned but
  deferred — checkpoint/rollback covers the rewind workflow.

`upsert()` also works on non-versioned tables (plain insert-or-update). It
requires the primary key column in `data`.

### Vector identity migration (0.8.0)

Chroma vectors are keyed by the logical primary key (`str(pk)`), not the
physical rowid — one identity shared across SQLite, the DuckDB mirror, and
Chroma. New collections are pk-keyed from creation; **pre-0.8 collections
keep rowid keys until migrated — upgrading changes nothing until you opt
in**:

```python
db.migrate_vector_identity(table=None)   # None = all longtext collections
```

Re-keys legacy collections by **copying embeddings (never recomputed —
asserted bit-identical)** and deleting orphan rowid-keyed vectors; idempotent;
default rowid-alias tables are no-ops.

### Metadata pre-filtering (multi-tenant scoping)

`where=` filters at the Chroma ANN level **before** the vector scan when the
keys are scalar columns mirrored into Chroma metadata (`TEXT`/`INTEGER`/
`REAL`/`BOOLEAN` — not `LONGTEXT`/`JSON`). Equality (`{"user_id": "u2"}`) and
Chroma operators (`{"score": {"$gte": 50}}`) are supported; operator-form
filters are enforced by the vector index only. The Python post-filter still
runs on top, so results are correct in every mode (keyword-mode operator
filtering fixed in 0.7.0).

## Long-Document Chunking

One embedding per LONGTEXT cell is right for messages and memory entries.
For multi-page knowledge documents, index **chunks as rows** using the
dependency-free splitter:

```python
from hybriddb.chunking import chunk_text

db.create_table("docs", {"id": TEXT, "title": TEXT, "full_text": LONGTEXT})
db.create_table(
    "doc_chunks",
    {"doc_id": TEXT, "chunk_seq": "INTEGER", "content": LONGTEXT},
)

full_text = "...a long document..."
db.insert("docs", {"id": doc_id, "title": "Design spec", "full_text": full_text})
for i, chunk in enumerate(chunk_text(f"{title}. {full_text}")):
    db.insert("doc_chunks", {"doc_id": doc_id, "chunk_seq": i, "content": chunk})
```

```python
chunk_text(text, max_chars=1200, overlap=True)
```

Splitting rules: paragraph boundaries first (``\n\n``), then sentences —
never mid-sentence; adjacent pieces merge until ~1200 chars (~300 tokens for
MiniLM-class models); oversize sentences hard-split as a last resort;
``overlap=True`` (default) prepends the previous chunk's final sentence.

Retrieval searches the chunk table and joins back to the parent:

```python
hits = db.search("doc_chunks", "content", "consistency guarantees", mode="hybrid", limit=10)
best_by_doc = {}
for h in hits:
    best_by_doc.setdefault(h["doc_id"], h)   # best chunk per document
results = [(db.get("docs", d), h) for d, h in best_by_doc.items()]
```

Chunks are ordinary rows, so versioning, checkpoints, and rollback work on
the chunk table unchanged.

## OLAP API

DuckDB analytics are optional. Install with:

```bash
pip install "hybriddb[analytics]"
```

Use the `db.olap` facade:

```python
db.create_table("events", {"category": TEXT, "value": REAL})
db.insert_batch("events", [{"category": "A", "value": 1.5}], sync=False)

rows = db.olap.query("SELECT category, SUM(value) AS total FROM events GROUP BY category")
```

The facade auto-registers app tables with DuckDB before queries. Direct methods remain available:

- `register_duckdb_table(table)`
- `unregister_duckdb_table(table)`
- `sync_duckdb_table(table)`
- `analytics(sql)`

Mirrors are created lazily on first OLAP use; tables the user never queries
with `olap` are not mirrored (since 0.7.0).

## Public vs Private

Stable public API:

- Methods documented in this file.
- Constants exported from `hybriddb` (including `EmbeddingModelError` and
  `default_embedding_fn`).
- `db.graph` and `db.olap` facades.

Private/internal API:

- Any method or attribute starting with `_`, including `_connect`, `_db_path`, `_vector_path`, and `_process_journal`.
- Private internals may change between minor versions.

Use `cursor()` instead of `_connect()` and `process_journal()` instead of `_process_journal()`.