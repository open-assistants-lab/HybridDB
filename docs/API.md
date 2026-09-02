# HybridDB API Reference

This document describes the stable public API for HybridDB `0.4.x`.

HybridDB is one embedded database object that coordinates SQLite, FTS5, ChromaDB, a self-healing journal, optional DuckDB analytics, and optional graph helpers.

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
    HybridDB,
    SearchMode,
)
```

## Constructor

```python
db = HybridDB(
    path="./data",
    embedding_fn=None,
    embedding_model_name=None,
    max_chroma_index_gb=5,
    auto_rebuild_chroma=False,
)
```

Defaults:

- `embedding_fn=None` uses ChromaDB's bundled local MiniLM embedding.
- Hash embedding is used only as a fallback if ChromaDB's default embedding cannot load.
- `embedding_model_name=None` records `chroma:all-MiniLM-L6-v2` for the default embedding.
- `max_chroma_index_gb=5` protects local disk usage.

Use a custom embedding function when you need a specific model or provider:

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
```

Public methods validate table and column identifiers. Use simple Python identifiers such as `docs`, `messages`, `content`, or `created_at`.

## CRUD

```python
row_id = db.insert("docs", {"title": "Hello", "body": "Hybrid search memory"})
rows = db.insert_batch("docs", [{"title": "A", "body": "..."}, {"title": "B", "body": "..."}])

row = db.get("docs", row_id)
ok = db.update("docs", row_id, {"title": "Updated"})
deleted = db.delete("docs", row_id)
total = db.count("docs")
```

`insert_batch()` returns `list[int | str]`. It returns strings when your table uses `id TEXT PRIMARY KEY`.

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

For advanced migrations or custom writes, use `raw_query()` or the public cursor context manager:

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
```
db.search("docs", "body", "how do I begin?", mode="semantic")
db.search("docs", "body", "getting started", mode="hybrid")

db.search("docs", "body", "hello", mode=SearchMode.KEYWORD)
db.search("docs", "body", "hello", mode=KEYWORD)
db.search("docs", "body", "hello", mode=HYBRID)
```

Behavior:

- `TEXT` columns support keyword search.
- `LONGTEXT` columns support keyword, semantic, and hybrid search.
- Hybrid search fuses keyword and vector results using reciprocal-rank fusion.
- Empty queries return `[]`.

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
processed = db.process_journal()
```

Health and repair:

```python
health = db.health("docs")
result = db.reconcile("docs")
```

`reconcile()` repairs missing ChromaDB documents, removes ghosts, and refreshes graph-derived state.

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

Graph helpers are available directly and through `db.graph`.

```python
alice = db.graph.add_node(label="Alice", type="person")
bob = db.graph.add_node(label="Bob", type="person")
db.graph.add_edge(None, alice, bob, edge_type="knows", weight=0.9)

neighbors = db.graph.get_neighbors(alice)
path = db.graph.shortest_path(alice, bob)
scores = db.graph.pagerank()
# semantic graph retrieval: vector-search seeds, then expand via PageRank
ppr = db.graph.search_graph_ppr("memory", hop_expansion=2, limit=5)
# re-sync registered table rows into graph nodes
synced = db.graph.sync_graph_nodes()
```

The namespaced `db.graph` facade exists for discoverability. Direct methods such as `db.add_node()` and `db.shortest_path()` remain supported.

## Versioned Tables

Opt-in per table. Versioned tables keep an append-only, hash-chained history
(`{table}__history`) of every insert/update/delete, while the main table
stays the current state — FTS5 and Chroma keep indexing current data only.

```python
db.create_table("docs", {"id": TEXT, "body": LONGTEXT}, versioned=True, hash_chain=True)
db.author = "agent-1"                       # optional, recorded per event

db.upsert("docs", {"id": 1, "body": "v1"})  # insert-or-update
db.upsert("docs", {"id": 1, "body": "v2"})

db.log("docs")                              # change log, newest first
db.history("docs", key=1)                   # every version of a row
db.diff("docs", from_seq=1, to_seq=2)       # added/removed/changed
db.as_of("docs", seq=1)                     # point-in-time read

cp = db.checkpoint("docs", "before-edit")
db.rollback("docs", checkpoint="before-edit")   # state re-applied as new versions
db.verify_chain("docs")                     # -> {"valid": True, "checked": N, ...}
db.archive("docs", "exports/docs", format="parquet")  # or "jsonl"
db.prune("docs", before_seq=5)              # retention; keeps the chain verifiable
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
  LONGTEXT rows (O(changed rows)).
- Schema changes (`add_column`/`drop_column`/`rename_column`) are rejected
  on versioned tables.
- History tables are engine-managed: excluded from `list_tables()`, DuckDB
  mirroring, graph sync, and FTS/Chroma indexing.
- Write overhead is ~13% (measured at 100k rows). `fork` is planned but
  deferred — checkpoint/rollback covers the rewind workflow.

`upsert()` also works on non-versioned tables (plain insert-or-update).

### Metadata pre-filtering (multi-tenant scoping)

`where=` filters at the Chroma ANN level **before** the vector scan when the
keys are scalar columns mirrored into Chroma metadata (`TEXT`/`INTEGER`/
`REAL`/`BOOLEAN` — not `LONGTEXT`/`JSON`). Equality (`{"user_id": "u2"}`) and
Chroma operators (`{"score": {"$gte": 50}}`) are supported; operator-form
filters are enforced by the vector index only. The Python post-filter still
runs on top, so results are correct in every mode.

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

Splitting rules: paragraph boundaries first (``\n\n``), then sentences —
never mid-sentence; adjacent pieces merge until ~1200 chars (~300 tokens for
MiniLM-class models); oversize sentences hard-split as a last resort;
``overlap=True`` prepends the previous chunk's final sentence.

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

Mirrors are created lazily on first OLAP use; tables the user never queries with `olap` are not mirrored.
- `sync_duckdb_table(table)`
- `analytics(sql)`

## Public vs Private

Stable public API:

- Methods documented in this file.
- Constants exported from `hybriddb`.
- `db.graph` and `db.olap` facades.

Private/internal API:

- Any method or attribute starting with `_`, including `_connect`, `_db_path`, `_vector_path`, and `_process_journal`.
- Private internals may change between minor versions.

Use `cursor()` instead of `_connect()` and `process_journal()` instead of `_process_journal()`.
