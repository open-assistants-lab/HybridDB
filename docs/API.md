# HybridDB API Reference

This document describes the stable public API for HybridDB `0.3.x`.

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
db.search("docs", "getting started")
db.search_columns("docs", "getting started")
```

Modes:

```python
db.search("docs", "body", "hello", mode="keyword")
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

## Graph API

Graph helpers are available directly and through `db.graph`.

```python
alice = db.graph.add_node("Alice", type="person")
bob = db.graph.add_node("Bob", type="person")
db.graph.add_edge(None, alice, bob, edge_type="knows", weight=0.9)

neighbors = db.graph.get_neighbors(alice)
path = db.graph.shortest_path(alice, bob)
scores = db.graph.pagerank()
```

The namespaced `db.graph` facade exists for discoverability. Direct methods such as `db.add_node()` and `db.shortest_path()` remain supported.

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

## Public vs Private

Stable public API:

- Methods documented in this file.
- Constants exported from `hybriddb`.
- `db.graph` and `db.olap` facades.

Private/internal API:

- Any method or attribute starting with `_`, including `_connect`, `_db_path`, `_vector_path`, and `_process_journal`.
- Private internals may change between minor versions.

Use `cursor()` instead of `_connect()` and `process_journal()` instead of `_process_journal()`.
