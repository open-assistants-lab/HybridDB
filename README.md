# HybridDB

[![PyPI version](https://img.shields.io/pypi/v/hybriddb)](https://pypi.org/project/hybriddb/)
[![Downloads](https://img.shields.io/pypi/dm/hybriddb)](https://pypi.org/project/hybriddb/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
> **Purposefully built for AI Agents.** HybridDB gives agents persistent, searchable memory — every conversation turn is indexed and retrievable via keyword, vector, or hybrid search. Used in production by the [Executive Assistant](https://github.com/open-assistants-lab) agent system.

> **Embedded. Local. Open source.** No cloud APIs, no vector DB services, no internet connection required. Runs entirely on-device with SQLite + ChromaDB + your choice of local embedding model. Ships as a single Python package with zero external infrastructure dependencies.

**SQLite + FTS5 + ChromaDB with a self-healing journal.** One Python class that gives you keyword search, vector search, SQL queries, and structured filtering — all kept in sync automatically.

```python
from hybriddb import HybridDB, LONGTEXT, TEXT

db = HybridDB("./my_data")
db.create_table("docs", {"title": TEXT, "body": LONGTEXT})

db.insert("docs", {"title": "Getting Started", "body": "A guide to using HybridDB..."})
db.insert("docs", {"title": "API Reference", "body": "Full API documentation..."})

# Search every text column
search_all = db.search_all("docs", "getting started")

# Search one column
db.search("docs", "body", "how do I begin", mode="hybrid")

# Structured query with parameters
db.query("docs", where="title LIKE ?", params=("%start%",))
```


## Long Documents? Chunk Them (v0.6.0)

Embedding one vector per row is right for messages and memory entries. For
multi-page knowledge documents, index **chunks as rows** so retrieval works
at paragraph granularity:

```python
from hybriddb.chunking import chunk_text

db.create_table("doc_chunks", {"doc_id": TEXT, "chunk_seq": "INTEGER", "content": LONGTEXT})
for i, chunk in enumerate(chunk_text(full_text)):   # ~300-token chunks, never mid-sentence
    db.insert("doc_chunks", {"doc_id": doc_id, "chunk_seq": i, "content": chunk})

# search chunks, then join back to the parent document
hits = db.search("doc_chunks", "content", "quarterly roadmap", mode="hybrid")
```

## Agent Memory You Can Audit (v0.6.0)

Opt any table into **versioned history with a tamper-evident hash chain** —
built for agent memory, session logs, and audit trails:

```python
from hybriddb import HybridDB, LONGTEXT, TEXT

db = HybridDB("./agent_memory")
db.create_table("memories", {"id": TEXT, "content": LONGTEXT}, versioned=True)
db.author = "assistant"                       # recorded on every event

db.upsert("memories", {"id": "m1", "content": "User prefers morning meetings"})
db.upsert("memories", {"id": "m1", "content": "User prefers afternoon standups"})

db.history("memories", key="m1")              # every version, with hashes
db.diff("docs", from_seq=1, to_seq=2)         # what changed between two points
db.as_of("memories", seq=3)                   # what the agent knew at seq 3

cp = db.checkpoint("memories", "before-cleanup")   # named restore point
db.rollback("memories", checkpoint="before-cleanup")  # rewind — nothing erased
db.verify_chain("memories")                   # -> {"valid": True, ...} tamper-evident
```

History is append-only: rollback re-applies state as *new* versions, so the
audit trail stays complete. `verify_chain()` detects any direct tampering
with the history store. See [docs/API.md](docs/API.md#versioned-tables).

## Why HybridDB?

Every serious project that needs **both** keyword and semantic search ends up wiring SQLite + FTS5 + ChromaDB together. You handle schema creation, FTS5 triggers, ChromaDB collection management, keeping them in sync, recovering from crashes, rebuilding indexes...

HybridDB does all of that once, done right.

| Feature | Status |
|---------|--------|
| SQL CRUD (insert, update, delete, get, query) | ✅ |
| FTS5 keyword search with BM25 scoring | ✅ |
| ChromaDB semantic/vector search with HNSW | ✅ |
| Hybrid search (RRF fusion of keyword + semantic) | ✅ |
| DuckDB columnar analytics (optional) | ✅ |
| Versioned tables (tamper-evident history, time travel) | ✅ |
| NetworkX graph algorithms (optional) | ✅ |
| Recency-weighted scoring | ✅ |
| Schema management (create, add/drop/rename columns) | ✅ |
| Self-healing journal (crash recovery) | ✅ |
| Import/export, backup/restore | ✅ |
| Sync + async APIs | ✅ |
| No external API dependencies (works offline) | ✅ |
| Embedding model pluggable (sentence-transformers, OpenAI, custom) | ✅ |

## Documentation

- [API reference](docs/API.md) — stable public methods, sync/async examples, graph and OLAP facades
- [Benchmarks](docs/BENCHMARKS.md) — smoke vs full benchmark commands and expected runtime behavior
- [Release guide](docs/RELEASE.md) — local build, wheel smoke test, TestPyPI/PyPI publishing

## Installation

```bash
pip install hybriddb
```

HybridDB uses ChromaDB's bundled local MiniLM embedding by default. No API key required.

## Core Concepts

### Column Types

HybridDB maps Python-friendly types to SQLite storage and automatically sets up the right search indexes:

| Type | SQLite | FTS5 | ChromaDB | Use for |
|------|--------|------|----------|---------|
| `TEXT` | TEXT | ✅ | — | Names, titles, short strings |
| `LONGTEXT` | TEXT | ✅ | ✅ | Documents, messages, memory content |
| `INTEGER` | INTEGER | — | — | Counts, ages, IDs |
| `REAL` | REAL | — | — | Prices, scores, confidence values |
| `BOOLEAN` | INTEGER | — | — | Flags, status indicators |
| `JSON` | TEXT | — | — | Tags, metadata, structured data |

**TEXT** columns get automated FTS5 keyword search.
**LONGTEXT** columns get FTS5 + ChromaDB semantic search.

### Search Modes

```python
from hybriddb import HYBRID, LONGTEXT, TEXT, Column, SearchMode

db.create_table("docs", {"title": Column(TEXT), "body": LONGTEXT})

# Keyword only — fast, exact, great for names and titles
db.search("contacts", "name", "Alice", mode="keyword")

# Semantic only — finds "9am standup" when searching for "morning meetings"
db.search("memories", "content", "team rituals", mode=SearchMode.SEMANTIC)

# Hybrid — best of both, RRF fusion, the default
db.search("docs", "body", "getting started guide", mode=SearchMode.HYBRID)
db.search("docs", "body", "getting started guide", mode=HYBRID)

# Search across ALL text columns at once
db.search_all("contacts", "engineering manager")
db.search_columns("contacts", "engineering manager")
```

### Async API

All core operations have async wrappers that run blocking SQLite/ChromaDB work in a worker thread:

```python
await db.acreate_table("messages", {"content": LONGTEXT})
await db.ainsert("messages", {"content": "async-safe memory"})
results = await db.asearch("messages", "content", "memory")
```

### Public Cursor

For small custom SQL reads or migrations, use the public cursor context manager:

```python
with db.cursor() as cur:
    cur.execute("SELECT COUNT(*) FROM messages")
    count = cur.fetchone()[0]
```

### Namespaced Advanced APIs

Graph and OLAP helpers remain available on `HybridDB`, with namespaced facades for discovery:

```python
node_id = db.graph.add_node(label="Alice", type="person")
rows = db.olap.query("SELECT COUNT(*) AS total FROM messages")
```

### Recency Scoring

Boost recent content over older content:

```python
results = db.search(
    "messages", "content", "project update",
    recency_weight=0.3,        # 30% weight to recency
    recency_column="timestamp"
)
```

### Self-Healing Journal

All ChromaDB mutations (adds, updates, deletes) are journaled in SQLite. On insert with `sync=True` (default), the journal is processed immediately. On `sync=False`, journal entries are deferred:

```python
# Batch insert — defer ChromaDB sync for speed
db.insert_batch("contacts", big_list_of_rows, sync=False)
db.process_journal()  # Sync everything at once
```

If your process crashes mid-write, the journal replays pending entries on next startup. No ghosts, no drift.

### Health & Maintenance

```python
# Check if SQLite and ChromaDB are in sync
health = db.health("contacts")
# {"sqlite_rows": 5000, "chroma_docs": {"contacts_bio": 5000}, "status": "ok"}

# Reconcile: delete ghosts, add missing docs
result = db.reconcile("contacts")
# {"ghosts_deleted": 0, "missing_added": 3, "metadata_updated": 0}
```

## Custom Embedding Models

By default, HybridDB uses ChromaDB's bundled local MiniLM embedding. Plug in any embedding function if you want a specific model or provider:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
db = HybridDB("./data", embedding_fn=lambda text: model.encode(text).tolist())
```

Works with any embedding provider — OpenAI, Cohere, Hugging Face, local models.

## Built On

| Technology | Role |
|-----------|------|
| [SQLite](https://sqlite.org) | Primary store, WAL mode, row-level CRUD |
| [FTS5](https://sqlite.org/fts5.html) | Keyword search with BM25 scoring |
| [ChromaDB](https://trychroma.com) | Vector/semantic search with HNSW index |
| [DuckDB](https://duckdb.org) | Columnar analytics (optional) |
| [NetworkX](https://networkx.org) | Graph algorithms — PageRank, shortest path, community detection (optional) |
| [sentence-transformers](https://sbert.net) | Custom embedding model support (optional) |

## License

MIT — see [LICENSE](LICENSE).

## Author

Eddy Xu

Inspired by [claude-mem](https://github.com/thedotmack/claude-mem) by [Matt Mack](https://github.com/thedotmack) and [Claude Code](https://github.com/anthropics/claude-code) by [Anthropic](https://anthropic.com).

## Status

Alpha — actively developed, API may evolve. Core CRUD and search are stable with full test coverage (35+ tests). Currently used in production in the Executive Assistant agent system.
