# HybridDB Strategy

## The vision

> **An LLM agent uses HybridDB to build data apps for non-technical users — CRM, inventory systems, project trackers, anything with tables and relationships — without a human writing a single line of SQL.**

This is the product. Everything else is a stepping stone.

---

## What this changes

The real question HybridDB answers is not "why not just import chromadb and sqlite3?" — that's a defensive framing that competes on packaging.

The real question is:

> **"What database is designed to be used by an LLM agent, not a human developer?"**

No existing database answers this. Every database since the 1970s assumes a human DBA writes schemas, migrations, and queries. HybridDB can be the first that assumes an LLM agent drives it.

That changes the competitor set, the technical priorities, the message, and the distribution.

---

## The competitor set

This positioning has almost no direct competitors:

| "Competitor" | Why it's not the same |
|---|---|
| **Supabase / Firebase** | Cloud hosting, Postgres, human to set up. Not agent-driven. |
| **Airtable / NocoDB / Baserow** | Web UI for humans. Spreadsheet metaphor. Not agent-driven. |
| **Appsmith / Tooljet / Budibase** | Low-code UI builders for humans. Not agent-driven. |
| **SQLite** | A library, not a platform. The agent would need to write raw SQL. |
| **LanceDB / DuckDB** | Vector/analytics focused. No CRUD app-building primitives. |
| **Mem0 / Zep / Letta** | Agent memory for chat history, not agent application data. |

The real competitor is: **the human developer who builds the app manually.** HybridDB's value is eliminating that human.

---

## CoreMem lives here too

CoreMem is built on HybridDB. One data engine serves two distinct workloads:

| | Agent memory (CoreMem) | User app data (CRM/inventory) | Shared |
|---|---|---|---|
| Write pattern | Append-heavy, sequential | CRUD, updates, deletes | SQLite transactions |
| Search | Semantic + recency + temporal | Keyword + filtered queries | FTS5 + vector |
| Schema | Fixed (memory schema) | Dynamic (user creates tables) | Schema mixin |
| Relationships | Entity association graph | Business entity relations | Graph module |

This creates one shared engine with two explicit API personas, each optimized for its use case but sharing the same journal, graph, and index infrastructure underneath.

Having both in one engine is a moat: no competitor in agent memory (Mem0, Zep, Letta) can pivot to app-building without rebuilding, and no competitor in embedded databases (LanceDB, DuckDB) can add agent memory without starting over. HybridDB already has both.

The constraint: CoreMem's design (append-heavy, fixed schema, recency bias) must not constrain the app-building API (relational schema, dynamic tables, CRUD). Keep the personas separate at the API level even if they share the engine.

---

## What HybridDB needs to be

The existing v0.4.2 covers about 40% of the vision. Here's what's missing:

### 1. Schema inference from natural language (suggest_schema)

The core differentiator. The agent says "I need a CRM" and HybridDB returns a suggested schema.

The agent reviews it, adjusts it, calls create_from_schema(schema) — and the app exists.

Why this wins: No other database lets an agent bootstrap schemas. The agent either writes raw SQL (brittle) or uses an ORM (too many layers). HybridDB fills the gap: the agent describes the app, HybridDB handles the data modeling.

### 2. Relationship-first API (relate, entity CRUD)

CRMs and inventory systems are defined by their relationships. Currently HybridDB has graph nodes + edges for this, but the entry point needs to be higher-level. A single relate() call should create the FOREIGN KEY, register the graph edge sync, and set up cascade behaviors.

At query time, the agent traverses naturally through neighbors() and compose() — leveraging the existing graph engine with a simpler surface.

Entity-level operations sit on top of CRUD + graph:

Why this wins: Every business app is a graph of entities. Supabase has foreign keys but no graph traversal. NocoDB has table links but no graph query. HybridDB's graph engine already exists — surface it as a relationship-first API.

### 3. Views and aggregations for reporting

CRMs need "deals by stage." Inventory needs "low stock alerts." The agent needs a simple abstraction:

Why this wins: The agent builds reporting without SQL. DuckDB backs it under the hood, but the agent doesn't need to know.

### 4. MCP server as the distribution layer

This is how any agent discovers HybridDB — not just Python agents, but Claude Desktop, Cursor, EA, or any MCP-compatible runtime. The MCP server exposes app-building tools:

| Tool | What it does |
|---|---|
| suggest_schema(description) | Returns schema proposal from natural language |
| create_table(name, columns) | Creates a table + FTS5 + vector indexes |
| insert(table, data) | Adds a row |
| query(table, where) | Queries with optional conditions |
| search(table, query) | Keyword + semantic search |
| relate(table_a, column, table_b) | Defines a relationship |
| create_view(name, config) | Creates an aggregate/report |
| neighbors(entity_id, type) | Graph traversal |
| export(table) | JSON/CSV export |
| app_status() | Backup, integrity, stats |

Why this wins: Any MCP-compatible agent discovers suggest_schema and starts building apps. No SDK integration, no language lock-in, no library import.

The MCP API is also the scope guardrail — whatever tools are exposed are what HybridDB does. If a capability isn't an MCP tool, HybridDB doesn't try to do it.

### 5. ChromaDB: keep it for now

ChromaDB runs in-process via PersistentClient. The journal handles consistency. For personal-scale workloads (under 100K vectors) it works fine. The concerns (50MB footprint, ONNX download on first use, version coupling) are real but not blocking:
- Pin version in pyproject.toml — manageable
- ONNX download is one-time per machine — minor friction
- 50MB is fine for pip install

Revisit ChromaDB only if: (a) version breaks become a support tax, or (b) you need same-transaction vector writes and ChromaDB can't do it. Don't block the strategy on replacing it.

---

## What this means for the rest of the stack

The architecture is:

  EA (Executive Assistant) — consumption layer
   |  "build me a CRM"
   |
   +--- MCP ---+
   |            |
  CoreMem    ConnectKit
  (memory)   (tools)
   |            |
   +------+-----+
          |
      HybridDB
   (SQLite + FTS5 + Chroma + Graph)

- CoreMem lives on HybridDB with a memory-optimized API persona
- ConnectKit feeds connector data into HybridDB tables
- EA is the surface where a user says "build me a CRM" and the agent makes it happen

The critical dependency: **EA must ship.** Without EA, HybridDB's app-building API has no consumption layer. The agent has nowhere to surface the CRM it built.

---

## The narrative

When someone asks "what is HybridDB?":

> "It's a database an LLM agent can use to build apps. You describe what you need — 'a CRM for tracking customers and deals' — and the agent calls HybridDB to create the schema, set up the relationships, and handle the queries. No SQL, no Postgres, no cloud."

When someone asks "why not just use SQLite?":

> "SQLite is a library for humans writing SQL. HybridDB is a platform for agents building apps. The agent can say 'create a contact with an email and a company relationship' and HybridDB infers the schema, sets up the foreign key, creates the vector index, and registers the graph edge — in one method call an agent can discover automatically via MCP."

When someone asks "why not Supabase?":

> "Supabase is a cloud Postgres with a pretty UI. HybridDB is one pip install, one constructor, one file on disk. The agent builds the app, you own the data, nothing leaves your machine."

---

## Strategic roadmap

| Priority | What | Why | Effort |
|----------|------|-----|--------|
| P0 | MCP server with CRUD, search, relate | Distribution. Any agent discovers HybridDB. | 1 week |
| P0 | suggest_schema + create_from_schema | The killer feature. Agent describes -> DB creates. | 1-2 weeks |
| P1 | relate() — first-class relationship API | CRMs and inventory live on relationships. | 1 week |
| P1 | create_view() — agent-friendly reporting | The agent builds reports without SQL. | 1 week |
| P1 | Entity-level API (db.entity().create()) | Sugar over CRUD + graph for the agent. | 1-2 weeks |
| P2 | CoreMem on HybridDB (two-persona API) | Unified data engine narrative. | Depends on CoreMem |
| P2 | Export/import in app-friendly formats | User owns their data. | 1 week |
| P2 | Schema migration helper | Apps evolve. The agent shouldn't re-enter data. | 1 week |
| P3 | ChromaDB replacement (sqlite-vec) | When version coupling becomes a tax, not before. | 2-3 weeks |

---

## The honest risk

This vision makes HybridDB more ambitious and more complex than a typical embedded database. The risks:

1. **Scope creep** — serving agent memory, agent tool connectors, and user-facing app data from one engine risks being average at all three. The MCP API is the guardrail: if it's not an MCP tool, HybridDB doesn't do it.
2. **EA dependency** — without EA shipped, the "agent builds a CRM" demo has no surface. The strategy timelines need to account for this.
3. **Single developer tempo** — v0.4.2 is real. The MCP server and schema inference are weeks, not months. But sustaining both the engine and the app-building layer is a lot of surface for one person.

---

## Should we try this?

Yes, for 6 weeks. The test:

| Week | Ship | Success signal |
|------|------|----------------|
| 1 | MCP server with CRUD + search + relate | Any MCP agent can create tables, insert rows, and query |
| 2-3 | suggest_schema + create_from_schema | Agent takes "CRM with contacts and deals" into working database |
| 3-4 | EA integrates HybridDB MCP | User says "build a CRM" in EA -> EA calls HybridDB MCP -> CRM exists |
| 5-6 | Dogfood: build a real app (inventory tracker) with the stack | End-to-end flow works without human writing SQL or Python |

By week 6 you'll know if the "agent builds apps" demo actually works and resonates. If yes, double down. If no, you still have a solid embedded DB with an MCP server — useful on its own, just not the category-defining bet.
