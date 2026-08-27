# HybridDB v0.6.0 — Versioned Tables (Git-like Primitives)

**Status:** Approved — planned
**Date:** 2026-08-26
**Target version:** 0.6.0 (additive; minor bump, no breaking changes)

---

## 0. Implementation amendments (2026-08-26, post-review)

Decisions locked during implementation review — these refine the sections below:

1. **Post-image recording.** History rows store the row state *after* the
   operation (insert/update post-images; delete = tombstone carrying the last
   known row). This makes `as_of(seq)` a single indexed query (latest
   post-image per pk with `_seq <= seq`, tombstones excluded) instead of a
   replay, and `diff` a last-wins comparison over the seq range.
2. **Fork deferred.** Checkpoint/rollback is the proven agent-memory
   primitive (LangGraph-style rewind); fork is a materialized copy with full
   Chroma re-embedding and adds value only with parallel versions, which the
   single-writer non-goals already discourage. Fork moves to a future
   release unless a named consumer needs it.
3. **Chain anchors make prune safe.** Prune deletes a contiguous *prefix* of
   history and records an anchor (`anchor_seq`, `anchor_hash` of the last
   pruned row). `verify_chain` accepts a `prev_hash` discontinuity exactly at
   an anchor boundary, so the retained tail stays verifiable. Prune with
   `keep_checkpoints=True` refuses to cut past a checkpoint.
4. **Schema changes forbidden on versioned tables** (`add/drop/rename_column`
   raise) — stored `row_json` must keep matching the table schema for
   `as_of`/`history` correctness.
5. **Exclusion matrix for history tables:** not in `_schema` (hence not in
   `list_tables`/`stats`/DuckDB auto-registry/graph-sync), created via raw
   SQL only, `create_table` rejects names ending in `__history`.
6. **`author`** is an instance attribute (`db.author = "agent-1"`), recorded
   per history event.
7. **Measured write overhead: ~13%** (20.6k → 18.0k rows/s on 100k rows with
   hash chain + history capture) — better than the "double write volume"
   estimate, because the journal already dominates per-row cost.
8. **PK changes on versioned tables** record a tombstone for the old pk plus
   the post-image under the new pk, keeping `as_of`/`history` correct.

## 1. Context & reason

### Where this comes from

HybridDB's primary consumer (the Assistant agent platform) studied two
production systems for git-like data versioning — **DeepSeek Harness (dsh)**
and **Dolt ("Git for Data")** — while designing its agent session log and
knowledge layer. The requirements that fell out of that study (assistant
roadmap `R-SL1` event-sourced session log, `A3` tamper-evident audit trail,
and the kit-factory "knowledge as files" workflow) all reduce to one need:

> **A storage engine whose data is versionable like code** — branch, fork,
> diff, rollback, verify, archive — without a server, without leaving Python,
> and without losing HybridDB's hybrid search (FTS5 + Chroma).

Dolt proves the demand (24k★, positioned explicitly as "the best database for
agent memory") but is a 103MB Go binary with a MySQL surface. HybridDB can
serve the same need as an **embeddable Python library** — lighter, per-user,
and with hybrid search built in.

### Why the engine (and not the application) must own this

- **Tamper-evidence must be engine-enforced.** If the application computes
  audit hashes, it can forge or skip them. A storage engine that maintains the
  hash chain on every write makes the guarantee *structural* — that is the
  difference between "we keep logs" and "the logs are cryptographically
  provable," which is what audit/compliance consumers (and any AI agent
  writing its own memory) actually need.
- **Versioning primitives are generic.** Point-in-time reads, row lineage,
  diffs, forks, and archival archives are useful to *any* HybridDB consumer
  (agent memory, audit logs, knowledge bases, CRMs) — not just one app.
- **Precedent:** Dolt (state-versioned MySQL), lakeFS (git semantics on S3),
  SQL:2011 system-versioned tables — versioning at the storage layer is an
  established pattern; HybridDB currently has none of it.

### What changed in the consumer to trigger this

The Assistant roadmap moved agent conversation history to an **event-sourced
session log** (append-only, no checkpoints) and its knowledge layer to
**content + review workflows**. Both need: append-only capture, tamper
evidence, point-in-time reads, diffs, forks for safe experimentation, and
archive/prune for retention. All of those are storage primitives, not agent
logic.

## 2. Design

### 2.1 Versioned tables (SQL:2011-style shadow history)

`create_table(..., versioned=True)` creates the table **plus a shadow history
table** (`{table}__history`):

- The **main table stays the current state** — normal SQL, joins, indexes,
  and (critically) FTS5/Chroma index *current data only*, so hybrid search
  keeps working unchanged.
- Every `insert`/`update`/`delete` **appends the prior version** to the
  history table with: `_seq` (monotonic per table), `_op`
  (insert/update/delete), `_ts`, `_author` (optional), and the hash-chain
  columns.
- **Hash chain:** `event_hash = SHA256(prev_hash ‖ op ‖ pk ‖ row_json)` —
  computed by the engine on write; `verify_chain()` recomputes and validates.
  Tamper-evident by construction: any modification/deletion of a history row
  breaks the chain.
- **Rollback** = write the historical version back as a *new* current version
  (the chain never rewinds — history stays append-only and complete).

> Why shadow-history instead of append+latest-wins (ReplacingMergeTree
> style): with a shadow table, FTS5/Chroma index **current data only** —
> search never hits stale versions, and current-state queries need no
> deduplication. Latest-wins-on-read would force a filter into every search
> and break the "search just works" property.

### 2.2 API surface (all additive; nothing existing changes)

```python
db = HybridDB(path)

# Create versioned (+ hash-chained) tables
db.create_table("contacts", {...}, versioned=True, hash_chain=True)

# Writes (unchanged signatures; engine captures history automatically)
db.upsert("contacts", {"id": 1, "name": "Alice"})   # NEW: insert-or-update → history
db.delete("contacts", 1)                            # → history records the delete

# Version control surface
db.log("contacts")                                  # change log (dolt_log equivalent)
db.history("contacts", key=1)                       # every version of a row (dolt_history_)
db.diff("contacts", from_seq=10, to_seq=50)         # changes between two points
db.as_of("contacts", seq=10)                        # point-in-time read
db.checkpoint("contacts", "before-migration")       # named restore point (tag)
db.rollback("contacts", checkpoint="before-migration")  # or at_seq=
db.fork("contacts_v2", from_table="contacts", at_seq=100)   # v1: materialized copy
db.verify_chain("contacts")                         # → bool + first broken seq
db.archive("contacts", "exports/contacts", format="parquet")  # current + history
db.prune("contacts", older_than=..., keep_checkpoints=True)   # GC after archive
```

### 2.3 Semantics

| Operation | Behavior |
|---|---|
| `upsert` | Insert if key absent, else update; **history records the prior state** (op: insert/update) |
| `delete` | Row removed from current; prior state appended to history (op: delete) |
| `history(key)` | Every version of a row, oldest → newest, with hashes |
| `diff(from_seq, to_seq)` | Rows added/modified/deleted between two log positions |
| `as_of(seq)` | Reconstruct table state at a log position (current + history replay) |
| `checkpoint(label)` | Tag the current seq — named restore point |
| `rollback(checkpoint=|seq=)` | Re-apply historical state as *new* versions (chain preserved) |
| `fork(name, at_seq)` | New table = materialized copy of state at seq, with history linkage (v1) |
| `verify_chain()` | Recompute the hash chain; report first broken link |
| `archive(path, format)` | Export current + history to parquet/jsonl (DuckDB attach) |
| `prune(older_than)` | Delete history rows **only after a successful archive** (safe GC) |

## 3. Out of scope (explicit non-goals for 0.6.0)

- **Remote push/pull** (DoltHub/GitHub sync) — distributed sync is a product of its own
- **3-way key-based merge** — single-writer per database; fork→evaluate→adopt-or-discard covers the workflow
- **SQL `AS OF` syntax / `dolt_`-style system tables** — Python API first; SQL surface later if demanded
- **Copy-on-write structural sharing for forks** — v1 forks are materialized copies (fine at per-user scale)
- **CLI** — the API is the surface; a CLI is a follow-on if demanded

## 4. Tasks (TDD, ordered)

| # | Task | Files | Est | Status |
|---|---|---|---|---|
| V1 | `versioned=True` + `hash_chain=True` on `create_table`; shadow history table; hash-chain maintenance on insert/update/delete | `hybriddb/schema.py`, `hybriddb/crud.py`, `hybriddb/db.py` | 2–3d | ✅ done |
| V2 | `upsert()` (insert-or-update with history capture) | `hybriddb/crud.py` | 1d | ✅ done |
| V3 | `log()` / `history(key)` / `verify_chain()` | `hybriddb/` (new `versioning.py` mixin) | 1–2d | ✅ done |
| V4 | `diff(from_seq, to_seq)` + `as_of(seq)` | `versioning.py` | 1–2d | ✅ done |
| V5 | `checkpoint(label)` / `rollback(...)` | `versioning.py` | 1–2d | ✅ done |
| V6 | ~~`fork(name, at_seq)`~~ — **deferred** (see amendment 2) | — | — | moved out |
| V7 | `archive(path, format)` + `prune(older_than)` (chain anchors) | `versioning.py` | 1–2d | ✅ done |
| V8 | Docs, changelog, release 0.6.0 | `docs/`, `CHANGELOG.md`, `pyproject.toml` | 0.5d | ✅ done |

Tests: `tests/test_versioning.py` — 34 tests, all green; plus the spec's
end-to-end flow (create → upsert ×3 → checkpoint → update → diff → as_of →
rollback → verify_chain → archive). Gates: `pytest`, `ruff` (mypy not
configured in this repo).

## 4b. Consumer (Assistant) integration — sequencing note

The Assistant roadmap consumes this as follows (do **not** block on it):

- Assistant Phase 0 ships its session log **app-level first** (same
  `prev_hash`/`event_hash` columns) — schema is column-compatible with V1, so
  the migration is "delegate computation to the engine" once `assistant`
  bumps `hybriddb>=0.6.0`.
- Enforcement upgrade path: app-computed hashes → engine-enforced hashes.
  The claim strengthens ("the storage engine maintains a tamper-evident
  chain"); no data migration.

## 5. Risks

- **Write amplification:** versioned tables double write volume (row + history row).
  Per-user stores are small; acceptable. Document it.
- **FTS5/Chroma must never index history tables** — name history tables with the
  existing `_fts_`-conflict guard convention and exclude from search paths.
- **Chain verification cost** is O(history) — fine at per-user scale; note in docs.
- **Scope creep:** fork/merge/remote features are explicitly deferred; resist adding
  them without a named consumer.

## 6. Success criteria

- [ ] `versioned=True` tables: every state change captured in `__history` with an intact hash chain
- [ ] `verify_chain` detects a tampered row (test mutates the store directly)
- [ ] `diff` / `as_of` / `history` / `log` / `checkpoint` / `rollback` / `fork` / `archive` / `prune` all green
- [ ] Existing suite green (no behavior change for non-versioned tables)
- [ ] Released as 0.6.0; `assistant` consumes it for the session log + audit layer