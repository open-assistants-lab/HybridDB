# Chroma Vector Identity → Logical Primary Key (v0.7 design)

**Status:** Proposed
**Date:** 2026-08-28
**Target version:** 0.7.0 (internal representation change; no public API break)

---

## 1. Problem

Chroma vector ids are currently `str(internal rowid)`. Rowid is *physical*
storage, not logical identity, and the two identity models (SQLite/DuckDB use
the primary key, Chroma uses rowid) have caused the majority of this
project's sync bugs:

| Incident | Root cause |
|---|---|
| v0.5.4 — `search_graph` returned `[]` | `SELECT rowid, id` aliasing dropped `rowid` from the row dict |
| v0.5.5 — rowid→pk mapping fixes across graph seed search | rowid ≠ pk on TEXT/custom PK tables |
| 0.5.6-era journal wedge — `DuplicateIDError` on TEXT-PK delete+reinsert | TEXT-PK tables **reuse rowids** (no AUTOINCREMENT); the journal carried two `add` events with the same rowid key |
| Rollback correctness work (0.6.0/0.6.1) | removal path had to resolve `internal_rowid` before deleting, because the Chroma key was the rowid |

The fix proposed here deletes the two-identity-model class: **Chroma vectors
are keyed by the logical primary key**, the same identity SQLite, the DuckDB
mirror, and the versioned-table history already use.

## 2. Decision

`chroma_id = str(pk_value)` — stable across deletes/re-inserts, identical
between SQLite/DuckDB/Chroma.

Per table type:

| Table shape | today (`str(rowid)`) | 0.7.0 (`str(pk)`) | Migration needed? |
|---|---|---|---|
| `id INTEGER PRIMARY KEY AUTOINCREMENT` (default) | rowid == pk | identical keys | **No** |
| custom `INTEGER PRIMARY KEY` (rowid alias) | rowid == pk | identical | **No** |
| `TEXT PRIMARY KEY` / any non-alias pk | rowid ≠ pk | changes | **Yes** |

So the default case — the majority of tables — is unaffected. Only
non-rowid-alias tables have legacy keys to migrate.

## 3. Journal plumbing (how the apply path learns the pk)

Journal chroma entries are currently keyed by `row_id` (the internal rowid),
stored in `_journal.row_id INTEGER`. For pk-keyed Chroma we need the pk at
apply time:

- **add / update events:** the row exists in SQLite at apply time → resolve
  pk with one batched `SELECT rowid, {pk} FROM {table} WHERE rowid IN (…)`
  (same pattern the DuckDB sync already uses). One query per batch, not per
  row.
- **delete events:** the row is already gone — but `crud.delete()` already
  stores the app-level pk in the journal's `data` column (`row_delete`
  entries). Chroma delete keys resolve from `data`.

**Rejected alternative:** adding a `pk_text` column to `_journal` and
writing pk at journal time. Rejected because it changes the journal schema
and every journal producer for a benefit the apply-time resolution already
delivers; and `row_delete.data` already carries the pk by prior design.

## 4. Compatibility: per-collection identity scheme

Existing collections hold rowid-keyed vectors; a 0.7.0 upgrade must not
silently produce duplicate/missing keys. Therefore the scheme is recorded
**per collection** in Chroma collection metadata:
`{"hybriddb:identity": "pk" | "rowid"}`.

- **New collections** (0.7.0+): created with `"pk"` from the start.
- **Existing collections:** no marker → treated as `"rowid"` → writes and
  reads continue exactly as today (no behavior change on upgrade).
- **Migration is explicit and opt-in** via
  `db.migrate_vector_identity(table=None)` (None = all tables):
  for each collection, re-key vectors to `str(pk)` and set the marker.

## 5. `migrate_vector_identity()` — no re-embedding

Chroma has no rename-id API, so migration is copy + delete, but **embeddings
are carried over, not recomputed**:

1. `GET` every vector: ids, embeddings, documents, metadatas
2. resolve `rowid → pk` for each id via SQLite (`SELECT rowid, {pk} FROM
   {table}`); rows whose rowid no longer exists = orphans → skip (they are
   ghosts; `reconcile()` deletes them)
3. `upsert` under `str(pk)` with the fetched embeddings/documents/metadata
4. `delete` the legacy rowid-style ids
5. set collection metadata `hybriddb:identity = "pk"` (via
   `collection.modify(metadata=…)`)

Idempotent: a second run finds pk-keyed ids, re-upserts them unchanged (upsert
= no-op content-wise), and the legacy-id delete is a no-op. Cost: O(vectors)
copy at DuckDB-copy speed (no embedding), ~seconds at 100k vectors. The
migration is safe while writers are quiesced (single RLock) and safe to
interrupt (upsert-then-delete order means the worst case re-runs cleanly).

**Default-table collections are skipped entirely** (keys identical) — the
common case migrates for free.

## 6. Read-path simplification

`_fetch_rows_by_ids` currently fetches by `_get_rowid_ref` and re-maps.
With pk-keyed vectors, the vector ids *are* pks: fetch by
`WHERE {pk_col} IN (…)` directly. `_get_rowid_ref` remains solely for FTS5
(`content_rowid` must be the rowid/pk alias — unchanged).

## 7. Success criteria

- [ ] New/created-in-0.7 collections are pk-keyed (marker set)
- [ ] `migrate_vector_identity()` re-keys a TEXT-PK collection with no
      re-embedding (assert embeddings identical before/after) and is
      idempotent
- [ ] The TEXT-PK delete+reinsert sequence that wedged the journal at 0.5.6
      cannot wedge under pk keys (regression test)
- [ ] `verify_chain`, `health()`, BEIR-accuracy unchanged
- [ ] Non-versioned and versioned tables both pass the full suite

## 8. Non-goals

- Re-embedding on migration (explicitly avoided — embeddings are copied)
- Changing FTS5 identity (FTS5 `content_rowid` stays rowid/pk — unchanged)
- Bumping Chroma versions or switching engines (sqlite-vec evaluated and
  declined on performance)
- **Watch: Quack Remote Protocol** (duckdb.org/quack) — DuckDB's new
  client-server layer (beta). Evaluated 2026-08-28: declined for now —
  HybridDB is embedded/local by positioning, no consumer needs
  multi-process analytics, and a beta wire protocol in an embedded library
  is asymmetric risk. Trigger to revisit: a named consumer needs
  shared/multi-process stores — design a "team memory server" deployment
  mode around Quack + DuckLake as a separate layer rather than changing
  this library. Meanwhile, a second process can read the SQLite file
  read-only (WAL allows concurrent readers).