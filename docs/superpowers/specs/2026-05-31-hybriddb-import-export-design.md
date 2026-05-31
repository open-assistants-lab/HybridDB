# HybridDB Import/Export & SQL Utilities

2026-05-31

## Problem

HybridDB combines SQLite, FTS5, ChromaDB, and DuckDB in a single local-first
directory. Today, there is no way to:

- **Export** your database for sharing, version control, or migration between
  machines. Users must write custom Python loops to serialize/deserialize.
- **Back up** the full directory (SQLite + vectors + analytics) atomically.
  A `cp -r` works but is racy if writes are happening.
- **Restore** from a backup when something goes wrong — disk failure, corrupt
  ChromaDB index, accidental `DELETE FROM messages`.
- **Diagnose** storage health across all four layers (SQLite integrity,
  ChromaDB collections, DuckDB sync status).
- **Reclaim disk space** after bulk deletes. SQLite doesn't automatically free
  pages.
- **Rebuild indexes** when ChromaDB vectors or FTS5 tables become corrupt or
  out of sync with the SQLite data.

These are table stakes for any database that claims "production-ready." Every
SQLite wrapper (better-sqlite3, sql.js, sqlite-vec, sqlite-vss) provides dump,
backup, and integrity checks as a baseline. HybridDB should too.

### Use cases

| Scenario | Method | Why |
|----------|--------|-----|
| **Version control your schema + seed data** | `export_sql` / `import_sql` | SQL dump is text — commit to git, diff, review |
| **Migrate between machines** | `export_sql` / `import_sql` | Dump on laptop, restore on server. Vectors rebuilt on import, no model mismatch |
| **Pre-deploy safety net** | `backup` | Copy entire DB directory. If deployment goes wrong, `restore` the snapshot |
| **CI test fixtures** | `export_sql` | Export known-good state, check into repo, restore in CI |
| **Corrupt ChromaDB recovery** | `reindex` | ChromaDB HNSW index file bloats or corrupts — rebuild vectors from SQLite rows |
| **Disk pressure diagnosis** | `stats` | See which layer is consuming space (SQLite vs ChromaDB vs DuckDB) |
| **After bulk delete cleanup** | `vacuum` | `delete_batch(100K rows)` frees space in SQLite but doesn't shrink the file. `vacuum()` does |
| **Health monitoring in prod** | `check_integrity` | Cron job: `check_integrity()["overall"] != "ok"` → alert |

## Design

Eight new methods on `HybridDB`. All are synchronous. None change schema or
data format. ChromaDB operations gracefully handle `_chroma is None`
(no LONGTEXT columns configured, or ChromaDB initialization failed).

All write operations (`import_sql`, `backup`, `restore`) hold `self._db_lock` for
the duration, blocking concurrent inserts/deletes. Read operations (`stats`,
`check_integrity`) do not block writes.

### Core: Export / Import

#### `export_sql(path: str | Path) -> None`

Standard SQLite `iterdump()`. Writes the entire database (schema + data) as
a portable SQL text file. Does NOT include ChromaDB vectors or DuckDB
analytics — those are rebuilt on import.

Before dumping:
1. Hold `self._db_lock` to prevent concurrent writes
2. Flush pending journal via `_process_journal()` so SQLite is consistent
   with ChromaDB/DuckDB at the export checkpoint
3. Run `PRAGMA wal_checkpoint(TRUNCATE)` to fold the WAL into the main
   `.db` file, ensuring the dump captures all committed data

```python
db.export_sql("./backups/2026-06-01.sql")
```

#### `import_sql(path: str | Path) -> None`

Loads an SQL dump, then rebuilds all indexes:

1. Flush pending journal via `_process_journal()`. Then call `backup()`
   implicitly (to `.old` at `self.path.with_suffix(".import_backup")`) so
   the previous state is recoverable.
2. Open a raw `sqlite3.connect()` on `self._db_path`
3. Drop **all** tables (including `_schema`, `_journal`, and all system
   tables). `iterdump()` emits bare `CREATE TABLE` (no `IF NOT EXISTS`),
   so every table in the dump must be absent for the script to succeed.
   For each dropped table's LONGTEXT columns, also delete the ChromaDB
   collection `{table}_{column}` so old vectors don't leak.
4. Call `conn.executescript(sql)` — creates all tables and inserts all
   rows from the dump. System tables are recreated with their backed-up
   content.
5. `reindex(table=None)` — for each LONGTEXT column: embed all rows → ChromaDB.
   For each TEXT column: rebuild FTS5 via `_rebuild_all_fts5()`.
   For each table: full DuckDB sync via `_full_sync_duckdb_table()`.

No need to rebuild in-memory metadata — `_table_meta()` reads from the `_schema`
table. After step 4, `list_tables()` returns the newly imported tables immediately.

**Destructive.** All existing tables (including system tables) and their
ChromaDB vectors are dropped before import. The previous state is moved to
`.import_backup` in step 1. Call `restore()` from that backup if needed.

After import, `search()` works immediately with full keyword + semantic +
DuckDB analytics.

```python
db.import_sql("./backups/2026-06-01.sql")
```

### Backup / Restore

#### `backup(path: str | Path) -> None`

Copies the entire HybridDB directory (SQLite .db + WAL + ChromaDB vector dir +
analytics.duckdb). Unlike `export_sql()`, this preserves ChromaDB vectors and
DuckDB tables — no rebuilding needed.

- Holds `self._db_lock` during the entire operation to prevent writes
  from creating inconsistent state between journal flush and copy
- Flushes pending journal via `_process_journal()`, then runs
  `PRAGMA wal_checkpoint(TRUNCATE)` to fold the WAL into the main `.db`
  file. Without a clean checkpoint, the OS page cache can serve partial
  reads of `.db-wal` during the file copy loop, producing a corrupt backup.
- No connection close needed — SQLite uses WAL mode and per-operation
  connections. ChromaDB `PersistentClient` does not hold file locks.
- `path` must not exist or must be empty

```python
db.backup("./backups/2026-06-01/")
```

#### `restore(path: str | Path) -> None`

Replaces the current database with a backup directory. The current DB is moved
to a `.old` backup before the restore, so the operation is reversible.

- Holds `self._db_lock` during the entire operation
- Flushes pending journal first
- Moves `self.path` → `self.path.with_suffix(".old")` via `shutil.move()`,
  copies `path` in. If `.old` already exists, appends a timestamp suffix.
- Invalidates the ChromaDB client pool entry for `self._vector_path` before
  reinitializing. Otherwise `_init_chroma()` returns the stale pooled client
  (which still holds in-memory file handles to the old `.old` directory).
  ```python
  with _chroma_pool_lock:
      _chroma_client_pool.pop(self._vector_path, None)
  self._init_chroma(force=True)
  ```
- DuckDB reconnect happens automatically on next `analytics()` call.
- `path` must be a valid HybridDB directory (contains `app.db`) and must exist

```python
db.restore("./backups/2026-06-01/")
```

### Maintenance Utilities

#### `vacuum() -> int`

Rebuilds the SQLite database file, reclaiming free space and defragmenting.
Returns the number of bytes freed (before - after file size).

Calls `conn.execute("VACUUM")`. ChromaDB and DuckDB are not affected.

```python
freed = db.vacuum()
print(f"Reclaimed {freed} bytes")
```

#### `check_integrity() -> dict`

Runs diagnostic checks and returns a structured report. Combines:

- SQLite `PRAGMA integrity_check` — verifies SQLite file structure.
  A failure here means `overall = "corrupt"` — data-loss risk, manual
  restore recommended.
- ChromaDB collection heartbeat — verifies collections exist and respond.
  Failures mean `overall = "degraded"` — recoverable by `reindex()`.
- DuckDB connectivity — verifies DuckDB can read registered tables.
  Failures mean `overall = "degraded"`.

Returns:

```python
{
    "sqlite_integrity": "ok",      # or error message → "corrupt"
    "chromadb_collections": 3,     # count of healthy collections
    "chromadb_errors": [],         # list of broken collections → "degraded"
    "duckdb_tables_synced": 1,     # count of synced tables
    "duckdb_errors": [],           # → "degraded"
    "overall": "ok",               # "ok" | "degraded" | "corrupt"
}
```

Escalation: `integrity_check` failure is always `"corrupt"` regardless of
ChromaDB/DuckDB health. ChromaDB or DuckDB failures with clean SQLite are
`"degraded"`. All clean → `"ok"`.

#### `stats() -> dict`

Returns size and count statistics for all storage layers. `chromadb_size_bytes`
is the total directory size from a recursive walk of `self._vector_path`
(ChromaDB stores all collections in one directory — there is no per-collection
file size API). Per-table `chromadb_vectors` requires calling
`collection.count()` for each `{table}_{column}` collection.

```python
{
    "sqlite_size_bytes": 2097152,
    "chromadb_size_bytes": 4194304,
    "duckdb_size_bytes": 524288,
    "total_size_bytes": 6815744,
    "tables": {
        "messages": {
            "rows": 5000,
            "fts_indexes": 1,
            "chromadb_collections": 1,
            "chromadb_vectors": 5000,
            "duckdb_synced": True,
            "duckdb_rows": 5000,
        },
        ...
    },
}
```

### Reindex

#### `reindex(table: str | None = None) -> None`

Manually rebuild ChromaDB, FTS5, and DuckDB for one or all tables. Called
automatically by `import_sql()`. Useful when the ChromaDB index file becomes
corrupt or out of sync.

- If `table=None`: reindex all user tables
- If `table="messages"`: reindex that table only

For each LONGTEXT column in the table:
1. Read all rows from SQLite in pages. `self.query(table, limit=0)` raises
   the default limit; use a manual offset loop (`LIMIT {CHROMA_BATCH} OFFSET {i}`)
   to avoid loading all rows into memory at once.
2. Batch embed via `self._embedding_fn` using `CHROMA_BATCH` size
3. Delete existing ChromaDB collection `{table}_{column}` via `self._chroma.delete_collection(name)`
4. Create fresh collection, batch upsert embeddings via `self._chroma.get_or_create_collection(name).upsert(...)`

For each TEXT column: call existing `self._rebuild_all_fts5(cur, table)`.

For each table: call existing `self._full_sync_duckdb_table(table)`.

```python
db.reindex("messages")  # rebuild just messages
db.reindex()             # rebuild all
```

## Non-Goals

- No streaming/chunked export. The dump is a single file. Large databases
  (>1GB) may produce large dumps — documented as a known trade-off.
- No incremental/partial export. Always full database.
- No CSV/JSON export. Use `query()` + Python serialization for app-specific
  formats.
- No encrypted or compressed backups. Use filesystem encryption or pipe
  through `gpg`.
- No incremental restore. Always full database replacement.
- No schema-only export. Schema is always bundled with data in the dump.

## SQLite Equivalents

| HybridDB | SQLite / stdlib |
|----------|----------------|
| `db.export_sql("dump.sql")` | `sqlite3 .dump` / `conn.iterdump()` |
| `db.import_sql("dump.sql")` | `sqlite3 .read` / `conn.executescript()` + FTS5 rebuild + vector re-embed |
| `db.backup("./backup/")` | `VACUUM INTO 'backup.db'` + manual WAL/vector copy |
| `db.restore("./backup/")` | Replace `app.db` + reattach |
| `db.vacuum()` | `VACUUM` |
| `db.check_integrity()` | `PRAGMA integrity_check` + `PRAGMA quick_check` |
| `db.stats()` | `SELECT COUNT(*)`, `PRAGMA page_count` |
| `db.reindex()` | No equivalent — HybridDB-specific ChromaDB + FTS5 + DuckDB rebuild |

## API Summary

| Method | Signature | Returns | Side Effects |
|--------|-----------|---------|-------------|
| `export_sql` | `(path: str \| Path) -> None` | — | Flushes journal, writes SQL file |
| `import_sql` | `(path: str \| Path) -> None` | — | Drops ALL tables, rebuilds indexes |
| `backup` | `(path: str \| Path) -> None` | — | Flushes journal, copies directory |
| `restore` | `(path: str \| Path) -> None` | — | Invalidates ChromaDB pool, replaces DB |
| `vacuum` | `() -> int` | Bytes freed | Defragments SQLite |
| `check_integrity` | `() -> dict` | Diagnostic report | None |
| `stats` | `() -> dict` | Size/count report | None |
| `reindex` | `(table: str \| None = None) -> None` | — | Rebuilds ChromaDB+FTS5+DuckDB |

## Testing

- `export_sql` / `import_sql`: create tables → insert data → export →
  fresh DB → import → verify search works. Also: import onto existing DB
  with different data → verify old data replaced, new data correct.
- `backup` / `restore`: create data → backup → delete → restore → verify
- `restore` pool invalidation: create data → backup → delete data →
  restore → verify ChromaDB search works (fails without pool invalidation)
- `export_sql` / `import_sql` with ChromaDB disabled (`_chroma = None`):
  verify paths that skip vector operations work without error
- `vacuum`: insert + delete rows → measure size → vacuum → verify smaller
- `check_integrity`: fresh DB → check → expect `"ok"`. Corrupt ChromaDB
  collection → check → expect `"degraded"`. Corrupt SQLite → check →
  expect `"corrupt"`
- `stats`: create 2 tables → insert 100 rows each → verify stats match
- `reindex`: build fresh → close → corrupt ChromaDB → reopen → reindex →
  verify search works
- `import_sql` embedding model mismatch: change `self._embedding_fn` between
  export and import → verify warning or mismatch detection

## Risks

- **`import_sql()` is destructive.** All tables (including system tables)
  are dropped before executing the dump, so bare `CREATE TABLE` statements
  from `iterdump()` succeed. ChromaDB vectors for the dropped tables are
  also deleted. An automatic backup at `self.path.with_suffix(".import_backup")`
  is created before the drop as a safety net; call `restore()` from there
  if needed.
- **`restore()` ChromaDB pool cache.** The `_chroma_client_pool` will return
  a stale client unless invalidated before `_init_chroma()`. The design
  explicitly pops the pool entry — don't skip this step.
- **Large DB export**: `iterdump()` loads everything into memory. For >1GB
  databases, this is slow and memory-intensive. Document the trade-off.
- **Import with embedding model mismatch**: If the embedding model changed
  between export and import, search results will differ. The model is not
  stored in the dump (but `_schema.embedding_model` is — consider reading
  it from the dump's `_schema` table and warning on mismatch).
- **Concurrent write during export / backup**: Hold `self._db_lock` during
  `export_sql()` and `backup()` to prevent partial writes in the dump.
  Writes will block during export/backup.
- **Reindex embeds everything**: If `self._embedding_fn` calls an external
  API (e.g. OpenAI), `import_sql()` and `reindex()` will be slow and may
  incur costs. The default hash-based embedding avoids this, but custom
  embedders should be aware.
