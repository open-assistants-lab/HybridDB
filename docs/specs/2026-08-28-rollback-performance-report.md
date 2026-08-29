# Rollback Performance Report — versioned tables (handoff from CoreMem 0.14.0)

**Date:** 2026-08-28 · **Reporter:** CoreMem perf gate (`scripts/bench_versioned_memory.py`)
**hybriddb:** 0.6.0 · **Python:** 3.13, macOS arm64 (CPU-side embeddings)
**Severity:** non-blocking for CoreMem (interactive governance op), but a clear
batch-efficiency opportunity — CoreMem's acceptance gate for memory rollback was
"≤ ingest time of the same rows", and this measured ~3× over.

---

## Measurement (exact repro below)

Append-heavy rollback scenario — the case the 0.6.0 changelog calls "cheap
(Chroma deletions, not re-embeds)":

| Step | Time |
|---|---:|
| Ingest 1,000 rows (batched, 500/batch) | **1.25 s** |
| `rollback(messages, checkpoint="pre-extra")` — removes those 1k rows | **3.85 s** |
| → rollback / ingest ratio | **~3.1×** |

Per-row: ~3.8 ms per removed row (delete tombstone + journal `delete` op +
Chroma deletion + journal processing). Chain stayed **valid** throughout
(`verify_chain` after rollback: `{'valid': True}`, 12k events). Count restored
exactly. For reference at the same scale:

- Full 10k-row ingestion (versioned): 12.45 s median (804 rows/s) — **+8.2%**
  over non-versioned (869 rows/s) — well within CoreMem's gate
- `verify_chain` at 12k events: **0.05 s** — excellent, no concern
- Storage overhead versioned: 1.13× after 10k inserts
- Recall latency: unchanged (63.5 vs 66.7 ms warm, 10-run median)

Correctness was never in question in any CoreMem spike/run: 200/200 history
events recorded through the batched path, `verify_chain` valid after every
mutator, tombstones land for `delete()`. This report is purely a
**batch-efficiency** ask.

## Repro

```python
# CoreMem's bench: scripts/bench_versioned_memory.py --rows 10000
# (uses CoreMem's batched ingest: insert_batch(sync=False) + shadowed
#  embedding flush — see coremem/core.py::_flush_journal_batched)
db.create_table("messages", {...TEXT PK, LONGTEXT content...}, versioned=True)
for batch in chunks(rows, 500):
    db.insert_batch("messages", batch, sync=False)   # + CoreMem journal flush
db.checkpoint("messages", "pre-extra")
# ingest 1,000 more rows ...                      # 1.25 s
db.rollback("messages", checkpoint="pre-extra")   # 3.85 s
db.verify_chain("messages")                       # valid
```

## Questions for investigation

1. **Journal `delete` ops** — are they applied one Chroma call per row during
   journal processing? A single batched `collection.delete(ids=[...])` should
   cover 1k removals in one round-trip.
2. **Rollback scan scope** — does `rollback()` re-process the full history
   table, or only events after the checkpoint seq? Should be O(since) for
   append-heavy cases.
3. **Tombstone writes** — are the 1k delete tombstones committed per-row (1k
   implicit transactions/fsyncs) or batched in one transaction? The hash chain
   is sequential per link, but that only forces ordering, not per-row commits.
4. **No-op second table** — CoreMem calls rollback on `messages` AND
   `journal_records`; `journal_records` had zero changes since the checkpoint.
   Does a no-change rollback still scan/rewrite anything measurable?
5. **Unexpected re-embeds?** — the changelog note says append-heavy rollback is
   "deletions, not re-embeds". Confirm no restored-row re-embedding happens in
   the pure-removal case.

## Acceptance target

CoreMem's gate rerun would want rollback of 1k removed rows **≤ ~1.25 s**
(i.e. ≥ 3× faster than today's 3.85 s), matching the "deletions only" design
intent. With that, CoreMem's perf gate passes clean end-to-end.

## Pointers

- `hybriddb/versioning.py` — `rollback()` (~line 245), history/tombstone path
- journal processing for `delete` ops (one-call vs per-row Chroma deletion)
- CoreMem cross-references: `docs/versioned-memory-design.md` §7 (gate table +
  measured numbers), `scripts/bench_versioned_memory.py` (repro script),
  `results/` bench outputs (2026-08-28 run)
- Prior spikes confirmed correctness (history recording, tombstones,
  `verify_chain`) — nothing to fix there