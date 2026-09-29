# Note: chromadb 1.5.9 native segfault during batched upserts (issue #5)

Status: **upstream chromadb bug**, reproduced in pure chromadb, with a real
HybridDB exposure path. Tracked in
https://github.com/open-assistants-lab/HybridDB/issues/5
— keep this doc in step with that issue.

## What happens

`chromadb` 1.5.9's native Rust bindings crash the interpreter with SIGSEGV
(exit 139) from `Collection.upsert` (`chromadb/api/rust.py::_upsert`), reached
through the ordinary journal path. It is a native memory-safety failure, so it
cannot be caught or retried in-process.

## The trigger

It is **not** batch size, not total record volume on its own, and — contra my
first analysis — **not sparsity alone**, and not vector similarity alone.

What reproduces:

- a **word-hash embedding** (`md5(word) % dims`, one increment per word,
  normalized) — i.e. our `hash_embedding` — interleaved with deletions,
  at volume and across repeated upsert/delete cycles.

What does NOT reproduce (6-10 runs each, matching call pattern):

- dense vectors (384/384 non-zero)
- sparseness alone: exactly 2, 4, 8, 24 or 96 non-zero dims out of 384
- near-duplicate clusters of 9 dims with very high cosine similarity (~0.89)
- HybridDB's journal path at 30k upserts **without** deletion

The distinguishing property is most likely tied to the specific structure the
word-hash construction produces — **collisions** (several words landing in one
dim, giving components of 2.0+) combined with a very small support. I did not
fully pin this down; what matters for us is the next line.

## The verdict on impact — corrected twice

1. My first assessment ("not reachable in practice") was **wrong**.
2. Our shipped offline fallback, `hybriddb.embedding.hash_embedding`, is
   **the same word-hash construction** that triggers the crash. It executes
   whenever Chroma's bundled MiniLM model cannot load.

Measured, pure chromadb, no HybridDB:

| arm (30k upserts + 30k deletes) | result |
|---|---|
| dense control (384/384 non-zero) | 3/3 clean |
| **`hybriddb.embedding.hash_embedding`** | **3/3 SIGSEGV (139)** |
| md5 word-hash control (identical algorithm) | 3/3 SIGSEGV (139) |

Measured against the **real HybridDB journal path**:

| workload | result |
|---|---|
| fallback forced, 30k upserts, deletes interleaved (rollback) | **4/4 SIGSEGV** |
| fallback, ~30k upserts, no deletes | 5/5 clean |
| fallback, 6k upserts, deletes interleaved | 5/5 clean |
| **shipped default (MiniLM), dense, 30k upserts + 30k deletes** | **5/5 clean** |

## Who is affected

Not everyone. Both conditions together are required:

1. Chroma's MiniLM model is **unavailable** — so `default_embedding_fn` falls
   back to `hash_embedding`. This is exactly the air-gapped / restricted-network
   / stripped-down environment case. `_get_default_ef()` fails quietly and the
   fallback ships vectors.
2. **Deletions interleaved with bulk writes at volume** — agent memory churn.

When both hold, the crash is **deterministic** (4/4), not flaky.

The shipped default (MiniLM loaded) is dense and did not crash at 30k upserts
plus 30k deletes. So normal online usage appears safe.

Note the fallback is documented for offline smoke tests, not production — but
nothing stops a user from running that way.

## Mitigation options, in order of preference

1. **Make the hash fallback dense.** The fallback's vectors carry no semantic
   weight; making it emit a dense pattern (e.g. a fixed spectral seed per text)
   would move every user onto the profile measured at 5/5 and 0/8 clean. This is
   the only fix that removes exposure without touching chromadb. It changes
   existing fallback-indexed stores (they would need a `reindex()`), so it
   should be versioned and noted in the changelog.
2. **Detect the risk and say so.** `create_table()` already knows where
   embeddings come from; a warning when the active embedding is the offline
   fallback *and* a volume threshold nears would turn a segfault into a
   diagnostic. Cheap, safe, and honest.
3. **Cap concurrent delete+upsert churn** in the journal — possible, but it
   slows a hot path and the chroma behavior is probabilistic; I do not recommend
   it as the primary fix.
4. **Switch `chroma_api_impl`** to the non-Rust segment implementation —
   untested; if it is unaffected, that is a real workaround worth measuring.

## Do not attempt

Repeating the churn/rollback cycle to take a median of wall-clock timings in
`tests/test_versioning.py`. That triples the Chroma write volume and reliably
provokes this crash (6/10 on Python 3.13, 6/15 on 3.14, versus 0/16 for the
shipped single-round test on both). The perf gate's margin was widened to 3x
instead.

## Reproducers

- `scripts/chroma_sparsity_matrix.py` — pure chromadb, parameterized by
  embedding style, round count, and record count; sparsity / similarity /
  density arms.
- `scripts/segfault_attribution.py` — earlier bisect arms; the `md5`
  word-hash arm is the trigger and the `hash` arm is the safe control.
---

## Measurement notes (so the numbers above can be trusted and re-checked)

- Crash detection is by **exit code 139** (SIGSEGV), not by message text. Without
  a fault handler installed, a segfaulting process prints nothing, so
  text-grepping undercounts: several earlier arms looked clean and were not.
- All arms below ran the full workload per process. A single process accumulating
  prior chroma work can change the rate, so arm-level numbers were taken from
  independent processes.
- Vector styles were validated against the real code by importing
  `hybriddb.embedding` and comparing outputs:
  - `default_embedding_fn(text)` returned a **numpy ndarray, 384/384 non-zero**
    (MiniLM loaded), and after the offline path was forced it returned exactly
    `hash_embedding(text)` (4/384 non-zero).
  - `_get_default_ef()` returning `None` is how the fallback engages.
