"""How risky is the chromadb sparse-vector segfault for HybridDB? (issue #5)

Measures the crash rate as a function of vector SPARSITY and total WRITE
VOLUME, in pure chromadb (no HybridDB), by exit code (139 = SIGSEGV).

Usage: chroma_sparsity_matrix.py <rounds> <per_round> <md5|N>
  rounds      cycles of (upsert per_round, delete per_round)
  per_round   records upserted per round
  md5         word-hash embedder style (the known trigger)
  N           exact count of non-zero dims out of 384 (384 = dense)
"""

import hashlib
import os
import shutil
import sys
import tempfile

import chromadb
from chromadb.config import Settings as ChromaSettings

D = 384


def md5_word_vector(text: str) -> list[float]:
    """The embedder style the failing test uses: hash each WORD to one dim."""
    vec = [0.0] * D
    for word in text.lower().split():
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % D] += 1.0
    mag = sum(x * x for x in vec) ** 0.5
    return [x / mag for x in vec] if mag else vec


SHARED_DIMS = (7, 23, 41, 59, 83, 97, 113, 131)


def sim_vector(varying: int, text: str) -> list[float]:
    """Near-duplicate cluster: `varying` dims differ per document, the rest are
    shared by every document. This is what a bag-of-words embedder produces for
    short, similar sentences -- and, unlike raw sparsity, it is also true of
    real embedders to a lesser degree."""
    vec = [0.0] * D
    for d in SHARED_DIMS:
        vec[d] += 1.0
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
    for k in range(varying):
        vec[(seed + k * 37) % D] += 1.0
    mag = sum(x * x for x in vec) ** 0.5
    return [x / mag for x in vec] if mag else vec


_HYBRIDDB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hdb_fallback_vector(text: str):
    """HybridDB's real *offline fallback* embedding: hash_embedding(), used
    when Chroma's MiniLM cannot load. This is the shipped code."""
    import sys as _s
    _s.path.insert(0, _HYBRIDDB)
    from hybriddb.embedding import hash_embedding
    return hash_embedding(text)


def hdb_default_vector(text: str):
    """HybridDB's shipped default: Chroma's MiniLM when available, else the
    hash fallback. Measured to see which path actually executes."""
    import sys as _s
    _s.path.insert(0, _HYBRIDDB)
    from hybriddb.embedding import default_embedding_fn
    return default_embedding_fn(text)


def vector(spec: str, text: str) -> list[float]:
    """spec: "md5" | "simN" | "hdb" (real HybridDB fallback) | int."""
    if spec == "md5":
        return md5_word_vector(text)
    if spec == "hdb":
        return hdb_fallback_vector(text)
    if spec == "minilm":
        out = hdb_default_vector(text)
        return out if isinstance(out, list) else list(out)
    if spec.startswith("sim"):
        return sim_vector(int(spec[3:]), text)
    nonzero = int(spec)
    vec = [0.0] * D
    if nonzero <= 0:
        return vec
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
    for k in range(nonzero):
        vec[(seed + k * 37) % D] += 1.0
    mag = sum(x * x for x in vec) ** 0.5
    return [x / mag for x in vec] if mag else vec


def main() -> None:
    rounds, per_round, nonzero = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    tmp = tempfile.mkdtemp()
    try:
        client = chromadb.PersistentClient(
            path=os.path.join(tmp, "vectors"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        col = client.get_or_create_collection("matrix")

        def push(ids: list[str], prefix: str) -> None:
            for start in range(0, len(ids), 500):
                part = ids[start : start + 500]
                col.upsert(
                    ids=part,
                    embeddings=[vector(nonzero, f"{prefix} {i}") for i in part],
                    documents=[f"{prefix} entry {i}" for i in part],
                )

        base = [f"k{i}" for i in range(2000)]
        push(base, "knowledge entry lorem ipsum")
        for r in range(rounds):
            ids = [f"x{r}_{i}" for i in range(per_round)]
            push(ids, f"extra entry {r}")
            col.delete(ids=ids)
        print("COMPLETED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
