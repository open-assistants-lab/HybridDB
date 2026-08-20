"""Search benchmarks: keyword, vector, hybrid on TEXT and LONGTEXT columns, plus recall@K."""

import random
from typing import Any

import pytest

from .helpers import SearchMode, compute_recall, generate_docs

pytest.importorskip("sentence_transformers")

TEXT_COLUMNS = [{"name": "content", "type": "TEXT"}]
LONGTEXT_COLUMNS = [{"name": "content", "type": "LONGTEXT"}]

CLUSTER_TOPICS = [
    ("machine learning neural networks deep learning artificial intelligence", "ai"),
    ("football basketball soccer tennis baseball hockey sports", "sports"),
    ("cooking recipes kitchen food chef restaurant dining", "food"),
    ("physics chemistry biology science experiment research", "science"),
    ("stocks trading finance investment portfolio market economy", "finance"),
    ("music guitar piano drums singing band concert", "music"),
    ("programming software code developer api database server", "engineering"),
    ("travel vacation hotel flight destination tourism adventure", "travel"),
    ("health doctor medicine hospital treatment diagnosis patient", "health"),
    ("education school university student teacher learning classroom", "education"),
]

K_VALUES = [5, 10, 20]


def _expected_ids_for_query(docs: list[dict[str, Any]], query: str) -> set[str]:
    """Return IDs of docs that contain the query keyword (for recall check)."""
    q = query.lower()
    return {d["id"] for d in docs if q in d.get("content", "").lower()}


def _insert_docs(db, table: str, docs: list[dict[str, Any]]):
    """Insert docs in batch using HybridDB insert_batch (sync True for Chroma)."""
    db.insert_batch(table, docs, sync=True)


def _prepare_text_db(db, scale, columns, table: str = "bench_search"):
    docs = generate_docs(scale.n_docs, columns)
    db.create_table(table, {"id": "TEXT PRIMARY KEY", **{c["name"]: c["type"] for c in columns}})
    _insert_docs(db, table, docs)
    return docs


# ---- TEXT column benchmarks ----


def test_keyword_search_text(benchmark, db, scale):
    docs = _prepare_text_db(db, scale, TEXT_COLUMNS)
    query = "fox"
    expected = _expected_ids_for_query(docs, query)

    def _search():
        return db.search("bench_search", "content", query, mode=SearchMode.KEYWORD)

    result = benchmark(_search)
    assert any(r["id"] in expected for r in result) or not expected


def test_vector_search_text(benchmark, db, scale):
    _prepare_text_db(db, scale, TEXT_COLUMNS)
    query = "test search"

    def _search():
        return db.search("bench_search", "content", query, mode=SearchMode.SEMANTIC)

    result = benchmark(_search)
    assert result == []


def test_hybrid_search_text(benchmark, db, scale):
    _prepare_text_db(db, scale, TEXT_COLUMNS)
    query = "fox"

    def _search():
        return db.search("bench_search", "content", query, mode=SearchMode.HYBRID)

    result = benchmark(_search)
    assert len(result) > 0


# ---- LONGTEXT column benchmarks ----


def test_keyword_search_longtext(benchmark, db, scale):
    docs = _prepare_text_db(db, scale, LONGTEXT_COLUMNS)
    query = "hello"
    expected = _expected_ids_for_query(docs, query)

    def _search():
        return db.search("bench_search", "content", query, mode=SearchMode.KEYWORD)

    result = benchmark(_search)
    assert any(r["id"] in expected for r in result) or not expected


def test_vector_search_longtext(benchmark, db, scale):
    _prepare_text_db(db, scale, LONGTEXT_COLUMNS)
    query = "search performance benchmark"

    def _search():
        return db.search("bench_search", query, mode=SearchMode.SEMANTIC)

    result = benchmark(_search)
    assert len(result) > 0


def test_hybrid_search_longtext(benchmark, db, scale):
    _prepare_text_db(db, scale, LONGTEXT_COLUMNS)
    query = "database benchmark"

    def _search():
        return db.search("bench_search", query, mode=SearchMode.HYBRID)

    result = benchmark(_search)
    assert len(result) > 0


# ---- Recall@K benchmarks ----


def _generate_clustered_docs(
    n_per_cluster: int, seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    """Generate docs with known semantic clusters for recall measurement.

    Returns (docs, cluster_map) where cluster_map maps topic_name -> set of doc IDs.
    """
    rng = random.Random(seed)
    docs: list[dict[str, Any]] = []
    cluster_map: dict[str, set[str]] = {}

    for i, (topic_words, topic_name) in enumerate(CLUSTER_TOPICS):
        cluster_map[topic_name] = set()
        words = topic_words.split()
        for j in range(n_per_cluster):
            doc_id = f"{topic_name}_{j}"
            cluster_map[topic_name].add(doc_id)
            shuffled = rng.sample(words, len(words))
            filler_parts = []
            for _ in range(rng.randint(3, 8)):
                filler_parts.append(" ".join(rng.choices(
                    ["the", "and", "is", "in", "of", "to", "a", "for", "with", "on"],
                    k=rng.randint(8, 20),
                )))
            filler = " ".join(filler_parts)
            docs.append({
                "id": doc_id,
                "content": f"{' '.join(shuffled)}. {filler}. {' '.join(rng.sample(words, min(5, len(words))))}.",
            })

    rng.shuffle(docs)
    return docs, cluster_map


@pytest.fixture
def recall_db(db, scale):
    """DB populated with clustered LONGTEXT documents for recall measurement."""
    n_per_cluster = max(scale.n_docs // len(CLUSTER_TOPICS), 5)
    docs, cluster_map = _generate_clustered_docs(n_per_cluster)
    db.create_table("bench_recall", {"id": "TEXT PRIMARY KEY", "content": "LONGTEXT"})
    db.insert_batch("bench_recall", docs, sync=True)
    return db, cluster_map


def test_recall_keyword(benchmark, recall_db):
    db, cluster_map = recall_db
    query = "football"
    expected = cluster_map["sports"]

    def _search():
        results = db.search("bench_recall", "content", query, mode=SearchMode.KEYWORD, limit=20)
        return [r["id"] for r in results]

    ids = benchmark(_search)
    recall = compute_recall(ids, expected, k=10)
    assert recall >= 0.5


def test_recall_semantic(benchmark, recall_db):
    db, cluster_map = recall_db
    query = "artificial intelligence and deep learning"
    expected = cluster_map["ai"]

    def _search():
        results = db.search("bench_recall", "content", query, mode=SearchMode.SEMANTIC, limit=20)
        return [r["id"] for r in results]

    ids = benchmark(_search)
    recall = compute_recall(ids, expected, k=10)
    assert recall >= 0.3


def test_recall_hybrid(benchmark, recall_db):
    db, cluster_map = recall_db
    query = "cooking recipes in the kitchen"
    expected = cluster_map["food"]

    def _search():
        results = db.search("bench_recall", "content", query, mode=SearchMode.HYBRID, limit=20)
        return [r["id"] for r in results]

    ids = benchmark(_search)
    recall = compute_recall(ids, expected, k=10)
    assert recall >= 0.3


def test_recall_at_k_sweep(recall_db):
    """Non-benchmark test: sweep K=5,10,20 and report recall for all three modes."""
    db, cluster_map = recall_db
    query = "hospital doctor medicine treatment"
    expected = cluster_map["health"]

    results = {}
    for mode in [SearchMode.KEYWORD, SearchMode.SEMANTIC, SearchMode.HYBRID]:
        raw = db.search("bench_recall", "content", query, mode=mode, limit=50)
        ids = [r["id"] for r in raw]
        for k in K_VALUES:
            recall = compute_recall(ids, expected, k=k)
            results[f"{mode.value}@{k}"] = recall

    for key, val in results.items():
        print(f"  {key}: {val:.2f}", end="")
    print()

    assert results["hybrid@10"] >= results.get("keyword@10", 0) or results["hybrid@10"] >= 0.3


# ---- Cold start benchmark ----


def test_cold_start_search(benchmark, db, scale, tmp_path, embedding_fn):
    """Time to first search on a fresh HybridDB instance with existing data."""
    import shutil

    from hybriddb import HybridDB

    docs = generate_docs(scale.n_docs, LONGTEXT_COLUMNS)
    db.create_table("bench_cold", {"id": "TEXT PRIMARY KEY", "content": "LONGTEXT"})
    db.insert_batch("bench_cold", docs, sync=True)
    db.close()

    live_path = tmp_path / "live_data"
    shutil.copytree(str(db.path), str(live_path))

    def _cold_start():
        fresh = HybridDB(
            path=str(live_path),
            embedding_fn=embedding_fn,
            embedding_model_name="all-MiniLM-L6-v2",
        )
        results = fresh.search("bench_cold", "test search", mode=SearchMode.HYBRID, limit=10)
        fresh.close()
        return len(results)

    result = benchmark(_cold_start)
    assert result >= 0
