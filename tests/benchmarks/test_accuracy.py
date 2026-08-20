"""Search accuracy evaluation on BEIR NFCorpus (graded qrels).

Measures retrieval quality — nDCG@10, recall@10, precision@10, MRR@10 —
for keyword, semantic, and hybrid search over 324 real queries with graded
relevance judgments (0-3), plus fusion-weight and embedding-model sensitivity.

Run with:  uv run python -m pytest tests/benchmarks/test_accuracy.py \
               --run-benchmarks --benchmark-disable
"""

import math
import statistics

import pytest

from hybriddb import LONGTEXT, TEXT, HybridDB, SearchMode
from hybriddb.embedding import hash_embedding

from .datasets import load_beir

K = 10


# ── IR metrics (graded relevance) ─────────────────────────────────────────

def _dcg(rels: list[int], k: int = K) -> float:
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def _ndcg_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int = K) -> float:
    rels = [qrels.get(doc_id, 0) for doc_id in ranked_ids[:k]]
    ideal = sorted(qrels.values(), reverse=True)
    dcg = _dcg(rels, k)
    idcg = _dcg(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int = K) -> float:
    relevant = {doc_id for doc_id, s in qrels.items() if s >= 1}
    if not relevant:
        return 0.0
    hit = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant)
    return hit / len(relevant)


def _precision_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int = K) -> float:
    relevant = {doc_id for doc_id, s in qrels.items() if s >= 1}
    if not ranked_ids:
        return 0.0
    hit = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant)
    return hit / k


def _mrr(ranked_ids: list[str], qrels: dict[str, int]) -> float:
    relevant = {doc_id for doc_id, s in qrels.items() if s >= 1}
    for i, doc_id in enumerate(ranked_ids):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _evaluate(db, queries, qrels, mode, fts_weight: float = 0.5, k: int = K,
              table: str = "beir") -> dict:
    """Average IR metrics over all queries for one search configuration."""
    ndcg, recall, precision, mrr = [], [], [], []
    for q in queries:
        ranked = db.search(
            table, "content", q["text"], mode=mode, limit=k, fts_weight=fts_weight,
        )
        ids = [r["id"] for r in ranked]
        rel = qrels.get(q["id"], {})
        ndcg.append(_ndcg_at_k(ids, rel, k))
        recall.append(_recall_at_k(ids, rel, k))
        precision.append(_precision_at_k(ids, rel, k))
        mrr.append(_mrr(ids, rel))
    return {
        "ndcg": statistics.mean(ndcg),
        "recall": statistics.mean(recall),
        "precision": statistics.mean(precision),
        "mrr": statistics.mean(mrr),
    }


def _fmt(m: dict) -> str:
    return (f"nDCG@{K}={m['ndcg']:.4f}  recall@{K}={m['recall']:.4f}  "
            f"P@{K}={m['precision']:.4f}  MRR={m['mrr']:.4f}")


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def beir():
    try:
        return load_beir("nfcorpus")
    except Exception as e:  # offline or download failure
        pytest.skip(f"BEIR NFCorpus unavailable: {e}")


@pytest.fixture(scope="session")
def scifact():
    try:
        return load_beir("scifact")
    except Exception as e:  # offline or download failure
        pytest.skip(f"BEIR SciFact unavailable: {e}")


def _build_db(tmp_path, embedding_fn, data) -> HybridDB:
    db = HybridDB(str(tmp_path), embedding_fn=embedding_fn)
    db.create_table("beir", {"id": "TEXT PRIMARY KEY", "content": LONGTEXT})
    docs = [
        {"id": d["_id"], "content": f"{d['title']}. {d['text']}".strip()}
        for d in data["docs"]
    ]
    db.insert_batch("beir", docs)
    return db


@pytest.fixture(scope="session")
def accuracy_db(embedding_fn, tmp_path_factory, beir):
    """NFCorpus embedded with MiniLM."""
    return _build_db(tmp_path_factory.mktemp("acc_minilm"), embedding_fn, beir)


@pytest.fixture(scope="session")
def hash_db(tmp_path_factory, beir):
    """NFCorpus embedded with the dependency-free hash fallback."""
    return _build_db(tmp_path_factory.mktemp("acc_hash"), hash_embedding, beir)


@pytest.fixture(scope="session")
def scifact_db(embedding_fn, tmp_path_factory, scifact):
    """SciFact embedded with MiniLM."""
    return _build_db(tmp_path_factory.mktemp("scifact_minilm"), embedding_fn, scifact)


# ── accuracy tests ─────────────────────────────────────────────────────────

class TestAccuracyModes:
    def test_mode_comparison(self, accuracy_db, beir):
        """Headline: keyword vs semantic vs hybrid on the same 324 queries."""
        results = {}
        for mode in (SearchMode.KEYWORD, SearchMode.SEMANTIC, SearchMode.HYBRID):
            m = _evaluate(accuracy_db, beir["queries"], beir["qrels"], mode)
            results[mode.value] = m
            print(f"  {mode.value:9s}: {_fmt(m)}")
        # fusion must not degrade below the better single mode by much
        best_single = max(results["keyword"]["ndcg"], results["semantic"]["ndcg"])
        assert results["hybrid"]["ndcg"] >= best_single - 0.02
        # keyword (BM25) is a strong baseline on NFCorpus; semantic must be close
        assert results["semantic"]["ndcg"] >= results["keyword"]["ndcg"] - 0.05

    def test_fts_weight_sweep(self, accuracy_db, beir):
        """How sensitive is hybrid accuracy to the keyword/semantic balance?"""
        for w in (0.2, 0.5, 0.8):
            m = _evaluate(accuracy_db, beir["queries"], beir["qrels"],
                          SearchMode.HYBRID, fts_weight=w)
            print(f"  fts_weight={w}: {_fmt(m)}")
        # sanity: all weights must be in a sane band (fusion is robust)
        ndcgs = [
            _evaluate(accuracy_db, beir["queries"], beir["qrels"],
                      SearchMode.HYBRID, fts_weight=w)["ndcg"]
            for w in (0.2, 0.5, 0.8)
        ]
        assert max(ndcgs) - min(ndcgs) < 0.05

    def test_embedding_gap(self, accuracy_db, hash_db, beir):
        """Accuracy lost when falling back to the hash embedding (no model)."""
        mini = _evaluate(accuracy_db, beir["queries"], beir["qrels"], SearchMode.SEMANTIC)
        hashed = _evaluate(hash_db, beir["queries"], beir["qrels"], SearchMode.SEMANTIC)
        print(f"  MiniLM semantic: {_fmt(mini)}")
        print(f"  hash   semantic: {_fmt(hashed)}")
        assert mini["ndcg"] > hashed["ndcg"]  # the model must actually help


class TestScifactAccuracy:
    """Second domain: scientific claim verification (binary qrels)."""

    def test_mode_comparison(self, scifact_db, scifact):
        results = {}
        for mode in (SearchMode.KEYWORD, SearchMode.SEMANTIC, SearchMode.HYBRID):
            m = _evaluate(scifact_db, scifact["queries"], scifact["qrels"], mode)
            results[mode.value] = m
            print(f"  {mode.value:9s}: {_fmt(m)}")
        best_single = max(results["keyword"]["ndcg"], results["semantic"]["ndcg"])
        assert results["hybrid"]["ndcg"] >= best_single - 0.02

    def test_fts_weight_sweep(self, scifact_db, scifact):
        for w in (0.2, 0.5, 0.8):
            m = _evaluate(scifact_db, scifact["queries"], scifact["qrels"],
                          SearchMode.HYBRID, fts_weight=w)
            print(f"  fts_weight={w}: {_fmt(m)}")
        ndcgs = [
            _evaluate(scifact_db, scifact["queries"], scifact["qrels"],
                      SearchMode.HYBRID, fts_weight=w)["ndcg"]
            for w in (0.2, 0.5, 0.8)
        ]
        assert max(ndcgs) - min(ndcgs) < 0.05
