"""Graph benchmarks: node/edge CRUD, traversal, algorithms."""

import pytest

from .helpers import generate_graph_data

networkx = pytest.importorskip("networkx")


@pytest.fixture
def graph_db(db):
    # nodes use a TEXT primary key: generated ids are strings like "n0"
    db.create_table("nodes", {"id": "TEXT PRIMARY KEY", "type": "TEXT", "label": "TEXT"})
    db.create_table("edges", {"id": "TEXT PRIMARY KEY", "source_id": "TEXT", "target_id": "TEXT", "type": "TEXT", "weight": "REAL"})
    return db


@pytest.fixture
def semantic_graph_db(db, scale):
    """Graph with LONGTEXT node content + edges, for semantic graph retrieval."""
    from .helpers import generate_docs, generate_graph_data

    n_nodes = scale.n_graph_nodes
    nodes, edges = generate_graph_data(n_nodes, scale.n_graph_edges)
    docs = generate_docs(n_nodes, [{"name": "content", "type": "LONGTEXT"}])
    db.create_table("entities", {"id": "TEXT PRIMARY KEY", "content": "LONGTEXT"})
    rows = [{"id": n["id"], "content": docs[i]["content"]} for i, n in enumerate(nodes)]
    db.insert_batch("entities", rows, sync=False)
    db.register_entity_node("entities", id_column="id")
    db.sync_graph_nodes()
    # edges must reference the namespaced synced node ids (entities:n0)
    ns_edges = [
        {"id": e["id"], "source_id": f"entities:{e['source_id']}",
         "target_id": f"entities:{e['target_id']}", "type": e["type"], "weight": e["weight"]}
        for e in edges
    ]
    db.add_edges(ns_edges)
    return db


def test_add_nodes_batch(benchmark, graph_db, scale):
    nodes, _ = generate_graph_data(scale.n_graph_nodes, 0)

    def _add():
        graph_db.raw_query("DELETE FROM nodes")
        graph_db.insert_batch("nodes", nodes, sync=False)

    benchmark(_add)


def test_add_edges_batch(benchmark, graph_db, scale):
    nodes, edges = generate_graph_data(scale.n_graph_nodes, scale.n_graph_edges)
    graph_db.insert_batch("nodes", nodes, sync=False)

    def _add():
        graph_db.raw_query("DELETE FROM edges")
        graph_db.insert_batch("edges", edges)

    benchmark(_add)


def _load_graph(graph_db, n_nodes: int, n_edges: int):
    nodes, edges = generate_graph_data(n_nodes, n_edges)
    graph_db.add_nodes(nodes)
    graph_db.add_edges(edges)
    return nodes, edges


def test_get_neighbors(benchmark, graph_db, scale):
    nodes, _ = _load_graph(graph_db, scale.n_graph_nodes, scale.n_graph_edges)

    target = nodes[len(nodes) // 2]["id"]

    def _neighbors():
        return graph_db.get_neighbors(target)

    benchmark(_neighbors)


def test_shortest_path(benchmark, graph_db, scale):
    nodes, _ = _load_graph(graph_db, scale.n_graph_nodes, scale.n_graph_edges)

    src = nodes[0]["id"]
    dst = nodes[-1]["id"]

    def _path():
        return graph_db.shortest_path(src, dst)

    benchmark(_path)


def test_pagerank(benchmark, graph_db, scale):
    _load_graph(graph_db, scale.n_graph_nodes, scale.n_graph_edges)

    def _pr():
        return graph_db.pagerank()

    benchmark(_pr)


def test_decay_edges(benchmark, graph_db, scale):
    _load_graph(graph_db, 100, 500)

    def _decay():
        return graph_db.decay_edges()

    benchmark(_decay)


# ── Semantic graph retrieval (flagship) ──────────────────────────────────

def test_search_graph(benchmark, semantic_graph_db):
    def _sg():
        return semantic_graph_db.graph.search_graph("lorem ipsum", hop_expansion=1, limit=5)

    result = benchmark(_sg)
    assert isinstance(result, list)


def test_search_graph_ppr(benchmark, semantic_graph_db):
    def _ppr():
        return semantic_graph_db.graph.search_graph_ppr("lorem ipsum", hop_expansion=1, limit=5)

    result = benchmark(_ppr)
    assert isinstance(result, list)


# ── Traversal & NetworkX ─────────────────────────────────────────────────

def test_traverse_depth3(benchmark, graph_db, scale):
    _load_graph(graph_db, scale.n_graph_nodes, scale.n_graph_edges)

    def _trav():
        return graph_db.traverse("n0", max_depth=3, direction="both")

    result = benchmark(_trav)
    assert isinstance(result, list)


def test_to_networkx_build(benchmark, graph_db, scale):
    _load_graph(graph_db, scale.n_graph_nodes, scale.n_graph_edges)

    def _build():
        return graph_db.to_networkx(use_cache=False)

    g = benchmark(_build)
    assert g.number_of_nodes() == scale.n_graph_nodes


def test_to_networkx_cache_hit(benchmark, graph_db, scale):
    _load_graph(graph_db, scale.n_graph_nodes, scale.n_graph_edges)
    graph_db.to_networkx()  # warm the cache

    def _cached():
        return graph_db.to_networkx(use_cache=True)

    g = benchmark(_cached)
    assert g.number_of_nodes() == scale.n_graph_nodes


# ── Graph sync & algorithms at scale ──────────────────────────────────────

def test_sync_graph_nodes(benchmark, db, scale):
    from .helpers import generate_docs

    db.create_table("entities", {"id": "TEXT PRIMARY KEY", "content": "LONGTEXT"})
    docs = generate_docs(scale.n_graph_nodes, [{"name": "content", "type": "LONGTEXT"}])
    db.insert_batch("entities", docs, sync=False)
    db.register_entity_node("entities", id_column="id")

    def _sync():
        return db.sync_graph_nodes()

    benchmark(_sync)
    # repeated runs are idempotent; the graph must end with one node per row
    assert len(db.graph.list_nodes(limit=0)) == scale.n_graph_nodes


def test_community_detect(benchmark, graph_db, scale):
    _load_graph(graph_db, scale.n_graph_nodes, scale.n_graph_edges)

    def _cd():
        return graph_db.community_detect()

    result = benchmark(_cd)
    assert isinstance(result, list)


def test_betweenness_centrality(benchmark, graph_db, scale):
    # O(N*E) — cap the size so the benchmark stays tractable
    n = min(scale.n_graph_nodes, 1000)
    m = min(scale.n_graph_edges, 5000)
    _load_graph(graph_db, n, m)

    def _bc():
        return graph_db.betweenness_centrality()

    result = benchmark(_bc)
    assert isinstance(result, dict)
