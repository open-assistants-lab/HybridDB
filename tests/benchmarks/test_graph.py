"""Graph benchmarks: node/edge CRUD, traversal, algorithms."""

import pytest

from .helpers import generate_graph_data

networkx = pytest.importorskip("networkx")


@pytest.fixture
def graph_db(db):
    db.create_table("nodes", {"type": "TEXT", "label": "TEXT"})
    db.create_table("edges", {"source_id": "TEXT", "target_id": "TEXT", "type": "TEXT", "weight": "REAL"})
    return db


def test_add_nodes_batch(benchmark, graph_db, scale):
    nodes, _ = generate_graph_data(scale.n_graph_nodes, 0)

    def _add():
        graph_db.insert_batch("nodes", nodes, sync=False)

    benchmark(_add)


def test_add_edges_batch(benchmark, graph_db, scale):
    nodes, edges = generate_graph_data(scale.n_graph_nodes, scale.n_graph_edges)
    graph_db.insert_batch("nodes", nodes, sync=False)

    def _add():
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
