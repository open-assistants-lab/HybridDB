"""Thin namespaced API facades for HybridDB."""

import uuid
from typing import Any


class GraphAPI:
    """Namespaced graph operations exposed as db.graph.*."""

    _METHODS = {
        "register_entity_node", "register_edge_rule", "add_node", "add_nodes",
        "get_node", "update_node", "delete_node", "list_nodes", "add_edge",
        "add_edges", "get_edge", "update_edge", "delete_edge", "get_edges",
        "neighbors", "get_neighbors", "traverse", "decay_edges", "to_networkx", "pagerank",
        "betweenness_centrality", "shortest_path", "connected_components",
        "community_detect", "search_graph", "search_graph_ppr", "sync_graph_nodes",
    }

    def __init__(self, db: Any) -> None:
        self._db = db

    def add_node(self, node_id: str | None = None, label: str = "", **kwargs: Any) -> str:
        """Add a graph node.

        Mirrors the underlying mixin signature: the first positional
        argument is the node id. Pass only ``label=`` (or nothing) to
        auto-generate an id.

        Example:
            >>> db.graph.add_node(label="Alice")   # generated id
            >>> db.graph.add_node("n1", label="Alice")
        """
        return self._db.add_node(node_id or str(uuid.uuid4()), label=label, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in self._METHODS:
            return getattr(self._db, name)
        raise AttributeError(name)


class AnalyticsAPI:
    """Namespaced OLAP operations exposed as db.olap.*."""

    def __init__(self, db: Any) -> None:
        self._db = db

    def query(self, sql: str) -> list[dict]:
        self._db._auto_register_duckdb_tables()
        return self._db.analytics(sql)

    def register_table(self, table: str) -> bool:
        return self._db.register_duckdb_table(table)

    def unregister_table(self, table: str) -> bool:
        return self._db.unregister_duckdb_table(table)

    def sync_table(self, table: str) -> None:
        self._db.sync_duckdb_table(table)
