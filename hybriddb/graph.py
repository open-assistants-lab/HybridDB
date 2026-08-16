from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import struct
import tempfile
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hybriddb.embedding import EMBEDDING_DIM
from hybriddb.types import Column, SearchMode
from hybriddb.utils import (
    CHROMA_BATCH,
    JOURNAL_CAP,
    RRF_K,
    _CHROMA_INDEX_MAX_ELEMENTS,
    _CHROMA_INDEX_MAX_M0,
    _CHROMA_INDEX_WARN_FACTOR,
    _CHROMA_REBUILD_BATCH,
    _SKIP_SEARCH_COLUMNS,
    _SYSTEM_TABLES,
    _coerce_search_mode,
    _column_spec,
    _is_safe_identifier,
    _now_iso,
    _sanitize_fts_query,
    _validate_identifier,
    _validate_order_by,
)

logger = logging.getLogger("hybriddb")

class GraphMixin:
    def _init_graph_tables(self) -> None:
        with self._connect() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS _graph_nodes (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL DEFAULT '',
                    type TEXT NOT NULL DEFAULT 'node',
                    domain TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.5,
                    source TEXT DEFAULT 'inferred',
                    properties JSON DEFAULT '{}',
                    embedding_model TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS _graph_edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    type TEXT NOT NULL DEFAULT 'relates_to',
                    weight REAL DEFAULT 1.0,
                    properties JSON DEFAULT '{}',
                    valid_from TEXT,
                    valid_until TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES _graph_nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES _graph_nodes(id) ON DELETE CASCADE
                )
            """)
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_edges_unique "
                "ON _graph_edges(source_id, target_id, type)"
            )
            for col, ref in (("source", "source_id"), ("target", "target_id"), ("type", "type")):
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_graph_edges_{col} "
                    f"ON _graph_edges({ref})"
                )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_nodes_type "
                "ON _graph_nodes(type)"
            )
            for col in ("domain", "confidence"):
                try:
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_graph_nodes_{col} "
                        f"ON _graph_nodes({col})"
                    )
                except sqlite3.OperationalError:
                    pass
            cur.execute("""
                CREATE TABLE IF NOT EXISTS _graph_sync (
                    table_name TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    id_column TEXT NOT NULL DEFAULT 'id',
                    label_template TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS _edge_rules (
                    source_table TEXT NOT NULL,
                    target_table TEXT NOT NULL,
                    target_match TEXT NOT NULL,
                    source_column TEXT,
                    target_column TEXT,
                    edge_type TEXT NOT NULL,
                    PRIMARY KEY (source_table, target_table, edge_type)
                )
            """)
            existing_cols = {
                row["name"]
                for row in cur.execute("PRAGMA table_info(_edge_rules)").fetchall()
            }
            if "source_column" not in existing_cols:
                cur.execute("ALTER TABLE _edge_rules ADD COLUMN source_column TEXT")
            if "target_column" not in existing_cols:
                cur.execute("ALTER TABLE _edge_rules ADD COLUMN target_column TEXT")

    def _invalidate_nx_cache(self) -> None:
        self._nx_cache["dirty"] = True

    def register_entity_node(
        self, table_name: str, type: str = "entity",
        id_column: str = "id", label_template: str = "",
    ) -> bool:
        _validate_identifier(table_name, "table")
        _validate_identifier(id_column, "column")
        meta = self._table_meta(table_name)
        if not meta:
            return False
        tmpl = label_template or f"{table_name}: {{{id_column}}}"
        with self._connect() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO _graph_sync "
                "(table_name, node_type, id_column, label_template) VALUES (?, ?, ?, ?)",
                (table_name, type, id_column, tmpl),
            )
        return True

    def register_edge_rule(
        self, source_table: str, target_table: str,
        target_match: str | None = None, edge_type: str = "relates_to",
        source_column: str | None = None, target_column: str | None = None,
    ) -> bool:
        _validate_identifier(source_table, "source table")
        _validate_identifier(target_table, "target table")
        if bool(source_column) != bool(target_column):
            raise ValueError("source_column and target_column must be provided together")
        if source_column is None and target_column is None:
            if target_match is None:
                raise ValueError("target_match or source_column/target_column is required")
            source_column = target_match
            target_column = "id"
        _validate_identifier(source_column, "source column")
        _validate_identifier(target_column, "target column")
        with self._connect() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO _edge_rules "
                "(source_table, target_table, target_match, source_column, target_column, edge_type) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (source_table, target_table, target_match or source_column, source_column, target_column, edge_type),
            )
        return True

    def _auto_sync_graph_nodes(self) -> dict:
        result = {"nodes_created": 0}
        rules = self.raw_query("SELECT * FROM _graph_sync")
        for rule in rules:
            table = rule["table_name"]
            if table in _SYSTEM_TABLES:
                continue
            id_col = rule["id_column"]
            if not _is_safe_identifier(table) or not _is_safe_identifier(id_col):
                continue
            tmpl = rule["label_template"]
            ntype = rule["node_type"]
            rows = self.raw_query(f"SELECT {id_col} FROM {table}")
            for row in rows:
                rid = str(row[id_col])
                label = tmpl.replace(f"{{{id_col}}}", rid)
                existing = self.get_node(rid)
                if existing and existing.get("type") == ntype:
                    continue
                self.add_node(rid, label=label, type=ntype, source="auto_sync")
                result["nodes_created"] += 1
        return result

    def _auto_sync_graph_edges(self) -> dict:
        result = {"edges_created": 0}
        rules = self.raw_query("SELECT * FROM _edge_rules")
        for rule in rules:
            src_table = rule["source_table"]
            tgt_table = rule["target_table"]
            src_col = rule.get("source_column") or rule["target_match"]
            tgt_col = rule.get("target_column") or rule["target_match"]
            if (not _is_safe_identifier(src_table) or not _is_safe_identifier(tgt_table)
                    or not _is_safe_identifier(src_col) or not _is_safe_identifier(tgt_col)):
                continue
            etype = rule["edge_type"]
            pairs = self.raw_query(
                f"SELECT s.id as sid, t.id as tid FROM {src_table} s "
                f"JOIN {tgt_table} t ON s.{src_col} = t.{tgt_col}"
            )
            for pair in pairs:
                self.add_edge(None, str(pair["sid"]), str(pair["tid"]), type=etype)
                result["edges_created"] += 1
        return result

    def add_node(
        self, node_id: str, label: str = "", type: str = "node",
        domain: str = "", confidence: float = 0.5, source: str = "inferred",
        properties: dict | None = None,
    ) -> str:
        self._invalidate_nx_cache()
        now = _now_iso()
        props_json = json.dumps(properties or {})
        with self._connect() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO _graph_nodes "
                "(id, label, type, domain, confidence, source, properties, "
                "embedding_model, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (node_id, label, type, domain, confidence, source, props_json,
                 self._embedding_model_name, now, now),
            )
        return node_id

    def add_nodes(self, nodes: list[dict]) -> list[str]:
        self._invalidate_nx_cache()
        ids: list[str] = []
        with self._connect() as cur:
            for n in nodes:
                node_id = n["id"]
                now = _now_iso()
                cur.execute(
                    "INSERT OR REPLACE INTO _graph_nodes "
                    "(id, label, type, domain, confidence, source, properties, "
                    "embedding_model, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (node_id, n.get("label", ""), n.get("type", "node"), n.get("domain", ""),
                     n.get("confidence", 0.5), n.get("source", "inferred"),
                     json.dumps(n.get("properties", {})), self._embedding_model_name, now, now),
                )
                ids.append(node_id)
        return ids

    def get_node(self, node_id: str) -> dict | None:
        with self._connect() as cur:
            cur.execute("SELECT * FROM _graph_nodes WHERE id = ?", (node_id,))
            row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        d["properties"] = json.loads(d.get("properties", "{}"))
        return d

    def update_node(self, node_id: str, data: dict) -> bool:
        self._invalidate_nx_cache()
        node = self.get_node(node_id)
        if not node:
            return False
        updated: dict[str, Any] = {"updated_at": _now_iso()}
        for field in ("label", "type", "domain", "confidence", "source"):
            if field in data:
                updated[field] = data[field]
        if "properties" in data:
            merged = dict(node["properties"])
            merged.update(data["properties"])
            updated["properties"] = json.dumps(merged)
        set_clause = ", ".join(f"{k} = ?" for k in updated)
        with self._connect() as cur:
            cur.execute(f"UPDATE _graph_nodes SET {set_clause} WHERE id = ?", list(updated.values()) + [node_id])
        return True

    def delete_node(self, node_id: str) -> bool:
        self._invalidate_nx_cache()
        with self._connect() as cur:
            cur.execute("DELETE FROM _graph_nodes WHERE id = ?", (node_id,))
            if cur.rowcount == 0:
                return False
        return True

    def list_nodes(
        self, type: str | None = None, domain: str | None = None,
        min_confidence: float = 0.0, limit: int = 100,
    ) -> list[dict]:
        sql = "SELECT * FROM _graph_nodes WHERE 1=1"
        params: list[Any] = []
        if type:
            sql += " AND type = ?"
            params.append(type)
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        if min_confidence > 0:
            sql += " AND confidence >= ?"
            params.append(min_confidence)
        sql += " ORDER BY created_at DESC"
        if limit > 0:
            sql += f" LIMIT {limit}"
        with self._connect() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["properties"] = json.loads(d.get("properties", "{}"))
            result.append(d)
        return result

    def add_edge(
        self, edge_id: str | None, source_id: str, target_id: str,
        type: str = "relates_to", weight: float = 1.0,
        properties: dict | None = None, valid_until: str | None = None,
        edge_type: str | None = None,
    ) -> str:
        self._invalidate_nx_cache()
        if edge_type is not None:
            type = edge_type
        if edge_id is None:
            edge_id = uuid.uuid4().hex[:16]
        now = _now_iso()
        props_json = json.dumps(properties or {})
        with self._connect() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO _graph_edges "
                "(id, source_id, target_id, type, weight, properties, valid_until, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (edge_id, source_id, target_id, type, weight, props_json, valid_until, now),
            )
        return edge_id

    def get_neighbors(
        self, node_id: str, direction: str = "both", type: str | None = None,
    ) -> list[dict]:
        """Alias for neighbors()."""
        return self.neighbors(node_id, direction=direction, type=type)

    def add_edges(self, edges: list[dict]) -> list[str]:
        self._invalidate_nx_cache()
        ids: list[str] = []
        with self._connect() as cur:
            for e in edges:
                edge_id = e.get("id") or uuid.uuid4().hex[:16]
                now = _now_iso()
                cur.execute(
                    "INSERT OR REPLACE INTO _graph_edges "
                    "(id, source_id, target_id, type, weight, properties, valid_until, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (edge_id, e["source_id"], e["target_id"], e.get("type", "relates_to"),
                     e.get("weight", 1.0), json.dumps(e.get("properties", {})),
                     e.get("valid_until"), now),
                )
                ids.append(edge_id)
        return ids

    def get_edge(self, edge_id: str) -> dict | None:
        with self._connect() as cur:
            cur.execute("SELECT * FROM _graph_edges WHERE id = ?", (edge_id,))
            row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        d["properties"] = json.loads(d.get("properties", "{}"))
        return d

    def update_edge(self, edge_id: str, data: dict) -> bool:
        self._invalidate_nx_cache()
        edge = self.get_edge(edge_id)
        if not edge:
            return False
        allowed = {"type", "weight", "properties", "source_id", "target_id", "valid_from", "valid_until"}
        filtered = {k: v for k, v in data.items() if k in allowed}
        if not filtered:
            return False
        if "properties" in filtered:
            merged = dict(edge["properties"])
            merged.update(filtered["properties"])
            filtered["properties"] = json.dumps(merged)
        set_clause = ", ".join(f"{k} = ?" for k in filtered)
        with self._connect() as cur:
            cur.execute(f"UPDATE _graph_edges SET {set_clause} WHERE id = ?", list(filtered.values()) + [edge_id])
        return True

    def delete_edge(self, edge_id: str) -> bool:
        self._invalidate_nx_cache()
        with self._connect() as cur:
            cur.execute("DELETE FROM _graph_edges WHERE id = ?", (edge_id,))
            if cur.rowcount == 0:
                return False
        return True

    def get_edges(
        self, source_id: str | None = None, target_id: str | None = None,
        type: str | None = None, limit: int = 100,
    ) -> list[dict]:
        sql = "SELECT * FROM _graph_edges WHERE 1=1"
        params: list[Any] = []
        if source_id:
            sql += " AND source_id = ?"
            params.append(source_id)
        if target_id:
            sql += " AND target_id = ?"
            params.append(target_id)
        if type:
            sql += " AND type = ?"
            params.append(type)
        sql += " ORDER BY created_at DESC"
        if limit > 0:
            sql += f" LIMIT {limit}"
        with self._connect() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["properties"] = json.loads(d.get("properties", "{}"))
            result.append(d)
        return result

    def neighbors(
        self, node_id: str, direction: str = "both", type: str | None = None,
    ) -> list[dict]:
        params: list[Any] = [node_id, node_id]
        if direction == "out":
            edge_clause = "e.source_id = ?"
            params.append(node_id)
        elif direction == "in":
            edge_clause = "e.target_id = ?"
            params.append(node_id)
        else:
            edge_clause = "(e.source_id = ? OR e.target_id = ?)"
            params.append(node_id)
            params.append(node_id)
        if type:
            edge_clause += " AND e.type = ?"
            params.append(type)

        sql = f"""
            SELECT e.id as edge_id, e.type as edge_type, e.weight,
                   e.properties as edge_properties, e.source_id, e.target_id,
                   n.id as node_id, n.label as node_label,
                   n.type as node_type, n.properties as node_properties
            FROM _graph_edges e
            JOIN _graph_nodes n ON (
                CASE WHEN e.source_id = ? AND e.target_id = n.id THEN 1
                     WHEN e.target_id = ? AND e.source_id = n.id THEN 1
                     ELSE 0 END = 1
            )
            WHERE {edge_clause}
            ORDER BY e.weight DESC
        """
        with self._connect() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        result = []
        seen = set()
        for r in rows:
            d = dict(r)
            nid = d["node_id"]
            if nid in seen:
                continue
            seen.add(nid)
            result.append({
                "node": {
                    "id": nid, "label": d["node_label"], "type": d["node_type"],
                    "properties": json.loads(d.get("node_properties", "{}")),
                },
                "edge": {
                    "id": d["edge_id"], "type": d["edge_type"], "weight": d["weight"],
                    "source_id": d["source_id"], "target_id": d["target_id"],
                    "properties": json.loads(d.get("edge_properties", "{}")),
                },
            })
        return result

    def traverse(
        self, start_id: str, max_depth: int = 3, direction: str = "out",
        type: str | None = None, max_cost: float = 3.0,
    ) -> list[dict]:
        if max_depth < 1 or max_depth > 10:
            raise ValueError("max_depth must be between 1 and 10")
        if direction not in ("in", "out", "both"):
            raise ValueError("direction must be 'in', 'out', or 'both'")

        base_params = (start_id, start_id, start_id)
        type_filter = ""
        if type:
            type_filter = " AND e.type = ?"
            base_params = base_params + (type,)

        sql = f"""
            WITH RECURSIVE graph_path(node_id, depth, path, cum_cost) AS (
                SELECT ?, 0, ?, 0.0
                UNION ALL
                SELECT
                    CASE WHEN '{direction}' = 'in' THEN e.source_id
                         WHEN '{direction}' = 'out' THEN e.target_id
                         ELSE CASE WHEN e.source_id = gp.node_id THEN e.target_id
                                   ELSE e.source_id END
                    END,
                    gp.depth + 1,
                    gp.path || '>' ||
                        CASE WHEN '{direction}' = 'in' THEN e.source_id
                             WHEN '{direction}' = 'out' THEN e.target_id
                             ELSE CASE WHEN e.source_id = gp.node_id THEN e.target_id
                                       ELSE e.source_id END
                        END,
                    gp.cum_cost + (1.0 - e.weight)
                FROM _graph_edges e
                JOIN graph_path gp ON (
                    CASE WHEN '{direction}' = 'in' THEN e.target_id = gp.node_id
                         WHEN '{direction}' = 'out' THEN e.source_id = gp.node_id
                         ELSE (e.source_id = gp.node_id OR e.target_id = gp.node_id)
                    END
                )
                {type_filter}
                WHERE gp.depth < {max_depth}
                  AND gp.cum_cost + (1.0 - e.weight) <= {max_cost}
            )
            SELECT DISTINCT node_id, MIN(depth) as depth, MIN(path) as path,
                   MIN(cum_cost) as cum_cost
            FROM graph_path
            WHERE node_id != ?
            GROUP BY node_id
            ORDER BY depth, cum_cost
        """
        return self.raw_query(sql, base_params)

    def decay_edges(self) -> int:
        self._invalidate_nx_cache()
        now = _now_iso()
        expired = self.raw_query(
            "SELECT id, weight FROM _graph_edges "
            "WHERE valid_until IS NOT NULL AND valid_until < ?",
            (now,),
        )
        dec = 0
        for e in expired:
            new_weight = max(e["weight"] - 0.15, 0.05)
            if new_weight <= 0.05:
                self.delete_edge(e["id"])
            else:
                self.update_edge(e["id"], {"weight": new_weight})
            dec += 1
        return dec

    def to_networkx(self, directed: bool = True, use_cache: bool = True):
        import networkx as nx

        if (use_cache and not self._nx_cache["dirty"]
                and self._nx_cache["graph"] is not None
                and self._nx_cache.get("directed") == directed):
            return self._nx_cache["graph"]

        g = nx.DiGraph() if directed else nx.Graph()
        nodes = self.raw_query(
            "SELECT id, label, type, domain, confidence, source, properties FROM _graph_nodes"
        )
        for n in nodes:
            props = (json.loads(n.get("properties", "{}"))
                     if isinstance(n.get("properties"), str)
                     else n.get("properties", {}))
            reserved = {"label", "type", "domain", "confidence", "source"}
            node_attrs = {k: v for k, v in props.items() if k not in reserved}
            g.add_node(n["id"], label=n["label"], type=n["type"], domain=n.get("domain", ""),
                       confidence=n.get("confidence", 0.5), source=n.get("source", "inferred"), **node_attrs)
        edges = self.raw_query(
            "SELECT id, source_id, target_id, type, weight, properties, valid_until FROM _graph_edges"
        )
        for e in edges:
            props = (json.loads(e.get("properties", "{}"))
                     if isinstance(e.get("properties"), str)
                     else e.get("properties", {}))
            reserved = {"id", "type", "weight", "valid_until"}
            edge_attrs = {k: v for k, v in props.items() if k not in reserved}
            g.add_edge(e["source_id"], e["target_id"], id=e["id"], type=e["type"],
                       weight=e["weight"], valid_until=e.get("valid_until"), **edge_attrs)
        if use_cache:
            self._nx_cache["graph"] = g
            self._nx_cache["dirty"] = False
            self._nx_cache["directed"] = directed
        return g

    def sync_graph_nodes(self) -> dict:
        """Sync registered table rows into graph nodes. Public API."""
        return self._auto_sync_graph_nodes()

    def pagerank(
        self,
        personalization: dict[str, float] | None = None,
        alpha: float = 0.85,
    ) -> dict[str, float]:
        import networkx as nx

        G = self.to_networkx(directed=True)
        return nx.pagerank(
            G, alpha=alpha, weight="weight",
            personalization=personalization,
        )

    def betweenness_centrality(self) -> dict[str, float]:
        import networkx as nx
        return nx.betweenness_centrality(self.to_networkx(directed=False), weight="weight")

    def shortest_path(self, source: str, target: str) -> list[str] | None:
        import networkx as nx
        try:
            return nx.shortest_path(self.to_networkx(directed=True), source=source, target=target, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def connected_components(self) -> list[set[str]]:
        import networkx as nx
        return [set(c) for c in nx.connected_components(self.to_networkx(directed=False))]

    def community_detect(self) -> list[set[str]]:
        import networkx as nx
        partition = nx.community.louvain_communities(self.to_networkx(directed=False), weight="weight")
        return [set(c) for c in partition]

    def _find_seed_nodes(
        self, query: str, limit: int = 10, min_similarity: float = 0.0,
    ) -> dict[str, dict]:
        """Vector search across registered graph-synced tables.

        Returns:
            {node_id: {"table": str, "similarity": float}} for matching nodes.
            node_id is the table's primary key value, not the ChromaDB rowid.
            Nodes with similarity below min_similarity are filtered out.
        """
        self._auto_sync_graph_nodes()
        self._auto_sync_graph_edges()
        registered = self.raw_query("SELECT table_name FROM _graph_sync")
        searchable_tables = [r["table_name"] for r in registered]
        found_nodes: dict[str, dict] = {}
        embedding: list[float] | None = None
        for table in searchable_tables:
            pk_col = self._get_pk_column(table)
            for col_name in self._get_longtext_columns(table):
                try:
                    if embedding is None:
                        embedding = self._get_embedding(query)
                    collection = self._get_collection(f"{table}_{col_name}")
                    if collection is None:
                        continue
                    vec_results = collection.query(
                        query_embeddings=[embedding], n_results=limit, include=["distances"],
                    )
                    chroma_ids = vec_results.get("ids", [[]])[0]
                    if not chroma_ids:
                        continue
                    rowid_list = ",".join("?" for _ in chroma_ids)
                    rows = self.raw_query(
                        f"SELECT rowid AS _rid, {pk_col} FROM {table} WHERE rowid IN ({rowid_list})",
                        tuple(chroma_ids),
                    )
                    rowid_to_pk = {str(r["_rid"]): str(r[pk_col]) for r in rows}
                    for i, doc_id in enumerate(chroma_ids):
                        distance = vec_results["distances"][0][i] if "distances" in vec_results else 0
                        pk_value = rowid_to_pk.get(str(doc_id), str(doc_id))
                        similarity = max(0.0, 1.0 - distance)
                        if similarity < min_similarity:
                            continue
                        found_nodes[pk_value] = {"table": table, "similarity": similarity}
                except Exception:
                    continue
        return found_nodes

    def search_graph(self, query: str, hop_expansion: int = 2, limit: int = 10) -> list[dict]:
        found_nodes = self._find_seed_nodes(query, limit)
        result_list = []
        for node_id, meta in found_nodes.items():
            entry = {"node_id": node_id, "similarity": meta["similarity"], "source_table": meta["table"]}
            if hop_expansion > 0:
                try:
                    entry["neighbors"] = self.neighbors(node_id, direction="both")[:hop_expansion * 2]
                except Exception:
                    entry["neighbors"] = []
            result_list.append(entry)
        result_list.sort(key=lambda x: x["similarity"], reverse=True)
        return result_list[:limit]

    def search_graph_ppr(
        self,
        query: str,
        hop_expansion: int = 2,
        limit: int = 10,
        alpha: float = 0.15,
        min_similarity: float = 0.0,
        k_seeds: int | None = None,
    ) -> list[dict]:
        import networkx as nx

        seed_limit = k_seeds if k_seeds is not None else limit
        seed_nodes = self._find_seed_nodes(query, seed_limit, min_similarity=min_similarity)
        if not seed_nodes:
            return []

        subgraph_node_ids: set[str] = set(seed_nodes.keys())
        node_depth: dict[str, int] = {nid: 0 for nid in seed_nodes}

        if hop_expansion > 0:
            for node_id in seed_nodes:
                try:
                    neighbors = self.traverse(
                        node_id, max_depth=hop_expansion, direction="both"
                    )
                except Exception:
                    neighbors = []
                for n in neighbors:
                    nid = n.get("node_id", "")
                    if not nid:
                        continue
                    subgraph_node_ids.add(nid)
                    depth = n.get("depth", hop_expansion)
                    if nid not in node_depth or depth < node_depth[nid]:
                        node_depth[nid] = depth

        G_full = self.to_networkx(directed=True)
        G_sub = G_full.subgraph(subgraph_node_ids).copy()

        if len(G_sub) == 0:
            return []

        personalization = {
            nid: max(meta["similarity"], 1e-6)
            for nid, meta in seed_nodes.items()
            if nid in G_sub
        }
        if not personalization:
            personalization = None

        scores = nx.pagerank(
            G_sub, alpha=alpha, weight="weight",
            personalization=personalization,
        )

        results = [
            {
                "node_id": nid,
                "ppr_score": score,
                "similarity": seed_nodes.get(nid, {}).get("similarity", 0.0),
                "source_table": seed_nodes.get(nid, {}).get("table", ""),
                "depth": node_depth.get(nid, hop_expansion),
            }
            for nid, score in scores.items()
        ]
        results.sort(key=lambda x: x["ppr_score"], reverse=True)
        return results[:limit]

