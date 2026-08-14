# HybridDB Graph Retrieval: Personalized PageRank

## Motivation

HybridDB already has graph infrastructure — nodes, typed weighted edges, traversal, and standard PageRank. But the current `pagerank()` computes **global** importance over the entire graph. For retrieval, we need **query-relevant** importance: given a set of seed nodes (from keyword/vector search), which other nodes are most relevant to them?

This is **Personalized PageRank (PPR)** — PageRank with a personalization vector that biases the random walk toward seed nodes. PPR is the standard graph-based retrieval algorithm and maps directly to the spreading-activation / hippocampal-replay model of memory retrieval.

Use case: [CoreMem](https://github.com/open-assistants-lab/CoreMem) builds a message graph (temporal, session, topic edges) and needs PPR to rank messages by graph proximity to search seeds. This is a general retrieval pattern — any HybridDB user with graph edges benefits from PPR-based ranking.

## Changes

### 1. `pagerank()` — add personalization and alpha

**Current:**

```python
def pagerank(self) -> dict[str, float]:
    import networkx as nx
    return nx.pagerank(self.to_networkx(directed=True), weight="weight")
```

**Proposed:**

```python
def pagerank(
    self,
    personalization: dict[str, float] | None = None,
    alpha: float = 0.85,
) -> dict[str, float]:
    """Compute PageRank over the graph.

    Args:
        personalization: Node ID → weight. If provided, runs Personalized
            PageRank (PPR): the random walk teleports to these nodes
            proportional to their weights instead of uniformly at random.
            Node IDs must exist in the graph. Weights are normalized
            internally. If None, runs standard PageRank (uniform teleport).
        alpha: Damping factor (1 - teleport probability). Higher alpha =
            more spreading through the graph. Lower alpha = more
            concentrated near seeds. Default 0.85 (standard PageRank).
            For PPR retrieval, 0.15–0.30 is typical (concentrate near seeds).

    Returns:
        {node_id: pagerank_score} for all nodes in the graph.
    """
    import networkx as nx
    G = self.to_networkx(directed=True)
    return nx.pagerank(
        G, alpha=alpha, weight="weight",
        personalization=personalization,
    )
```

**Backward compatibility:** fully compatible. `pagerank()` with no args returns the same result as before (standard PageRank, alpha=0.85, uniform teleport).

### 2. `search_graph_ppr()` — graph-aware retrieval

The current `search_graph()` does vector search → sort by similarity → attach neighbors. It finds seed nodes but doesn't rank by graph structure. The new method adds PPR ranking:

```python
def search_graph_ppr(
    self,
    query: str,
    hop_expansion: int = 2,
    limit: int = 10,
    alpha: float = 0.15,
    min_similarity: float = 0.1,
) -> list[dict]:
    """Graph-aware semantic retrieval via Personalized PageRank.

    Pipeline:
        1. Vector search → seed nodes with similarity scores (filtered by min_similarity)
        2. Expand subgraph: traverse hop_expansion hops from each seed
        3. Run PPR on the subgraph with seeds as personalization
        4. Return nodes ranked by PPR score

    This finds nodes that are semantically relevant to the query (seeds)
    AND structurally close to those seeds (via graph edges). Nodes that
    don't match the query directly but are connected to matching nodes
    can surface through the graph walk.

    Args:
        query: Search query text.
        hop_expansion: How many hops to expand from each seed node to
            build the subgraph. 0 = seeds only (no graph expansion).
            1–2 is typical. Higher values = larger subgraph, more
            spreading, slower.
        limit: Maximum number of results to return.
        alpha: PPR damping factor. Lower = concentrate near seeds.
            Default 0.15 (retrieval-tuned, not standard PageRank's 0.85).
        min_similarity: Minimum vector similarity for seed nodes. Filters
            out near-zero-similarity results that ChromaDB returns but
            are not actual matches. Default 0.1.

    Returns:
        List of {node_id, ppr_score, similarity, source_table, depth}
        sorted by PPR score descending.
    """
```

**Why a new method instead of modifying `search_graph`:**
- `search_graph` returns `{node_id, similarity, source_table, neighbors}` — different shape
- Changing the return shape would break existing callers
- The two methods have different semantics: `search_graph` = find + attach neighbors; `search_graph_ppr` = find + rank by graph structure

## Design Decisions

### Why PPR instead of standard PageRank?

Standard PageRank ranks nodes by global importance — the same ranking regardless of query. It's useful for finding "the most central nodes in the graph" but useless for retrieval where relevance is query-dependent.

PPR biases the walk toward seed nodes. A node that's globally unimportant but close to the query seeds gets a high PPR score. This is the right behavior for retrieval.

### Why `alpha=0.15` default for `search_graph_ppr`?

Standard PageRank uses `alpha=0.85` (85% follow edges, 15% teleport). For retrieval, we want the walk to stay near the seeds — too much spreading dilutes relevance. `alpha=0.15` means 15% follow edges, 85% teleport back to seeds. This concentrates the score near seeds while still allowing the graph to pull in structurally close neighbors.

The `pagerank()` method keeps `alpha=0.85` as default (standard PageRank), while `search_graph_ppr` defaults to `alpha=0.15` (retrieval-tuned). Different use cases, different defaults.

### Why extract a subgraph before PPR?

Running PPR on the full graph is O(n) per iteration and includes all nodes. For a memory graph with thousands of nodes, most are irrelevant to a given query. By extracting a subgraph (seeds + `hop_expansion` hops) first, we:
- Reduce computation to the relevant neighborhood
- Avoid noise from irrelevant nodes diluting scores
- Allow multiple queries in parallel (each query has its own subgraph)

### Why not use NetworkX subgraph directly?

NetworkX's `G.subgraph(nodes)` returns a view, but `nx.pagerank` on a subgraph view can behave unexpectedly with dangling nodes (nodes with no outgoing edges). The implementation should build a clean subgraph and handle dangling nodes (distribute their score uniformly, as NetworkX does by default).

## Implementation Notes

### Subgraph extraction

The seed-finding logic in `search_graph` (lines 614–629 of `graph.py`) iterates registered tables and queries ChromaDB collections directly. `search_graph_ppr` needs the same logic. To avoid duplication, refactor `search_graph` to extract a `_find_seed_nodes(query, limit)` helper that both methods share. This is an internal refactor — `search_graph`'s public API stays the same.

```python
# 0. Ensure graph nodes exist for registered tables
self._auto_sync_graph_nodes()

# 1. Vector search → seeds (reuse search_graph's seed-finding logic)
#    Refactor: extract _find_seed_nodes(query, limit) from search_graph
#    so both search_graph and search_graph_ppr share it.
seed_nodes = self._find_seed_nodes(query, limit)
if not seed_nodes:
    return []

# 2. Expand subgraph via traverse()
subgraph_node_ids = set(seed_nodes.keys())
node_depth: dict[str, int] = {nid: 0 for nid in seed_nodes}
for node_id in seed_nodes:
    neighbors = self.traverse(
        node_id, max_depth=hop_expansion, direction="both"
    )
    for n in neighbors:
        nid = n["node_id"]
        subgraph_node_ids.add(nid)
        depth = n.get("depth", hop_expansion)
        if nid not in node_depth or depth < node_depth[nid]:
            node_depth[nid] = depth

# 3. Build NetworkX subgraph
G_full = self.to_networkx(directed=True)
G_sub = G_full.subgraph(subgraph_node_ids).copy()

# 4. PPR with seeds as personalization
personalization = {
    nid: score for nid, score in seed_nodes.items()
    if nid in G_sub
}
if not personalization:
    # No seeds in graph → fall back to uniform (standard PageRank on subgraph)
    personalization = None
scores = nx.pagerank(G_sub, alpha=alpha, weight="weight",
                     personalization=personalization)

# 5. Format results
results = [
    {
        "node_id": nid,
        "ppr_score": score,
        "similarity": seed_nodes.get(nid, 0.0),
        "source_table": seed_nodes.get(nid, {}).get("table", ""),
        "depth": node_depth.get(nid, hop_expansion),
    }
    for nid, score in scores.items()
]
results.sort(key=lambda x: x["ppr_score"], reverse=True)
return results[:limit]
```

### Edge cases

- **No seeds found:** return `[]`
- **No edges in graph:** PPR degenerates to seed similarity scores (no spreading). Return seeds sorted by similarity.
- **Dangling nodes (no outgoing edges):** NetworkX handles this by default (redistributes dangling mass uniformly).
- **Seed node not in graph (no `_graph_nodes` entry):** skip it. Vector search returns table rows, but graph nodes are separate. Only seeds that have corresponding graph nodes participate in PPR.
- **Empty personalization after filtering:** fall back to uniform personalization (standard PageRank on subgraph).

## Test Plan

### `pagerank()` tests

```python
def test_pagerank_with_personalization(self, db):
    db.add_node("a"), db.add_node("b"), db.add_node("c"), db.add_node("d")
    db.add_edge(None, "a", "b", weight=1.0)
    db.add_edge(None, "b", "c", weight=1.0)
    db.add_edge(None, "c", "a", weight=1.0)
    db.add_edge(None, "a", "d", weight=0.5)
    # Standard PageRank
    pr = db.pagerank()
    assert "a" in pr and "d" in pr
    # Personalized — bias toward "c"
    ppr = db.pagerank(personalization={"c": 1.0}, alpha=0.15)
    # "c" and its neighbors should score higher than "d"
    assert ppr["c"] > pr["c"]

def test_pagerank_alpha(self, db):
    db.add_node("a"), db.add_node("b"), db.add_node("c")
    db.add_edge(None, "a", "b", weight=1.0)
    db.add_edge(None, "b", "c", weight=1.0)
    pr_high_alpha = db.pagerank(alpha=0.85)
    pr_low_alpha = db.pagerank(alpha=0.15)
    # Lower alpha = more teleportation = scores flatten toward uniform
    # Higher alpha = more edge-following = high-degree nodes dominate
    # Check that scores are valid and sum is preserved
    assert abs(sum(pr_high_alpha.values()) - 1.0) < 0.01
    assert abs(sum(pr_low_alpha.values()) - 1.0) < 0.01
    # With alpha=0.15, scores are more uniform (less variance)
    import statistics
    assert statistics.pstdev(pr_low_alpha.values()) < statistics.pstdev(pr_high_alpha.values())

def test_pagerank_backward_compat(self, db):
    db.add_node("n1"), db.add_node("n2")
    db.add_edge(None, "n1", "n2", weight=0.9)
    pr = db.pagerank()
    # Same as before — no args, standard PageRank
    assert "n1" in pr and "n2" in pr
```

### `search_graph_ppr()` tests

```python
def test_search_graph_ppr_basic(self, db):
    db.create_table("docs", {"id": "TEXT PRIMARY KEY", "body": "LONGTEXT"})
    db.register_entity_node("docs", type="doc")
    db.insert("docs", {"id": "d1", "body": "machine learning basics"})
    db.insert("docs", {"id": "d2", "body": "deep neural networks"})
    db.insert("docs", {"id": "d3", "body": "cooking pasta recipes"})
    # Connect d1 → d2 (related topic)
    db.add_edge(None, "d1", "d2", type="related", weight=1.0)
    results = db.search_graph_ppr("machine learning", hop_expansion=2)
    assert len(results) > 0
    # d1 matches the query, d2 is connected → both should appear
    node_ids = [r["node_id"] for r in results]
    assert "d1" in node_ids

def test_search_graph_ppr_no_edges(self, db):
    db.create_table("docs", {"id": "TEXT PRIMARY KEY", "body": "LONGTEXT"})
    db.register_entity_node("docs", type="doc")
    db.insert("docs", {"id": "d1", "body": "hello world"})
    results = db.search_graph_ppr("hello", hop_expansion=2)
    assert len(results) > 0
    # No edges → returns seeds by similarity
    assert results[0]["node_id"] == "d1"

def test_search_graph_ppr_graph_brings_indirect_match(self, db):
    db.create_table("docs", {"id": "TEXT PRIMARY KEY", "body": "LONGTEXT"})
    db.register_entity_node("docs", type="doc")
    db.insert("docs", {"id": "d1", "body": "python programming tutorial"})
    db.insert("docs", {"id": "d2", "body": "xqz unrelated content zzz"})
    db.insert("docs", {"id": "d3", "body": "cooking italian food"})
    # d1 ↔ d2 connected by graph edge, but d2 has no semantic match to query
    db.add_edge(None, "d1", "d2", type="related", weight=1.0)
    # d3 is disconnected
    # Query "python" → d1 matches, d2 surfaces via graph (not similarity)
    results = db.search_graph_ppr("python", hop_expansion=2)
    node_ids = [r["node_id"] for r in results]
    assert "d1" in node_ids
    assert "d2" in node_ids  # surfaced via graph, not via similarity
    assert "d3" not in node_ids  # disconnected, not surfaced

def test_search_graph_ppr_alpha(self, db):
    db.create_table("docs", {"id": "TEXT PRIMARY KEY", "body": "LONGTEXT"})
    db.register_entity_node("docs", type="doc")
    db.insert("docs", {"id": "d1", "body": "topic a unique phrase"})
    db.insert("docs", {"id": "d2", "body": "xqz intermediate zzz"})
    db.insert("docs", {"id": "d3", "body": "xqz distant zzz far"})
    db.add_edge(None, "d1", "d2", type="rel", weight=1.0)
    db.add_edge(None, "d2", "d3", type="rel", weight=1.0)
    # Low alpha = concentrate near seed (d1), d3 gets very little
    results_low = db.search_graph_ppr("topic a", alpha=0.15)
    # High alpha = more spreading, d3 gets more
    results_high = db.search_graph_ppr("topic a", alpha=0.85)
    # d1 should be top in both
    assert results_low[0]["node_id"] == "d1"
    assert results_high[0]["node_id"] == "d1"
    # With low alpha, d2 (1 hop) should rank higher than d3 (2 hops)
    low_scores = {r["node_id"]: r["ppr_score"] for r in results_low}
    if "d2" in low_scores and "d3" in low_scores:
        assert low_scores["d2"] > low_scores["d3"]
    # With high alpha, the gap between d2 and d3 should shrink
    high_scores = {r["node_id"]: r["ppr_score"] for r in results_high}
    if "d2" in high_scores and "d3" in high_scores:
        gap_low = low_scores["d2"] - low_scores["d3"]
        gap_high = high_scores["d2"] - high_scores["d3"]
        assert gap_high < gap_low
```

## Scope

**In scope:**
- `pagerank(personalization, alpha)` — 5-line change to existing method
- `search_graph_ppr(query, hop_expansion, limit, alpha)` — new method in `GraphMixin`
- Tests for both

**Out of scope:**
- GNN / learned edge weights (future, after PPR validates the approach)
- Edge creation logic (domain-specific, lives in CoreMem not HybridDB)
- Subgraph caching across queries (premature optimization)

## Dependencies

- NetworkX (already a dependency — `pagerank()` and `to_networkx()` use it)
- No new dependencies

## Version

Minor version bump: 0.4.5 → 0.5.0 (new public API method `search_graph_ppr`, backward-compatible change to `pagerank`).