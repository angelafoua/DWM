"""
Phase 6 — Similarity Graph Construction.

WHY ENTITY RESOLUTION BECOMES A GRAPH PROBLEM
==============================================
In entity resolution we ask: which records refer to the same real-world entity?
A pairwise similarity score between record A and record B tells us *something*
about their relationship, but it does not directly give us groups.  By modelling
records as nodes and high-similarity pairs as edges, we transform the problem
into graph partitioning:

  - Each **node** represents one record in the corpus.
  - Each **edge** (A, B) with weight w = cosine_similarity(A, B) represents
    evidence that A and B describe the same entity.
  - Clustering the graph (Phase 7) groups nodes into entity clusters.

HOW SIMILARITY EDGES REPRESENT ENTITY RELATIONSHIPS
=====================================================
An edge A–B exists iff cosine_similarity(A, B) ≥ threshold.  This encodes:
  - direct evidence (A and B share enough blocking tokens to be candidates).
  - The edge weight quantifies *how* similar they are, enabling downstream
    quality filtering (prune edges below a stricter threshold).

HOW TRANSITIVITY EMERGES NATURALLY
====================================
If A–B and B–C are both edges (A similar to B, B similar to C), then A and C
are in the same connected component even if A and C share no direct edge.  This
is the *transitive closure* property of graph connectivity — the graph naturally
captures indirect evidence through chains of similar records.  Phase 8 audits
these chains for pathological cases where the chain propagates dissimilarity.

DEDUPLICATION
=============
ANN retrieval is asymmetric: record i may list j as a neighbour and record j
may separately list i.  Both produce the same undirected edge {i, j}.  We use
networkx's add_edge with a deduplication check (or graph.has_edge) to ensure
at most one edge per pair and keep the higher cosine score when duplicates occur.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx

from ..similarity.cosine import RefinementResult
from ..ann.retrieval import NeighborResult
from ..utils.config import GraphConfig
from ..utils.logging_utils import get_logger
from ..utils.profiling import profile_block

logger = get_logger(__name__)


@dataclass
class GraphStats:
    """Summary statistics for the constructed similarity graph."""

    n_nodes: int = 0
    n_edges: int = 0
    avg_degree: float = 0.0
    n_components: int = 0
    component_sizes: List[int] = field(default_factory=list)
    largest_component: int = 0
    density: float = 0.0

    def __str__(self) -> str:
        return (
            f"Graph | nodes={self.n_nodes} | edges={self.n_edges} | "
            f"avg_degree={self.avg_degree:.2f} | "
            f"components={self.n_components} | "
            f"largest_component={self.largest_component} | "
            f"density={self.density:.6f}"
        )


def build_similarity_graph(
    refinement_result: RefinementResult,
    record_ids: List[str],
    config: GraphConfig | None = None,
) -> tuple[nx.Graph, GraphStats]:
    """Phase 6 entry point: construct a weighted similarity graph.

    Parameters
    ----------
    refinement_result : RefinementResult — exact-cosine edges from Phase 5.
    record_ids        : list[str]        — maps integer indices to record IDs.
    config            : GraphConfig | None

    Returns
    -------
    G      : nx.Graph — nodes are record IDs (strings), edges carry
             'cosine_similarity' and 'ann_score' attributes.
    stats  : GraphStats
    """
    cfg = config or GraphConfig()

    with profile_block("Phase 6 — similarity graph construction"):
        G: nx.Graph = nx.Graph()

        # Add every record as a node even if it has no edges (isolated node =
        # singleton cluster, representing a unique / unmatched entity).
        G.add_nodes_from(record_ids)

        n_added = 0
        n_skipped_duplicate = 0

        for edge in refinement_result.edges:
            src_id = record_ids[edge.src_idx]
            dst_id = record_ids[edge.dst_idx]

            if cfg.deduplicate_edges and G.has_edge(src_id, dst_id):
                # Keep the higher cosine score.
                existing = G[src_id][dst_id]["cosine_similarity"]
                if edge.cosine_score > existing:
                    G[src_id][dst_id]["cosine_similarity"] = edge.cosine_score
                    G[src_id][dst_id]["ann_score"] = edge.ann_score
                n_skipped_duplicate += 1
                continue

            G.add_edge(
                src_id,
                dst_id,
                cosine_similarity=edge.cosine_score,
                ann_score=edge.ann_score,
                weight=edge.cosine_score,   # standard networkx weight key
            )
            n_added += 1

    stats = _compute_stats(G)
    if n_skipped_duplicate:
        logger.info("Deduplicated %d duplicate edges (kept higher score)", n_skipped_duplicate)
    logger.info(str(stats))
    return G, stats


def _compute_stats(G: nx.Graph) -> GraphStats:
    """Compute summary statistics for the graph."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / max(n_nodes, 1)

    components = list(nx.connected_components(G))
    component_sizes = sorted([len(c) for c in components], reverse=True)
    n_components = len(component_sizes)
    largest = component_sizes[0] if component_sizes else 0

    # Graph density = actual edges / max possible edges.
    density = nx.density(G)

    return GraphStats(
        n_nodes=n_nodes,
        n_edges=n_edges,
        avg_degree=avg_degree,
        n_components=n_components,
        component_sizes=component_sizes,
        largest_component=largest,
        density=density,
    )
