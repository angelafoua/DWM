"""
Phase 7 — Cluster Extraction.

After graph construction every connected component is a candidate entity
cluster.  Two alternative algorithms are supported:

connected_components (default)
  Fast, deterministic, O(n + e).  Every connected component becomes one
  cluster.  Simple and interpretable but susceptible to the chaining problem
  analysed in Phase 8.

louvain (optional)
  Community detection via the Louvain algorithm (requires python-louvain /
  community package or networkx >= 3.3 via nx.community.louvain_communities).
  Optimises modularity — finds denser, more cohesive subgraphs — but is
  non-deterministic and slower than connected components.  Use when the
  similarity graph is dense enough to benefit from intra-cluster cohesion
  scoring.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set

import networkx as nx

from ..graph.builder import GraphStats
from ..utils.config import ClusteringConfig
from ..utils.logging_utils import get_logger
from ..utils.profiling import profile_block

logger = get_logger(__name__)


@dataclass
class Cluster:
    """A single entity cluster.

    Attributes
    ----------
    cluster_id : int — zero-based cluster identifier (sorted by size desc).
    members    : list[str] — record IDs belonging to this cluster.
    size       : int
    is_singleton : bool — True iff the cluster contains exactly one record
                  (no entity matches found).
    """

    cluster_id: int
    members: List[str]
    size: int
    is_singleton: bool


@dataclass
class ClusteringResult:
    """Full output of the clustering phase.

    Attributes
    ----------
    clusters          : list[Cluster] — all clusters, sorted by size desc.
    n_clusters        : int
    n_singletons      : int
    n_non_singletons  : int
    largest_cluster   : int
    avg_cluster_size  : float
    cluster_map       : dict {record_id: cluster_id}
    """

    clusters: List[Cluster] = field(default_factory=list)
    n_clusters: int = 0
    n_singletons: int = 0
    n_non_singletons: int = 0
    largest_cluster: int = 0
    avg_cluster_size: float = 0.0
    cluster_map: Dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Clusters: total={self.n_clusters} | "
            f"singletons={self.n_singletons} | "
            f"multi-record={self.n_non_singletons} | "
            f"largest={self.largest_cluster} | "
            f"avg_size={self.avg_cluster_size:.2f}"
        )


def extract_clusters(
    G: nx.Graph,
    config: ClusteringConfig | None = None,
) -> ClusteringResult:
    """Phase 7 entry point: extract entity clusters from the similarity graph.

    Parameters
    ----------
    G      : nx.Graph — similarity graph from Phase 6.
    config : ClusteringConfig | None

    Returns
    -------
    ClusteringResult
    """
    cfg = config or ClusteringConfig()
    method = cfg.method

    with profile_block(f"Phase 7 — cluster extraction ({method})"):
        if method == "louvain":
            raw_clusters = _louvain_clusters(G, cfg)
        else:
            raw_clusters = _connected_components(G)

        # Sort clusters largest-first for consistent numbering.
        raw_clusters.sort(key=len, reverse=True)

        clusters: List[Cluster] = []
        cluster_map: Dict[str, int] = {}

        for cid, members in enumerate(raw_clusters):
            members_list = sorted(members)
            is_singleton = len(members_list) == 1
            c = Cluster(
                cluster_id=cid,
                members=members_list,
                size=len(members_list),
                is_singleton=is_singleton,
            )
            clusters.append(c)
            for rid in members_list:
                cluster_map[rid] = cid

    sizes = [c.size for c in clusters]
    n_singletons = sum(1 for c in clusters if c.is_singleton)

    result = ClusteringResult(
        clusters=clusters,
        n_clusters=len(clusters),
        n_singletons=n_singletons,
        n_non_singletons=len(clusters) - n_singletons,
        largest_cluster=max(sizes, default=0),
        avg_cluster_size=sum(sizes) / max(len(sizes), 1),
        cluster_map=cluster_map,
    )
    logger.info(str(result))
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _connected_components(G: nx.Graph) -> List[Set[str]]:
    """Return connected components as sets of node IDs."""
    return [set(comp) for comp in nx.connected_components(G)]


def _louvain_clusters(G: nx.Graph, cfg: ClusteringConfig) -> List[Set[str]]:
    """Louvain community detection via networkx >= 3.3 or python-louvain."""
    # Prefer networkx built-in (available in networkx >= 3.3).
    try:
        import networkx.algorithms.community as nx_comm
        communities = nx_comm.louvain_communities(G, weight="cosine_similarity", seed=42)
        return [set(c) for c in communities]
    except (AttributeError, ImportError):
        pass

    # Fallback: python-louvain (community package).
    try:
        import community as community_louvain
        partition: Dict[str, int] = community_louvain.best_partition(
            G, weight="cosine_similarity", random_state=42
        )
        groups: Dict[int, Set[str]] = {}
        for node, cid in partition.items():
            groups.setdefault(cid, set()).add(node)
        return list(groups.values())
    except ImportError:
        logger.warning(
            "Louvain requested but neither networkx >= 3.3 community support "
            "nor python-louvain is available. Falling back to connected_components."
        )
        return _connected_components(G)
