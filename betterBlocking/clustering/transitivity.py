"""
Phase 8 — Transitivity Analysis.

THE TRANSITIVE CLOSURE PROBLEM
================================
Entity resolution via graph connectivity suffers from *transitivity
propagation* (also called the *chaining problem*):

  - A–B: high similarity (0.95) → edge exists.
  - B–C: high similarity (0.92) → edge exists.
  - A–C: low similarity (0.35) → NO direct edge.

Yet A, B, and C land in the same connected component and hence the same cluster.
If A and C are truly different entities that happen to share a common
intermediate record B, the cluster over-merges two distinct entities.

WHY CONNECTED COMPONENTS CAN OVER-MERGE
=========================================
Connected components cluster by *reachability*, not by *cohesion*.  Any two
nodes reachable through a chain of edges belong to the same cluster regardless
of whether they are directly similar.  This is desirable for true entity chains
(A = "John Smith, 123 Main St" ↔ B = "J. Smith, 123 Main" ↔ C = "John Smith,
Main St") but harmful when B is an ambiguous record that bridges two genuinely
different entities.

MITIGATION STRATEGIES
======================
1. Raise the similarity threshold (SimilarityConfig.threshold).
2. Prune low-similarity edges within clusters before final clustering
   (ClusteringConfig.prune_low_similarity_edges).
3. Use community detection (Louvain) instead of connected components; Louvain
   optimises for intra-community density and resists chaining.
4. Post-hoc: flag suspicious clusters (this module) for manual review.

This module implements detection and optional pruning, leaving the decision to
accept or act on suspicious clusters to the user.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx

from .extractor import ClusteringResult, Cluster
from ..ann.normalizer import NormalizedVectors
from ..utils.config import ClusteringConfig
from ..utils.logging_utils import get_logger
from ..utils.profiling import profile_block

logger = get_logger(__name__)


@dataclass
class SuspiciousCluster:
    """A cluster flagged by the transitivity analyser.

    Attributes
    ----------
    cluster_id        : int
    members           : list[str]
    sim_variance      : float — variance of all intra-cluster pairwise cosine
                        scores.  High variance signals heterogeneous pairs.
    min_pairwise_sim  : float — minimum pairwise similarity within the cluster.
    low_sim_pairs     : list[(str, str, float)] — pairs below the low-sim
                        threshold, each as (src_id, dst_id, score).
    suspected_chains  : list[(str, str, str)] — (A, B, C) triples where A–B
                        and B–C are edges but A–C is not.
    """

    cluster_id: int
    members: List[str]
    sim_variance: float
    min_pairwise_sim: float
    low_sim_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    suspected_chains: List[Tuple[str, str, str]] = field(default_factory=list)


@dataclass
class TransitivityReport:
    """Full output of Phase 8.

    Attributes
    ----------
    suspicious_clusters : list[SuspiciousCluster]
    n_clusters_analysed : int
    n_flagged           : int
    pruned_graph        : nx.Graph | None — graph after optional edge pruning.
    """

    suspicious_clusters: List[SuspiciousCluster] = field(default_factory=list)
    n_clusters_analysed: int = 0
    n_flagged: int = 0
    pruned_graph: Optional[nx.Graph] = None

    def __str__(self) -> str:
        return (
            f"Transitivity: {self.n_clusters_analysed} clusters analysed | "
            f"{self.n_flagged} flagged as suspicious"
        )


def analyze_transitivity(
    G: nx.Graph,
    clustering_result: ClusteringResult,
    normalized: NormalizedVectors,
    config: ClusteringConfig | None = None,
) -> TransitivityReport:
    """Phase 8 entry point.

    Parameters
    ----------
    G                 : nx.Graph — similarity graph from Phase 6.
    clustering_result : ClusteringResult — clusters from Phase 7.
    normalized        : NormalizedVectors — sparse L2-normalised matrix.
    config            : ClusteringConfig | None

    Returns
    -------
    TransitivityReport
    """
    cfg = config or ClusteringConfig()
    var_threshold = cfg.transitivity_variance_threshold
    low_sim_thr = cfg.intra_cluster_low_sim_threshold

    # Build a reverse lookup: record_id → row index in the normalised matrix.
    id_to_idx: Dict[str, int] = {
        rid: i for i, rid in enumerate(normalized.record_ids)
    }

    suspicious: List[SuspiciousCluster] = []
    multi_clusters = [c for c in clustering_result.clusters if not c.is_singleton]

    with profile_block("Phase 8 — transitivity analysis"):
        for cluster in multi_clusters:
            sus = _analyse_cluster(
                cluster, G, normalized, id_to_idx, var_threshold, low_sim_thr
            )
            if sus is not None:
                suspicious.append(sus)

    pruned_graph: Optional[nx.Graph] = None
    if cfg.prune_low_similarity_edges and suspicious:
        pruned_graph = _prune_edges(G, suspicious, low_sim_thr)
        logger.info(
            "Edge pruning: removed low-similarity edges from %d suspicious clusters. "
            "Pruned graph has %d edges (was %d).",
            len(suspicious),
            pruned_graph.number_of_edges(),
            G.number_of_edges(),
        )

    report = TransitivityReport(
        suspicious_clusters=suspicious,
        n_clusters_analysed=len(multi_clusters),
        n_flagged=len(suspicious),
        pruned_graph=pruned_graph,
    )
    logger.info(str(report))
    return report


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _pairwise_cosine_in_cluster(
    members: List[str],
    normalized: NormalizedVectors,
    id_to_idx: Dict[str, int],
) -> np.ndarray:
    """Compute the upper-triangle pairwise cosine scores within a cluster.

    Works with the pre-normalised sparse matrix so the dot product equals
    cosine similarity.  Returns a flat array of all n*(n-1)/2 unique scores.
    """
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cos

    indices = [id_to_idx[m] for m in members if m in id_to_idx]
    if len(indices) < 2:
        return np.array([1.0])

    sub = normalized.sparse_matrix[indices]   # (k, d) sparse
    sim_matrix = sklearn_cos(sub)             # (k, k) dense

    # Extract upper triangle (excluding diagonal).
    k = len(indices)
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            pairs.append(sim_matrix[i, j])
    return np.array(pairs)


def _detect_chains(
    cluster: Cluster,
    G: nx.Graph,
    low_sim_thr: float,
) -> List[Tuple[str, str, str]]:
    """Find (A, B, C) triples where A–B, B–C are edges but A–C is not."""
    chains = []
    members_set = set(cluster.members)

    for B in cluster.members:
        neighbours_of_B = [
            n for n in G.neighbors(B) if n in members_set
        ]
        for i, A in enumerate(neighbours_of_B):
            for C in neighbours_of_B[i + 1:]:
                if not G.has_edge(A, C):
                    chains.append((A, B, C))
    return chains


def _analyse_cluster(
    cluster: Cluster,
    G: nx.Graph,
    normalized: NormalizedVectors,
    id_to_idx: Dict[str, int],
    var_threshold: float,
    low_sim_thr: float,
) -> Optional[SuspiciousCluster]:
    """Return a SuspiciousCluster if the cluster looks problematic, else None."""
    pairwise = _pairwise_cosine_in_cluster(cluster.members, normalized, id_to_idx)
    variance = float(np.var(pairwise))
    min_sim = float(np.min(pairwise))

    # Identify specific low-similarity direct edges within the cluster.
    low_sim_pairs: List[Tuple[str, str, float]] = []
    for u, v, data in G.edges(cluster.members, data=True):
        if u not in set(cluster.members) or v not in set(cluster.members):
            continue
        sim = data.get("cosine_similarity", 0.0)
        if sim < low_sim_thr:
            low_sim_pairs.append((u, v, sim))

    suspected_chains = _detect_chains(cluster, G, low_sim_thr)

    is_suspicious = (
        variance > var_threshold
        or min_sim < low_sim_thr
        or len(suspected_chains) > 0
    )
    if not is_suspicious:
        return None

    return SuspiciousCluster(
        cluster_id=cluster.cluster_id,
        members=cluster.members,
        sim_variance=variance,
        min_pairwise_sim=min_sim,
        low_sim_pairs=low_sim_pairs,
        suspected_chains=suspected_chains,
    )


def _prune_edges(
    G: nx.Graph,
    suspicious: List[SuspiciousCluster],
    low_sim_thr: float,
) -> nx.Graph:
    """Return a copy of G with low-similarity intra-cluster edges removed."""
    H = G.copy()
    for sus in suspicious:
        for u, v, sim in sus.low_sim_pairs:
            if H.has_edge(u, v) and sim < low_sim_thr:
                H.remove_edge(u, v)
    return H
