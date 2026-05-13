"""
Phase 9 — Visualisation.

All plots use matplotlib only (no seaborn, plotly, or other dependencies).
Figures are saved to disk as PNG files and optionally displayed interactively.

Available plots
---------------
1. plot_similarity_graph        — networkx graph layout (spring / kamada-kawai).
2. plot_cluster_size_distribution — histogram of cluster sizes.
3. plot_cosine_similarity_histogram — distribution of edge cosine scores.
4. plot_pca_vectors             — PCA projection of normalised vectors coloured
                                  by cluster assignment.
5. save_all_artifacts           — convenience wrapper that calls all four.
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend; safe in headless environments.
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np

from ..clustering.extractor import ClusteringResult
from ..similarity.cosine import RefinementResult
from ..ann.normalizer import NormalizedVectors
from ..utils.config import VisualizationConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_similarity_graph(
    G: nx.Graph,
    clustering_result: ClusteringResult,
    config: VisualizationConfig | None = None,
) -> str:
    """Draw the similarity graph, colouring nodes by cluster assignment.

    For large graphs only a random sample of max_graph_nodes nodes is drawn
    to keep the plot readable.

    Returns
    -------
    str — path to the saved PNG file.
    """
    cfg = config or VisualizationConfig()
    _ensure_dir(cfg.output_dir)
    out_path = os.path.join(cfg.output_dir, "similarity_graph.png")

    nodes = list(G.nodes())
    if len(nodes) > cfg.max_graph_nodes:
        rng = np.random.default_rng(42)
        nodes = list(rng.choice(nodes, size=cfg.max_graph_nodes, replace=False))
        subG = G.subgraph(nodes).copy()
        logger.info("Graph plot: sampling %d / %d nodes", cfg.max_graph_nodes, G.number_of_nodes())
    else:
        subG = G

    cluster_map = clustering_result.cluster_map
    n_clusters = clustering_result.n_clusters
    colormap = cm.get_cmap("tab20", min(n_clusters, 20))
    node_colors = [
        colormap(cluster_map.get(n, 0) % 20) for n in subG.nodes()
    ]

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(subG, seed=42, k=0.5)
    edge_weights = [subG[u][v].get("cosine_similarity", 0.5) for u, v in subG.edges()]

    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=30, ax=ax, alpha=0.8)
    nx.draw_networkx_edges(
        subG, pos,
        edge_color=edge_weights,
        edge_cmap=cm.Blues,
        width=0.6,
        alpha=0.6,
        ax=ax,
    )
    ax.set_title(
        f"Similarity Graph  |  nodes={subG.number_of_nodes()}  "
        f"edges={subG.number_of_edges()}  clusters={n_clusters}",
        fontsize=12,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)
    logger.info("Saved similarity graph → %s", out_path)
    return out_path


def plot_cluster_size_distribution(
    clustering_result: ClusteringResult,
    config: VisualizationConfig | None = None,
) -> str:
    """Histogram of cluster sizes (log-scale x-axis for skewed distributions).

    Returns
    -------
    str — path to the saved PNG file.
    """
    cfg = config or VisualizationConfig()
    _ensure_dir(cfg.output_dir)
    out_path = os.path.join(cfg.output_dir, "cluster_size_distribution.png")

    sizes = [c.size for c in clustering_result.clusters]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Linear histogram.
    axes[0].hist(sizes, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Cluster size")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Cluster size distribution (linear)")
    axes[0].axvline(
        np.mean(sizes), color="red", linestyle="--",
        label=f"mean={np.mean(sizes):.1f}"
    )
    axes[0].legend()

    # Log-scale histogram (useful when most clusters are singletons).
    log_sizes = np.log1p(sizes)
    axes[1].hist(log_sizes, bins=30, color="darkorange", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("log(1 + cluster size)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Cluster size distribution (log1p)")

    fig.suptitle(
        f"Clusters: total={clustering_result.n_clusters} | "
        f"singletons={clustering_result.n_singletons} | "
        f"largest={clustering_result.largest_cluster}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)
    logger.info("Saved cluster size distribution → %s", out_path)
    return out_path


def plot_cosine_similarity_histogram(
    refinement_result: RefinementResult,
    config: VisualizationConfig | None = None,
) -> str:
    """Histogram of exact cosine similarity scores for all retained edges.

    Also shows the ANN approximate scores side-by-side to visualise the
    approximation error introduced by HNSW.

    Returns
    -------
    str — path to the saved PNG file.
    """
    cfg = config or VisualizationConfig()
    _ensure_dir(cfg.output_dir)
    out_path = os.path.join(cfg.output_dir, "cosine_similarity_histogram.png")

    exact_scores = [e.cosine_score for e in refinement_result.edges]
    ann_scores = [e.ann_score for e in refinement_result.edges]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(exact_scores, bins=40, color="mediumseagreen", edgecolor="white", alpha=0.85)
    axes[0].axvline(
        refinement_result.threshold, color="red", linestyle="--",
        label=f"threshold={refinement_result.threshold:.2f}"
    )
    axes[0].set_xlabel("Exact cosine similarity")
    axes[0].set_ylabel("Edge count")
    axes[0].set_title("Exact cosine similarity distribution")
    axes[0].legend()

    axes[1].hist(
        [e.cosine_score - e.ann_score for e in refinement_result.edges],
        bins=40, color="mediumpurple", edgecolor="white", alpha=0.85,
    )
    axes[1].set_xlabel("Exact − ANN score (approximation error)")
    axes[1].set_ylabel("Edge count")
    axes[1].set_title(
        f"ANN approximation error  (mean={refinement_result.mean_ann_error:.4f})"
    )

    fig.suptitle(
        f"Cosine similarity for {len(exact_scores)} edges", fontsize=12
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)
    logger.info("Saved cosine similarity histogram → %s", out_path)
    return out_path


def plot_pca_vectors(
    normalized: NormalizedVectors,
    clustering_result: ClusteringResult,
    config: VisualizationConfig | None = None,
    max_points: int = 2000,
) -> str:
    """2-D PCA scatter of L2-normalised record vectors coloured by cluster.

    PCA is computed on a random sample of at most *max_points* records to keep
    runtime manageable for large corpora.

    Returns
    -------
    str — path to the saved PNG file.
    """
    from sklearn.decomposition import TruncatedSVD   # sparse-friendly PCA

    cfg = config or VisualizationConfig()
    _ensure_dir(cfg.output_dir)
    out_path = os.path.join(cfg.output_dir, "pca_vectors.png")

    n = normalized.n_records
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n, size=min(max_points, n), replace=False)

    sparse_sample = normalized.sparse_matrix[sample_idx]
    record_ids_sample = [normalized.record_ids[i] for i in sample_idx]
    cluster_map = clustering_result.cluster_map

    # TruncatedSVD is the sparse-friendly equivalent of PCA.
    n_components = min(2, sparse_sample.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    coords = svd.fit_transform(sparse_sample)   # (sample, 2)

    cluster_ids = np.array([cluster_map.get(rid, -1) for rid in record_ids_sample])
    unique_cids = np.unique(cluster_ids)
    colormap = cm.get_cmap("tab20", min(len(unique_cids), 20))
    cid_to_color = {cid: colormap(i % 20) for i, cid in enumerate(unique_cids)}
    colors = [cid_to_color[cid] for cid in cluster_ids]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1] if coords.shape[1] > 1 else np.zeros(len(coords)),
               c=colors, s=10, alpha=0.7)
    ax.set_xlabel(f"PC1 ({svd.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(
        f"PC2 ({svd.explained_variance_ratio_[1]*100:.1f}% var)"
        if coords.shape[1] > 1 else "PC2"
    )
    ax.set_title(
        f"PCA of L2-normalised vectors  "
        f"(sample n={len(sample_idx)}, clusters={clustering_result.n_clusters})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)
    logger.info("Saved PCA plot → %s", out_path)
    return out_path


def save_all_artifacts(
    G: nx.Graph,
    clustering_result: ClusteringResult,
    refinement_result: RefinementResult,
    normalized: NormalizedVectors,
    config: VisualizationConfig | None = None,
) -> List[str]:
    """Convenience wrapper: generate and save all visualisation artefacts.

    Returns
    -------
    list[str] — paths to all saved PNG files.
    """
    paths = []
    paths.append(plot_cluster_size_distribution(clustering_result, config))
    if refinement_result.edges:
        paths.append(plot_cosine_similarity_histogram(refinement_result, config))
    if G.number_of_nodes() > 0:
        paths.append(plot_similarity_graph(G, clustering_result, config))
    if normalized.n_records > 1 and normalized.vocab_size > 2:
        paths.append(plot_pca_vectors(normalized, clustering_result, config))
    return paths
