"""
Configuration dataclasses for every stage of the ANN-based Entity Resolution pipeline.

Centralising parameters here makes it easy to tune the system from a single
location and to serialise / log the exact configuration used for a given run.
"""

from dataclasses import dataclass, field


@dataclass
class NormalizationConfig:
    """Controls Phase 2 — sparse L2 vector normalisation."""

    # When True the aggregated multi-hot record vector is written to disk so it
    # can be reused across runs without re-computing.
    cache_normalized: bool = False
    cache_path: str = "normalized_vectors.npz"


@dataclass
class ANNConfig:
    """Controls Phase 3-4 — FAISS HNSW index construction and retrieval.

    HNSW (Hierarchical Navigable Small World) parameters:
      M              — number of bi-directional links per node in the graph;
                       higher M ⇒ better recall but more memory.
      efConstruction — beam width during index build; higher ⇒ better graph
                       quality but slower build.
      efSearch       — beam width during query; higher ⇒ better recall but
                       slower search. Can be changed after building the index.
      top_k          — neighbours to retrieve per query (including self).
      batch_size     — number of vectors sent to FAISS per search call; tune
                       for GPU/CPU cache behaviour.
    """

    M: int = 32
    efConstruction: int = 200
    efSearch: int = 128
    top_k: int = 10
    batch_size: int = 1024
    use_gpu: bool = False          # Flip to True when a CUDA device is present


@dataclass
class SimilarityConfig:
    """Controls Phase 5 — exact cosine similarity refinement."""

    # Only pairs whose exact cosine similarity meets this threshold become
    # candidate edges in the similarity graph.
    threshold: float = 0.80

    # Maximum number of ANN candidates refined per source record in one batch.
    # Keeping this moderate avoids materialising large dense sub-matrices.
    refinement_batch_size: int = 512


@dataclass
class GraphConfig:
    """Controls Phase 6 — similarity graph construction."""

    # When True, only the highest-similarity edge between two nodes is kept if
    # multiple ANN batches produce duplicates.
    deduplicate_edges: bool = True


@dataclass
class ClusteringConfig:
    """Controls Phase 7-8 — cluster extraction and transitivity analysis."""

    # 'connected_components' (fast, no extra deps) or 'louvain' (requires
    # python-louvain / community package; finds denser communities).
    method: str = "connected_components"

    # Intra-cluster similarity below this value triggers a transitivity warning.
    transitivity_variance_threshold: float = 0.10

    # Pairs within a cluster whose cosine similarity falls below this value are
    # flagged as suspicious (potential over-merge via chaining).
    intra_cluster_low_sim_threshold: float = 0.50

    # When True, edges whose cosine similarity is below
    # intra_cluster_low_sim_threshold are removed and clustering is re-run.
    prune_low_similarity_edges: bool = False


@dataclass
class VisualizationConfig:
    """Controls Phase 9 — visualisation artefacts."""

    output_dir: str = "artifacts"
    dpi: int = 150
    show_plots: bool = False       # Set True in an interactive notebook
    max_graph_nodes: int = 500     # Cap node count drawn in graph plot
    pca_components: int = 2        # Dimensions for PCA scatter (2 or 3)


@dataclass
class PipelineConfig:
    """Top-level configuration that bundles all stage configs."""

    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    ann: ANNConfig = field(default_factory=ANNConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Global random seed for reproducibility.
    random_seed: int = 42

    # When True every stage logs timing and peak-memory figures.
    enable_profiling: bool = True
