"""
Phase 3 — FAISS HNSW Index Construction.

WHY ANN IS NECESSARY
=====================
Exact entity resolution requires comparing every record to every other record —
an O(n²) operation.  For n = 1 000 000 records with vectors of dimension d,
this means computing 5 × 10¹¹ dot products.  Even at 10⁹ operations/second,
that is ~500 000 seconds (~6 days).

Approximate Nearest Neighbour (ANN) search trades a small, controllable
accuracy loss for dramatic speed improvements:

  Exhaustive flat search  → O(n × d) per query, O(n² × d) total
  HNSW ANN search         → O(log n × d) per query (empirically sub-linear)

At n = 10⁶ the speed-up can exceed 1 000× while retaining > 95% recall for
carefully chosen index parameters.

WHY HNSW (Hierarchical Navigable Small World)
==============================================
HNSW builds a multi-layer proximity graph where:
  - The bottom layer 0 contains all vectors with dense connections.
  - Upper layers contain progressively fewer "long-range" links that enable
    fast traversal across the graph.

A query enters the graph at the top layer and greedily navigates toward the
query vector, descending layer by layer.  This mirrors skip-list navigation.

HNSW advantages over IVF-based indexes:
  - No training step (no k-means on vectors).
  - Excellent recall at moderate M and ef values.
  - Supports incremental additions without a full rebuild.
  - Low latency per query (important for interactive pipelines).

FAISS INNER PRODUCT METRIC AND COSINE SIMILARITY
=================================================
After L2 normalisation (Phase 2), every vector ‖x‖₂ = 1.  Therefore:

    cosine(a, b) = a · b  (when ‖a‖ = ‖b‖ = 1)

FAISS's IndexHNSWFlat with METRIC_INNER_PRODUCT computes exact inner products
within the ANN graph traversal, giving us cosine retrieval without a dedicated
cosine index type.

COMPLEXITY SUMMARY
==================
Build:  O(n × M × log n × d)   — each of n vectors connects to M neighbours
Query:  O(efSearch × log n × d) — beam of efSearch candidates per query
Space:  O(n × (M × 2 + d))     — graph links + raw vectors
"""

import numpy as np

try:
    import faiss
except ImportError as exc:
    raise ImportError(
        "faiss-cpu (or faiss-gpu) is required. Install with: pip install faiss-cpu"
    ) from exc

from ..utils.config import ANNConfig
from ..utils.logging_utils import get_logger
from ..utils.profiling import profile_block

logger = get_logger(__name__)


class ANNIndex:
    """FAISS HNSW index wrapper with cosine-similarity semantics.

    Attributes
    ----------
    index    : faiss.Index — the underlying FAISS index.
    dim      : int         — vector dimensionality.
    n_vectors: int         — number of vectors added to the index.
    config   : ANNConfig
    """

    def __init__(self, config: ANNConfig | None = None) -> None:
        self.config = config or ANNConfig()
        self.index: faiss.Index | None = None
        self.dim: int = 0
        self.n_vectors: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, dense_vectors: np.ndarray) -> None:
        """Construct the HNSW index from a dense float32 matrix.

        Parameters
        ----------
        dense_vectors : np.ndarray, shape (n, d), dtype float32
            L2-normalised vectors produced by Phase 2.  Must be C-contiguous.
        """
        n, d = dense_vectors.shape
        if dense_vectors.dtype != np.float32:
            dense_vectors = dense_vectors.astype(np.float32)
        if not dense_vectors.flags["C_CONTIGUOUS"]:
            dense_vectors = np.ascontiguousarray(dense_vectors)

        self.dim = d
        cfg = self.config

        with profile_block(f"Phase 3 — HNSW index build (n={n}, d={d}, M={cfg.M})"):
            if cfg.use_gpu:
                self._build_gpu(dense_vectors, n, d)
            else:
                self._build_cpu(dense_vectors, n, d)

        self.n_vectors = n
        logger.info(
            "HNSW index built: %d vectors | dim=%d | M=%d | "
            "efConstruction=%d | efSearch=%d",
            n, d, cfg.M, cfg.efConstruction, cfg.efSearch,
        )

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search the index for approximate nearest neighbours.

        Parameters
        ----------
        query_vectors : np.ndarray, shape (q, d), dtype float32
        top_k         : int | None — overrides config.top_k when provided.

        Returns
        -------
        distances : np.ndarray, shape (q, k) — inner-product scores (≈ cosine).
        indices   : np.ndarray, shape (q, k) — 0-based row indices into the
                    original dense_vectors matrix passed to build().
        """
        if self.index is None:
            raise RuntimeError("Index has not been built. Call build() first.")

        k = top_k if top_k is not None else self.config.top_k
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)
        if not query_vectors.flags["C_CONTIGUOUS"]:
            query_vectors = np.ascontiguousarray(query_vectors)

        distances, indices = self.index.search(query_vectors, k)
        return distances, indices

    def set_ef_search(self, ef: int) -> None:
        """Dynamically adjust the efSearch beam width without rebuilding."""
        if self.index is None:
            raise RuntimeError("Index has not been built.")
        self.index.hnsw.efSearch = ef
        logger.info("efSearch updated to %d", ef)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_cpu(self, vectors: np.ndarray, n: int, d: int) -> None:
        cfg = self.config
        # IndexHNSWFlat stores raw vectors (flat storage) with HNSW graph.
        # METRIC_INNER_PRODUCT → dot product; equals cosine when normalised.
        index = faiss.IndexHNSWFlat(d, cfg.M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = cfg.efConstruction
        index.hnsw.efSearch = cfg.efSearch
        index.add(vectors)
        self.index = index

    def _build_gpu(self, vectors: np.ndarray, n: int, d: int) -> None:
        """GPU-accelerated index build.

        FAISS GPU resources must be freed explicitly; they are kept as an
        instance attribute so the GC does not prematurely reclaim them.
        """
        cfg = self.config
        try:
            res = faiss.StandardGpuResources()
            self._gpu_res = res  # prevent GC
            cpu_index = faiss.IndexHNSWFlat(d, cfg.M, faiss.METRIC_INNER_PRODUCT)
            cpu_index.hnsw.efConstruction = cfg.efConstruction
            cpu_index.hnsw.efSearch = cfg.efSearch
            # HNSW graphs cannot be transferred to GPU; build on CPU, move
            # the flat storage for GPU-accelerated search.
            # Alternatively, use IndexIVFFlat on GPU for very large corpora.
            cpu_index.add(vectors)
            self.index = cpu_index
            logger.warning(
                "HNSW graph construction runs on CPU; GPU acceleration for "
                "flat search is available via IndexFlatIP if desired."
            )
        except AttributeError:
            logger.warning("faiss-gpu not available; falling back to CPU build.")
            self._build_cpu(vectors, n, d)
