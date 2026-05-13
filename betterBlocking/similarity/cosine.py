"""
Phase 5 — Exact Cosine Similarity Refinement.

ANN retrieval (Phase 4) gives us a small set of *candidate* neighbour pairs
per record at sub-linear cost.  However, HNSW scores are approximate: the
graph traversal can miss exact nearest neighbours and the inner-product values
may differ slightly from true cosine similarity (floating-point path, graph
approximation bias).

This phase re-computes the *exact* cosine similarity for every candidate pair
using sklearn's cosine_similarity on the original sparse normalised vectors.
Working with the sparse matrix directly avoids materialising a dense n × n
comparison matrix; only the (source_rows × candidate_rows) sub-matrix is ever
instantiated, and only when candidates exist for a source record.

Design choices
--------------
* Sparse-aware computation: sklearn.metrics.pairwise.cosine_similarity accepts
  CSR matrices.  When the source row is a (1 × d) sparse row and the target
  rows are a (batch × d) sparse slice, the result is a (1 × batch) dense
  matrix — far smaller than a full n × d dense matrix.
* Batching: candidates are grouped by source record to minimise repeated sparse
  row lookups.
* Threshold filtering: only pairs whose exact cosine similarity ≥ threshold
  survive to graph construction, removing ANN false positives.
* Comparison logging: the module logs aggregate statistics on ANN vs. exact
  score differences so the quality of the ANN approximation can be monitored.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity

from ..ann.retrieval import NeighborResult
from ..ann.normalizer import NormalizedVectors
from ..utils.config import SimilarityConfig
from ..utils.logging_utils import get_logger
from ..utils.profiling import profile_block

logger = get_logger(__name__)


@dataclass
class RefinedEdge:
    """A pair of records whose exact cosine similarity passes the threshold.

    Attributes
    ----------
    src_idx       : int   — row index of source record in the normalised matrix.
    dst_idx       : int   — row index of destination record.
    ann_score     : float — approximate inner-product score from FAISS.
    cosine_score  : float — exact cosine similarity.
    """

    src_idx: int
    dst_idx: int
    ann_score: float
    cosine_score: float


@dataclass
class RefinementResult:
    """Output of the exact cosine refinement stage.

    Attributes
    ----------
    edges            : list[RefinedEdge]
    n_candidates_in  : int — candidate pairs entering refinement.
    n_edges_out      : int — pairs surviving the threshold filter.
    mean_ann_error   : float — mean |ann_score − cosine_score| over all pairs.
    threshold        : float — the similarity threshold applied.
    """

    edges: List[RefinedEdge] = field(default_factory=list)
    n_candidates_in: int = 0
    n_edges_out: int = 0
    mean_ann_error: float = 0.0
    threshold: float = 0.8

    def summary(self) -> str:
        return (
            f"Refinement: {self.n_candidates_in} candidates → "
            f"{self.n_edges_out} edges "
            f"(threshold={self.threshold:.2f}, "
            f"mean ANN error={self.mean_ann_error:.4f})"
        )


def refine_candidates(
    neighbor_result: NeighborResult,
    normalized: NormalizedVectors,
    config: SimilarityConfig | None = None,
) -> RefinementResult:
    """Phase 5 entry point: exact cosine refinement of ANN candidates.

    Parameters
    ----------
    neighbor_result : NeighborResult  — output of Phase 4.
    normalized      : NormalizedVectors — sparse normalised vectors (Phase 2).
    config          : SimilarityConfig | None

    Returns
    -------
    RefinementResult
    """
    cfg = config or SimilarityConfig()
    threshold = cfg.threshold
    sparse = normalized.sparse_matrix     # (n, vocab_size) CSR, L2-normalised

    edges: List[RefinedEdge] = []
    ann_errors: List[float] = []
    n_candidates_in = neighbor_result.n_candidates

    with profile_block("Phase 5 — exact cosine refinement"):
        for src_idx, candidates in neighbor_result.neighbors.items():
            if not candidates:
                continue

            candidate_indices = [nb_idx for nb_idx, _ in candidates]
            ann_scores = [score for _, score in candidates]

            # Extract source row as (1, d) CSR slice — stays sparse.
            src_row = sparse[src_idx]          # (1, d)

            # Extract all candidate rows at once as (k, d) CSR slice.
            # scipy fancy indexing with a list returns CSR.
            cand_matrix = sparse[candidate_indices]  # (k, d)

            # cosine_similarity accepts sparse; returns (1, k) dense array.
            # Because both matrices are already L2-normalised, this is
            # equivalent to the dot product, but sklearn handles the edge
            # case of zero-norm rows gracefully.
            exact_scores: np.ndarray = cosine_similarity(
                src_row, cand_matrix
            ).ravel()   # shape (k,)

            for rank, (nb_idx, ann_score) in enumerate(candidates):
                exact = float(exact_scores[rank])
                ann_errors.append(abs(ann_score - exact))

                if exact >= threshold:
                    edges.append(
                        RefinedEdge(
                            src_idx=src_idx,
                            dst_idx=nb_idx,
                            ann_score=ann_score,
                            cosine_score=exact,
                        )
                    )

    mean_error = float(np.mean(ann_errors)) if ann_errors else 0.0

    result = RefinementResult(
        edges=edges,
        n_candidates_in=n_candidates_in,
        n_edges_out=len(edges),
        mean_ann_error=mean_error,
        threshold=threshold,
    )
    logger.info(result.summary())
    return result
