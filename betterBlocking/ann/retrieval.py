"""
Phase 4 — Approximate Nearest Neighbour Retrieval.

For each record we retrieve the top-k most similar records according to the
HNSW index (inner product ≈ cosine, post normalisation).  Self-matches (a
record is always its own nearest neighbour) are filtered out.

Batch retrieval strategy
------------------------
Sending all n query vectors to FAISS in a single call would allocate an
(n × k) index-result matrix in memory.  For n = 10⁶ and k = 20 that is
~160 MiB for int64 indices alone.  We therefore process the corpus in
configurable batches of size ANNConfig.batch_size, streaming the results
into a neighbour dictionary.

Scalability note
----------------
This module avoids storing the dense query matrix for the entire dataset at
once.  It accepts a reference to the pre-built NormalizedVectors and slices
rows in-place, keeping peak memory proportional to batch_size rather than n.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from .index import ANNIndex
from .normalizer import NormalizedVectors
from ..utils.config import ANNConfig
from ..utils.logging_utils import get_logger
from ..utils.profiling import profile_block

logger = get_logger(__name__)


@dataclass
class NeighborResult:
    """ANN retrieval output for the full corpus.

    Attributes
    ----------
    neighbors : dict {record_idx (int): list of (neighbor_idx, ann_score)}
        Only neighbours with a valid (non-negative) FAISS index are stored.
        Self-matches are excluded.  ann_score is the raw FAISS inner-product
        score (≈ cosine similarity after L2 normalisation).
    record_ids : list[str]
        The same ordering used in NormalizedVectors; maps int indices to IDs.
    n_candidates : int
        Total (source, neighbor) pairs before exact refinement.
    """

    neighbors: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)
    record_ids: List[str] = field(default_factory=list)
    n_candidates: int = 0


def retrieve_neighbors(
    index: ANNIndex,
    normalized: NormalizedVectors,
    config: ANNConfig | None = None,
) -> NeighborResult:
    """Phase 4 entry point: batch ANN retrieval across the full corpus.

    Parameters
    ----------
    index      : ANNIndex — built HNSW index (Phase 3).
    normalized : NormalizedVectors — output of Phase 2.
    config     : ANNConfig | None — uses ANNIndex's config if None.

    Returns
    -------
    NeighborResult
    """
    cfg = config or index.config
    dense = normalized.dense_matrix
    n = normalized.n_records
    top_k = cfg.top_k
    batch_size = cfg.batch_size

    neighbors: Dict[int, List[Tuple[int, float]]] = {}
    total_candidates = 0

    with profile_block(f"Phase 4 — ANN retrieval (n={n}, top_k={top_k})"):
        n_batches = (n + batch_size - 1) // batch_size

        for batch_idx in tqdm(
            range(n_batches),
            desc="ANN retrieval",
            unit="batch",
            ncols=90,
        ):
            start = batch_idx * batch_size
            end = min(start + batch_size, n)

            query_batch = dense[start:end]               # (batch, d) float32
            distances, indices = index.search(query_batch, top_k)
            # distances[i, j] = inner-product score (≈ cosine) of query i
            # to its j-th approximate neighbour.
            # indices[i, j]   = row index in the original corpus (0-based).

            for local_i in range(end - start):
                global_i = start + local_i
                nn_list: List[Tuple[int, float]] = []

                for rank in range(top_k):
                    nb_idx = int(indices[local_i, rank])
                    nb_score = float(distances[local_i, rank])

                    # FAISS returns -1 for padded slots when fewer than k
                    # neighbours exist.
                    if nb_idx < 0:
                        continue
                    # Exclude self-match.
                    if nb_idx == global_i:
                        continue

                    nn_list.append((nb_idx, nb_score))

                if nn_list:
                    neighbors[global_i] = nn_list
                    total_candidates += len(nn_list)

    logger.info(
        "ANN retrieval complete: %d records with neighbours | "
        "%d total candidate pairs",
        len(neighbors),
        total_candidates,
    )

    return NeighborResult(
        neighbors=neighbors,
        record_ids=normalized.record_ids,
        n_candidates=total_candidates,
    )
