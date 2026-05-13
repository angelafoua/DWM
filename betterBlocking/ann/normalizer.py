"""
Phase 2 — Sparse Vector Aggregation and L2 Normalisation.

WHY NORMALISATION IS REQUIRED
==============================
The one-hot encoding stage (BB10) produces a *per-token* sparse matrix for
every record.  Before computing cosine similarity, two things must happen:

  1. Aggregation: collapse the (k × vocab_size) per-record matrix into a single
     (1 × vocab_size) multi-hot vector by taking the column-wise OR / sum of all
     token rows.  The result is a "bag-of-blocking-tokens" vector where entry j
     is 1 if the record contains blocking token j, and 0 otherwise.

  2. L2 normalisation: divide each vector by its Euclidean norm:

         x̂  =  x / ‖x‖₂

     After normalisation every vector has unit length and lies on the surface of
     the unit hypersphere.

GEOMETRIC INTERPRETATION OF COSINE SIMILARITY
===============================================
The cosine similarity between two vectors a and b is:

    cos(θ) = (a · b) / (‖a‖ · ‖b‖)

where θ is the angle between them.  When both vectors are already L2-normalised
(‖a‖ = ‖b‖ = 1), the formula simplifies to a pure dot product:

    cos(θ) = a · b

This equivalence is crucial: FAISS's inner-product index computes a · b in
O(d) time, which becomes *exact* cosine similarity once vectors are normalised.
We therefore get cosine retrieval without any special index type.

WHY COSINE SIMILARITY WORKS WELL FOR SPARSE BINARY VECTORS
============================================================
Sparse binary (one-hot / multi-hot) vectors have a natural geometric
interpretation: each active dimension is a "feature flag".  Two records that
share blocking tokens will have overlapping active dimensions, producing a
positive dot product.  Records sharing no tokens at all are orthogonal
(dot product = 0, cosine = 0).

After L2 normalisation the similarity score adjusts for vector length — a
record with many blocking tokens does not dominate a shorter one simply because
it has more 1-bits.  This mirrors the tf-idf intuition: normalisation controls
for document length.  The result is a similarity measure that faithfully
captures *relative* token overlap, which is exactly what entity resolution
blocking needs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse
from sklearn.preprocessing import normalize

from ..utils.config import NormalizationConfig
from ..utils.logging_utils import get_logger
from ..utils.profiling import profile_block

logger = get_logger(__name__)


@dataclass
class NormalizedVectors:
    """Container returned by normalize_vectors.

    Attributes
    ----------
    record_ids : list[str]
        Record identifiers in the row order of the matrix.  The i-th entry of
        record_ids corresponds to the i-th row of both sparse_matrix and
        dense_matrix.
    sparse_matrix : scipy.sparse.csr_matrix, shape (n, vocab_size)
        L2-normalised multi-hot vectors preserved in sparse format.  Used for
        exact cosine similarity refinement (Phase 5).
    dense_matrix : np.ndarray, shape (n, vocab_size), dtype float32
        Same data as a contiguous C-order float32 array.  FAISS requires dense
        float32 input; we materialise it once here and reuse across phases.
    vocab_size : int
    n_records : int
    """

    record_ids: List[str]
    sparse_matrix: scipy.sparse.csr_matrix
    dense_matrix: np.ndarray
    vocab_size: int
    n_records: int


def aggregate_record_vectors(
    vectors: Dict[str, scipy.sparse.csr_matrix],
) -> Tuple[List[str], scipy.sparse.csr_matrix]:
    """Collapse per-token matrices into one multi-hot vector per record.

    The input from BB10 maps each refID to a (k, vocab_size) sparse matrix
    where each of the k rows is the one-hot vector for one blocking token.
    Summing the rows gives a multi-hot row vector that encodes *which* blocking
    tokens the record contains, regardless of how many.

    Empty records (k == 0) produce an all-zero row, which after normalisation
    becomes a zero vector.  Zero vectors have undefined cosine similarity; they
    are kept in the index to preserve row/index alignment but will never be
    the top match for any query with non-zero norms.

    Parameters
    ----------
    vectors : dict {refID: csr_matrix(k, vocab_size)}

    Returns
    -------
    record_ids : list[str]
        Deterministic order (sorted refIDs).
    aggregated  : csr_matrix, shape (n_records, vocab_size)
    """
    if not vectors:
        raise ValueError("vectors dict is empty — nothing to aggregate.")

    # Stable ordering makes downstream index alignment deterministic.
    record_ids: List[str] = sorted(vectors.keys())

    # Peek at vocab_size from the first matrix.
    first = next(iter(vectors.values()))
    vocab_size = first.shape[1]

    rows = []
    for rid in record_ids:
        mat = vectors[rid]  # shape (k, vocab_size)
        if mat.shape[0] == 0:
            # No qualifying blocking tokens; insert explicit zero row.
            rows.append(scipy.sparse.csr_matrix((1, vocab_size), dtype=np.float32))
        else:
            # Sum token rows → multi-hot row vector; clip to binary 0/1.
            summed = mat.sum(axis=0)          # (1, vocab_size) np.matrix
            summed = scipy.sparse.csr_matrix(summed, dtype=np.float32)
            # Clip values > 1 to 1 to keep binary encoding semantics.
            summed.data = np.clip(summed.data, 0, 1)
            rows.append(summed)

    aggregated = scipy.sparse.vstack(rows, format="csr")
    logger.info(
        "Aggregated %d records into multi-hot matrix of shape %s "
        "(nnz=%d, density=%.4f%%)",
        len(record_ids),
        aggregated.shape,
        aggregated.nnz,
        100.0 * aggregated.nnz / max(aggregated.shape[0] * aggregated.shape[1], 1),
    )
    return record_ids, aggregated


def normalize_vectors(
    vectors: Dict[str, scipy.sparse.csr_matrix],
    config: NormalizationConfig | None = None,
) -> NormalizedVectors:
    """Phase 2 entry point: aggregate + L2-normalise sparse vectors.

    Steps
    -----
    1. Aggregate per-token matrices into one multi-hot vector per record.
    2. Apply sklearn's L2 normalisation in-place on the sparse matrix so that
       the sparse structure is preserved (sklearn normalise with sparse input
       returns a sparse matrix).
    3. Convert to a dense float32 array for FAISS.

    Parameters
    ----------
    vectors : dict {refID: csr_matrix(k, vocab_size)}
        Output of BB10_OneHotEncoding.build_one_hot_vectors.
    config  : NormalizationConfig | None

    Returns
    -------
    NormalizedVectors
    """
    if config is None:
        config = NormalizationConfig()

    with profile_block("Phase 2 — vector aggregation + L2 normalisation"):
        record_ids, aggregated = aggregate_record_vectors(vectors)
        vocab_size = aggregated.shape[1]

        # sklearn.preprocessing.normalize preserves sparsity when given a
        # sparse matrix.  norm='l2' divides each row by its Euclidean norm.
        # Rows with zero norm (empty records) are left as-is (all-zero).
        sparse_normalised: scipy.sparse.csr_matrix = normalize(
            aggregated, norm="l2", copy=True
        )

        # Materialise dense float32 for FAISS; use C-contiguous layout.
        dense = np.ascontiguousarray(sparse_normalised.toarray(), dtype=np.float32)

    n_records = len(record_ids)
    logger.info(
        "Normalisation complete: %d records × %d features | "
        "dense matrix %.2f MiB",
        n_records,
        vocab_size,
        dense.nbytes / (1024 ** 2),
    )

    return NormalizedVectors(
        record_ids=record_ids,
        sparse_matrix=sparse_normalised,
        dense_matrix=dense,
        vocab_size=vocab_size,
        n_records=n_records,
    )
