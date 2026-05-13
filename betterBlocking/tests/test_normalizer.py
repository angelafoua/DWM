"""
Unit tests for Phase 2 — vector aggregation and L2 normalisation.
"""

import unittest

import numpy as np
import scipy.sparse

from ..ann.normalizer import aggregate_record_vectors, normalize_vectors, NormalizedVectors
from ..utils.config import NormalizationConfig


def _make_vectors(vocab_size: int = 5):
    """Create a minimal synthetic vectors dict matching BB10 output format."""
    # Record A: tokens at columns 0 and 2.
    vec_A = scipy.sparse.csr_matrix(
        ([1, 1], ([0, 1], [0, 2])), shape=(2, vocab_size), dtype=int
    )
    # Record B: token at column 1 only.
    vec_B = scipy.sparse.csr_matrix(
        ([1], ([0], [1])), shape=(1, vocab_size), dtype=int
    )
    # Record C: empty (no qualifying tokens).
    vec_C = scipy.sparse.csr_matrix((0, vocab_size), dtype=int)
    return {"A": vec_A, "B": vec_B, "C": vec_C}


class TestAggregation(unittest.TestCase):

    def setUp(self):
        self.vectors = _make_vectors()

    def test_output_shape(self):
        record_ids, agg = aggregate_record_vectors(self.vectors)
        self.assertEqual(agg.shape, (3, 5))   # 3 records × 5 vocab
        self.assertEqual(len(record_ids), 3)

    def test_deterministic_order(self):
        record_ids, _ = aggregate_record_vectors(self.vectors)
        self.assertEqual(record_ids, ["A", "B", "C"])

    def test_record_A_multi_hot(self):
        record_ids, agg = aggregate_record_vectors(self.vectors)
        idx = record_ids.index("A")
        row = agg[idx].toarray().ravel()
        # Columns 0 and 2 should be 1; clipped to 1 even if summed > 1.
        self.assertEqual(row[0], 1.0)
        self.assertEqual(row[2], 1.0)
        self.assertEqual(row[1], 0.0)

    def test_empty_record_is_all_zero(self):
        record_ids, agg = aggregate_record_vectors(self.vectors)
        idx = record_ids.index("C")
        row = agg[idx].toarray().ravel()
        np.testing.assert_array_equal(row, np.zeros(5))

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            aggregate_record_vectors({})


class TestNormalization(unittest.TestCase):

    def setUp(self):
        self.vectors = _make_vectors()

    def test_returns_normalized_vectors(self):
        result = normalize_vectors(self.vectors)
        self.assertIsInstance(result, NormalizedVectors)

    def test_unit_norm_for_non_empty_records(self):
        result = normalize_vectors(self.vectors)
        dense = result.dense_matrix
        # Records A and B should have unit L2 norm.
        for i, rid in enumerate(result.record_ids):
            if rid == "C":
                continue  # zero vector has zero norm
            norm = float(np.linalg.norm(dense[i]))
            self.assertAlmostEqual(norm, 1.0, places=5,
                                   msg=f"Record {rid} norm={norm}")

    def test_zero_vector_not_nan(self):
        result = normalize_vectors(self.vectors)
        idx = result.record_ids.index("C")
        row = result.dense_matrix[idx]
        self.assertFalse(np.any(np.isnan(row)))

    def test_sparse_matrix_preserved(self):
        result = normalize_vectors(self.vectors)
        self.assertIsInstance(result.sparse_matrix, scipy.sparse.csr_matrix)

    def test_dense_dtype_float32(self):
        result = normalize_vectors(self.vectors)
        self.assertEqual(result.dense_matrix.dtype, np.float32)

    def test_dot_product_equals_cosine(self):
        """After L2 norm, dot product of two rows equals their cosine similarity."""
        from sklearn.metrics.pairwise import cosine_similarity as sklearn_cos

        result = normalize_vectors(self.vectors)
        dense = result.dense_matrix

        # Use records A and B (both non-zero).
        idx_A = result.record_ids.index("A")
        idx_B = result.record_ids.index("B")

        dot = float(np.dot(dense[idx_A], dense[idx_B]))
        cos = float(sklearn_cos(
            result.sparse_matrix[idx_A], result.sparse_matrix[idx_B]
        )[0, 0])
        self.assertAlmostEqual(dot, cos, places=5)
