"""
Unit tests for Phase 3 (FAISS HNSW index) and Phase 4 (ANN retrieval).
"""

import unittest

import numpy as np
import scipy.sparse

from ..ann.normalizer import normalize_vectors
from ..ann.index import ANNIndex
from ..ann.retrieval import retrieve_neighbors, NeighborResult
from ..utils.config import ANNConfig


def _make_vectors(n: int = 20, vocab_size: int = 10, seed: int = 0):
    """Generate n random sparse binary vectors for testing."""
    rng = np.random.default_rng(seed)
    vectors = {}
    for i in range(n):
        # Each record gets 2-4 active tokens at random column positions.
        k = rng.integers(2, 5)
        cols = rng.choice(vocab_size, size=k, replace=False)
        rows = np.arange(k)
        mat = scipy.sparse.csr_matrix(
            (np.ones(k, dtype=int), (rows, cols)),
            shape=(k, vocab_size),
            dtype=int,
        )
        vectors[f"rec_{i:03d}"] = mat
    return vectors


class TestANNIndex(unittest.TestCase):

    def setUp(self):
        self.vectors = _make_vectors(n=20, vocab_size=10)
        self.normalized = normalize_vectors(self.vectors)
        self.cfg = ANNConfig(M=4, efConstruction=20, efSearch=10, top_k=5)
        self.index = ANNIndex(self.cfg)
        self.index.build(self.normalized.dense_matrix)

    def test_index_built(self):
        self.assertIsNotNone(self.index.index)
        self.assertEqual(self.index.n_vectors, 20)

    def test_search_returns_correct_shapes(self):
        q = self.normalized.dense_matrix[:3]
        distances, indices = self.index.search(q, top_k=5)
        self.assertEqual(distances.shape, (3, 5))
        self.assertEqual(indices.shape, (3, 5))

    def test_search_scores_bounded(self):
        q = self.normalized.dense_matrix
        distances, _ = self.index.search(q)
        # After L2 norm, inner product ∈ [-1, 1]; in practice ≥ 0 for binary.
        self.assertTrue(np.all(distances <= 1.01))

    def test_set_ef_search(self):
        self.index.set_ef_search(50)
        self.assertEqual(self.index.index.hnsw.efSearch, 50)


class TestANNRetrieval(unittest.TestCase):

    def setUp(self):
        self.vectors = _make_vectors(n=20, vocab_size=10)
        self.normalized = normalize_vectors(self.vectors)
        self.cfg = ANNConfig(M=4, efConstruction=20, efSearch=10, top_k=5, batch_size=7)
        self.index = ANNIndex(self.cfg)
        self.index.build(self.normalized.dense_matrix)

    def test_returns_neighbor_result(self):
        result = retrieve_neighbors(self.index, self.normalized, self.cfg)
        self.assertIsInstance(result, NeighborResult)

    def test_no_self_matches(self):
        result = retrieve_neighbors(self.index, self.normalized, self.cfg)
        for src_idx, neighbors in result.neighbors.items():
            nb_indices = [nb for nb, _ in neighbors]
            self.assertNotIn(src_idx, nb_indices,
                             f"Self-match found for record {src_idx}")

    def test_candidate_count_positive(self):
        result = retrieve_neighbors(self.index, self.normalized, self.cfg)
        self.assertGreater(result.n_candidates, 0)

    def test_record_ids_preserved(self):
        result = retrieve_neighbors(self.index, self.normalized, self.cfg)
        self.assertEqual(result.record_ids, self.normalized.record_ids)
