"""
Unit tests for Phase 5 — exact cosine similarity refinement.
"""

import unittest

import numpy as np
import scipy.sparse

from ..ann.normalizer import normalize_vectors
from ..ann.index import ANNIndex
from ..ann.retrieval import retrieve_neighbors, NeighborResult
from ..similarity.cosine import refine_candidates, RefinedEdge, RefinementResult
from ..utils.config import ANNConfig, SimilarityConfig


def _build_pipeline(n: int = 20, vocab_size: int = 10, threshold: float = 0.1):
    rng = np.random.default_rng(1)
    vectors = {}
    for i in range(n):
        k = rng.integers(2, 5)
        cols = rng.choice(vocab_size, size=k, replace=False)
        rows = np.arange(k)
        mat = scipy.sparse.csr_matrix(
            (np.ones(k, dtype=int), (rows, cols)), shape=(k, vocab_size), dtype=int
        )
        vectors[f"r{i}"] = mat

    normalized = normalize_vectors(vectors)
    cfg = ANNConfig(M=4, efConstruction=20, efSearch=10, top_k=5, batch_size=8)
    index = ANNIndex(cfg)
    index.build(normalized.dense_matrix)
    neighbors = retrieve_neighbors(index, normalized, cfg)
    return normalized, neighbors, threshold


class TestRefinement(unittest.TestCase):

    def test_returns_refinement_result(self):
        normalized, neighbors, threshold = _build_pipeline()
        result = refine_candidates(
            neighbors, normalized, SimilarityConfig(threshold=threshold)
        )
        self.assertIsInstance(result, RefinementResult)

    def test_all_edges_above_threshold(self):
        normalized, neighbors, threshold = _build_pipeline(threshold=0.3)
        result = refine_candidates(
            neighbors, normalized, SimilarityConfig(threshold=threshold)
        )
        for edge in result.edges:
            self.assertGreaterEqual(edge.cosine_score, threshold)

    def test_no_duplicate_directed_pairs(self):
        normalized, neighbors, _ = _build_pipeline(threshold=0.0)
        result = refine_candidates(
            neighbors, normalized, SimilarityConfig(threshold=0.0)
        )
        seen = set()
        for edge in result.edges:
            pair = (edge.src_idx, edge.dst_idx)
            self.assertNotIn(pair, seen, f"Duplicate edge: {pair}")
            seen.add(pair)

    def test_ann_error_small(self):
        """ANN and exact scores should be close (inner product on normalised vecs)."""
        normalized, neighbors, _ = _build_pipeline(threshold=0.0)
        result = refine_candidates(
            neighbors, normalized, SimilarityConfig(threshold=0.0)
        )
        # Mean error should be tiny for small vocab (almost exact inner product).
        self.assertLess(result.mean_ann_error, 0.1)

    def test_candidate_count_bookkeeping(self):
        normalized, neighbors, _ = _build_pipeline(threshold=0.0)
        result = refine_candidates(
            neighbors, normalized, SimilarityConfig(threshold=0.0)
        )
        self.assertEqual(result.n_candidates_in, neighbors.n_candidates)
