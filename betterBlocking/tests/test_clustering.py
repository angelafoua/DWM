"""
Unit tests for Phase 7 (cluster extraction) and Phase 8 (transitivity).
"""

import unittest

import networkx as nx
import numpy as np
import scipy.sparse

from ..clustering.extractor import extract_clusters, ClusteringResult, Cluster
from ..clustering.transitivity import analyze_transitivity, TransitivityReport
from ..ann.normalizer import normalize_vectors, NormalizedVectors
from ..similarity.cosine import RefinedEdge, RefinementResult
from ..graph.builder import build_similarity_graph
from ..utils.config import ClusteringConfig


def _build_graph_and_vectors():
    """
    Construct a graph with a deliberate transitivity chain:
      A–B (high sim 0.95)
      B–C (high sim 0.92)
      A–C has no direct edge but both share B → same component.

    Vectors are crafted so that A·C is low (they share fewer tokens than A·B
    or B·C) to trigger the transitivity warning.
    """
    vocab_size = 6
    # A: tokens 0,1   B: tokens 1,2   C: tokens 2,3
    # A and B share token 1 → similar
    # B and C share token 2 → similar
    # A and C share nothing → cosine = 0
    def _vec(cols):
        rows = list(range(len(cols)))
        return scipy.sparse.csr_matrix(
            ([1] * len(cols), (rows, cols)), shape=(len(cols), vocab_size), dtype=int
        )

    vectors = {
        "A": _vec([0, 1]),
        "B": _vec([1, 2]),
        "C": _vec([2, 3]),
        "D": _vec([4, 5]),   # isolated
    }
    normalized = normalize_vectors(vectors)

    # Build graph manually to control edges.
    G = nx.Graph()
    for rid in ["A", "B", "C", "D"]:
        G.add_node(rid)
    G.add_edge("A", "B", cosine_similarity=0.95, ann_score=0.94, weight=0.95)
    G.add_edge("B", "C", cosine_similarity=0.92, ann_score=0.91, weight=0.92)
    # No A–C edge.

    return G, normalized


class TestClusterExtraction(unittest.TestCase):

    def setUp(self):
        G, self.normalized = _build_graph_and_vectors()
        self.G = G

    def test_returns_clustering_result(self):
        result = extract_clusters(self.G)
        self.assertIsInstance(result, ClusteringResult)

    def test_correct_cluster_count(self):
        result = extract_clusters(self.G)
        # {A, B, C} and {D} → 2 clusters.
        self.assertEqual(result.n_clusters, 2)

    def test_singletons(self):
        result = extract_clusters(self.G)
        # D is a singleton.
        self.assertEqual(result.n_singletons, 1)

    def test_largest_cluster_size(self):
        result = extract_clusters(self.G)
        self.assertEqual(result.largest_cluster, 3)

    def test_cluster_map_complete(self):
        result = extract_clusters(self.G)
        for rid in ["A", "B", "C", "D"]:
            self.assertIn(rid, result.cluster_map)

    def test_members_in_same_cluster(self):
        result = extract_clusters(self.G)
        cid_A = result.cluster_map["A"]
        cid_B = result.cluster_map["B"]
        cid_C = result.cluster_map["C"]
        self.assertEqual(cid_A, cid_B)
        self.assertEqual(cid_B, cid_C)

    def test_singleton_in_different_cluster(self):
        result = extract_clusters(self.G)
        cid_A = result.cluster_map["A"]
        cid_D = result.cluster_map["D"]
        self.assertNotEqual(cid_A, cid_D)


class TestTransitivityAnalysis(unittest.TestCase):

    def setUp(self):
        self.G, self.normalized = _build_graph_and_vectors()
        self.clustering = extract_clusters(self.G)

    def test_returns_transitivity_report(self):
        report = analyze_transitivity(
            self.G, self.clustering, self.normalized
        )
        self.assertIsInstance(report, TransitivityReport)

    def test_suspicious_cluster_flagged(self):
        """The A-B-C chain should be flagged (A and C have zero cosine similarity)."""
        cfg = ClusteringConfig(
            intra_cluster_low_sim_threshold=0.5,
            transitivity_variance_threshold=0.01,
        )
        report = analyze_transitivity(
            self.G, self.clustering, self.normalized, cfg
        )
        # At least the A-B-C cluster should be flagged.
        self.assertGreater(report.n_flagged, 0)

    def test_chain_detected(self):
        """A–B–C is a transitivity chain (A–C not a direct edge)."""
        cfg = ClusteringConfig(
            intra_cluster_low_sim_threshold=0.5,
            transitivity_variance_threshold=0.01,
        )
        report = analyze_transitivity(
            self.G, self.clustering, self.normalized, cfg
        )
        all_chains = [
            chain
            for sus in report.suspicious_clusters
            for chain in sus.suspected_chains
        ]
        self.assertGreater(len(all_chains), 0,
                           "Expected at least one transitivity chain A-B-C")
