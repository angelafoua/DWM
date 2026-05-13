"""
Unit tests for Phase 6 — similarity graph construction.
"""

import unittest

import networkx as nx

from ..similarity.cosine import RefinedEdge, RefinementResult
from ..graph.builder import build_similarity_graph, GraphStats
from ..utils.config import GraphConfig


def _make_refinement(edges):
    """Helper: create a RefinementResult from a list of (src, dst, score) tuples."""
    refined_edges = [
        RefinedEdge(src_idx=s, dst_idx=d, ann_score=sc, cosine_score=sc)
        for s, d, sc in edges
    ]
    return RefinementResult(
        edges=refined_edges,
        n_candidates_in=len(edges),
        n_edges_out=len(edges),
        threshold=0.5,
    )


class TestGraphBuilder(unittest.TestCase):

    def setUp(self):
        self.record_ids = ["A", "B", "C", "D", "E"]
        # Edges: A-B, B-C (A and C in same component); D isolated; A-B duplicate.
        self.refinement = _make_refinement([
            (0, 1, 0.9),   # A-B
            (1, 2, 0.85),  # B-C
            (0, 1, 0.92),  # A-B duplicate (higher score should win)
        ])

    def test_returns_graph_and_stats(self):
        G, stats = build_similarity_graph(self.refinement, self.record_ids)
        self.assertIsInstance(G, nx.Graph)
        self.assertIsInstance(stats, GraphStats)

    def test_all_nodes_present(self):
        G, _ = build_similarity_graph(self.refinement, self.record_ids)
        for rid in self.record_ids:
            self.assertIn(rid, G.nodes)

    def test_deduplication_keeps_higher_score(self):
        G, _ = build_similarity_graph(self.refinement, self.record_ids)
        self.assertAlmostEqual(G["A"]["B"]["cosine_similarity"], 0.92, places=5)

    def test_edge_count_after_dedup(self):
        G, stats = build_similarity_graph(self.refinement, self.record_ids)
        # 2 unique edges: A-B and B-C.
        self.assertEqual(G.number_of_edges(), 2)
        self.assertEqual(stats.n_edges, 2)

    def test_isolated_node_in_graph(self):
        G, _ = build_similarity_graph(self.refinement, self.record_ids)
        # D (index 3) and E (index 4) have no edges → degree 0.
        self.assertEqual(G.degree("D"), 0)
        self.assertEqual(G.degree("E"), 0)

    def test_components(self):
        G, stats = build_similarity_graph(self.refinement, self.record_ids)
        # {A, B, C}, {D}, {E} → 3 components.
        self.assertEqual(stats.n_components, 3)
        self.assertEqual(stats.largest_component, 3)

    def test_edge_attributes(self):
        G, _ = build_similarity_graph(self.refinement, self.record_ids)
        data = G["A"]["B"]
        self.assertIn("cosine_similarity", data)
        self.assertIn("ann_score", data)
        self.assertIn("weight", data)
