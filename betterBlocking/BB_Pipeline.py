"""
ANN-based Entity Resolution Pipeline — Main Orchestrator.

This module ties together Phases 2-12 into a single callable pipeline.
It is designed to be imported programmatically or executed as a script.

Pipeline stages
---------------
Phase 2  normalizer.normalize_vectors
Phase 3  ANNIndex.build
Phase 4  retrieve_neighbors
Phase 5  refine_candidates
Phase 6  build_similarity_graph
Phase 7  extract_clusters
Phase 8  analyze_transitivity
Phase 9  save_all_artifacts
Phase 10 runtime + memory profiling (woven into each stage)

Complexity overview (Phase 10)
-------------------------------
| Stage              | Time complexity      | Space complexity        |
|--------------------|----------------------|-------------------------|
| Normalisation      | O(n × d_nnz)        | O(n × d_nnz) sparse     |
| HNSW build         | O(n × M × log n × d)| O(n × M × 2 + n × d)   |
| ANN retrieval      | O(n × ef × log n × d)| O(n × k) for results   |
| Exact refinement   | O(n × k × d_nnz)   | O(k × d_nnz) per batch  |
| Graph build        | O(n × k)            | O(n + e)                |
| Connected comps.   | O(n + e)            | O(n)                    |
| Transitivity       | O(clusters × k²)   | O(k²) per cluster       |

Where:
  n     = number of records
  d     = vocabulary size (vector dimension)
  d_nnz = average non-zeros per vector (≪ d for sparse vectors)
  k     = top_k neighbours
  M     = HNSW connectivity parameter
  e     = number of graph edges
  ef    = efSearch parameter

Key bottleneck: if the similarity threshold is low, the number of edges e
grows rapidly and graph operations dominate.  Raise the threshold or reduce
top_k to control memory and time for very large datasets.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx
import scipy.sparse

from .ann.normalizer import normalize_vectors, NormalizedVectors
from .ann.index import ANNIndex
from .ann.retrieval import retrieve_neighbors, NeighborResult
from .similarity.cosine import refine_candidates, RefinementResult
from .graph.builder import build_similarity_graph, GraphStats
from .clustering.extractor import extract_clusters, ClusteringResult
from .clustering.transitivity import analyze_transitivity, TransitivityReport
from .visualization.plots import save_all_artifacts
from .utils.config import PipelineConfig
from .utils.logging_utils import get_logger
from .utils.profiling import profile_block

logger = get_logger(__name__)


@dataclass
class PipelineOutput:
    """All outputs produced by the pipeline.

    Attributes
    ----------
    normalized          : NormalizedVectors
    neighbor_result     : NeighborResult
    refinement_result   : RefinementResult
    graph               : nx.Graph
    graph_stats         : GraphStats
    clustering_result   : ClusteringResult
    transitivity_report : TransitivityReport
    artifact_paths      : list[str] — paths to visualisation PNGs.
    total_elapsed_s     : float — wall-clock seconds for the full pipeline.
    """

    normalized: Optional[NormalizedVectors] = None
    neighbor_result: Optional[NeighborResult] = None
    refinement_result: Optional[RefinementResult] = None
    graph: Optional[nx.Graph] = None
    graph_stats: Optional[GraphStats] = None
    clustering_result: Optional[ClusteringResult] = None
    transitivity_report: Optional[TransitivityReport] = None
    artifact_paths: List[str] = field(default_factory=list)
    total_elapsed_s: float = 0.0


def run_pipeline(
    vectors: Dict[str, scipy.sparse.csr_matrix],
    config: PipelineConfig | None = None,
    visualize: bool = True,
) -> PipelineOutput:
    """Execute the full ANN-ER pipeline.

    Parameters
    ----------
    vectors   : dict {refID: csr_matrix(k, vocab_size)}
                Output of BB10_OneHotEncoding.build_one_hot_vectors.
                Vectors are NOT regenerated inside this function.
    config    : PipelineConfig | None — uses defaults if None.
    visualize : bool — whether to produce and save visualisation artefacts.

    Returns
    -------
    PipelineOutput
    """
    if config is None:
        config = PipelineConfig()

    output = PipelineOutput()
    t_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("ANN Entity Resolution Pipeline — starting")
    logger.info("n_records=%d | threshold=%.2f | top_k=%d | M=%d",
                len(vectors),
                config.similarity.threshold,
                config.ann.top_k,
                config.ann.M)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Phase 2: Normalisation
    # ------------------------------------------------------------------
    logger.info("--- Phase 2: Vector normalisation ---")
    normalized = normalize_vectors(vectors, config.normalization)
    output.normalized = normalized

    # ------------------------------------------------------------------
    # Phase 3: HNSW index construction
    # ------------------------------------------------------------------
    logger.info("--- Phase 3: HNSW index build ---")
    ann_index = ANNIndex(config.ann)
    ann_index.build(normalized.dense_matrix)

    # ------------------------------------------------------------------
    # Phase 4: ANN retrieval
    # ------------------------------------------------------------------
    logger.info("--- Phase 4: ANN retrieval ---")
    neighbor_result = retrieve_neighbors(ann_index, normalized, config.ann)
    output.neighbor_result = neighbor_result

    # ------------------------------------------------------------------
    # Phase 5: Exact cosine refinement
    # ------------------------------------------------------------------
    logger.info("--- Phase 5: Exact cosine refinement ---")
    refinement_result = refine_candidates(neighbor_result, normalized, config.similarity)
    output.refinement_result = refinement_result

    # ------------------------------------------------------------------
    # Phase 6: Similarity graph
    # ------------------------------------------------------------------
    logger.info("--- Phase 6: Graph construction ---")
    G, graph_stats = build_similarity_graph(
        refinement_result, normalized.record_ids, config.graph
    )
    output.graph = G
    output.graph_stats = graph_stats

    # ------------------------------------------------------------------
    # Phase 7: Cluster extraction
    # ------------------------------------------------------------------
    logger.info("--- Phase 7: Cluster extraction ---")
    clustering_result = extract_clusters(G, config.clustering)
    output.clustering_result = clustering_result

    # ------------------------------------------------------------------
    # Phase 8: Transitivity analysis
    # ------------------------------------------------------------------
    logger.info("--- Phase 8: Transitivity analysis ---")
    transitivity_report = analyze_transitivity(
        G, clustering_result, normalized, config.clustering
    )
    output.transitivity_report = transitivity_report

    # ------------------------------------------------------------------
    # Phase 9: Visualisation
    # ------------------------------------------------------------------
    if visualize and refinement_result.n_edges_out > 0:
        logger.info("--- Phase 9: Visualisation ---")
        paths = save_all_artifacts(
            G, clustering_result, refinement_result, normalized,
            config.visualization,
        )
        output.artifact_paths = paths

    output.total_elapsed_s = time.perf_counter() - t_start

    logger.info("=" * 60)
    logger.info("Pipeline complete in %.2fs", output.total_elapsed_s)
    logger.info(str(clustering_result))
    logger.info("Artifacts: %s", output.artifact_paths)
    logger.info("=" * 60)

    return output


def print_cluster_report(output: PipelineOutput) -> None:
    """Print a human-readable cluster report to stdout."""
    cr = output.clustering_result
    if cr is None:
        print("No clustering result available.")
        return

    print(f"\n{'='*60}")
    print("ENTITY RESOLUTION CLUSTER REPORT")
    print(f"{'='*60}")
    print(f"Total clusters     : {cr.n_clusters}")
    print(f"Singleton clusters : {cr.n_singletons}")
    print(f"Multi-record       : {cr.n_non_singletons}")
    print(f"Largest cluster    : {cr.largest_cluster}")
    print(f"Average size       : {cr.avg_cluster_size:.2f}")
    print(f"Pipeline time      : {output.total_elapsed_s:.2f}s")
    print(f"{'='*60}")

    print("\nTop 10 largest clusters:")
    for c in cr.clusters[:10]:
        print(f"  Cluster {c.cluster_id:>5}: {c.size:>4} records → {c.members[:5]}"
              + (" ..." if c.size > 5 else ""))

    tr = output.transitivity_report
    if tr and tr.n_flagged > 0:
        print(f"\nTransitivity warnings: {tr.n_flagged} suspicious clusters")
        for sus in tr.suspicious_clusters[:5]:
            print(f"  Cluster {sus.cluster_id}: variance={sus.sim_variance:.3f} "
                  f"min_sim={sus.min_pairwise_sim:.3f} "
                  f"chains={len(sus.suspected_chains)}")
    print()


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import re
    import sys
    import os

    # Ensure the repo root (parent of betterBlocking/) is on sys.path so that
    # both `python -m betterBlocking.BB_Pipeline` and direct script execution
    # can resolve absolute imports.
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    # Absolute imports — relative imports are not permitted inside __main__.
    from betterBlocking.BB10_OneHotEncoding import build_one_hot_vectors
    from betterBlocking.utils.config import (
        PipelineConfig, ANNConfig, SimilarityConfig, VisualizationConfig
    )

    parser = argparse.ArgumentParser(
        description="ANN-based Entity Resolution Pipeline (Phases 2-12)"
    )
    parser.add_argument("inputFile", help="CSV input file (refID in first column)")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--hasHeader", action="store_true")
    parser.add_argument("--beta", type=int, default=2)
    parser.add_argument("--minLen", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Cosine similarity threshold (default 0.80)")
    parser.add_argument("--topK", type=int, default=10)
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--efConstruction", type=int, default=200)
    parser.add_argument("--efSearch", type=int, default=128)
    parser.add_argument("--outputDir", default="artifacts")
    parser.add_argument("--noViz", action="store_true", help="Skip visualisation")
    args = parser.parse_args()

    # ---- Build vectors (Phase 1 — already done, replayed here for CLI use) ----
    refDict: Dict[str, List[str]] = {}
    with open(args.inputFile) as fh:
        if args.hasHeader:
            fh.readline()
        for line in fh:
            parts = line.strip().split(args.delimiter)
            if not parts or not parts[0]:
                continue
            refID = parts[0]
            body = " ".join(parts[1:])
            tokens = re.sub(r"\W", " ", body.lower()).split()
            refDict[refID] = tokens

    tokenFreqDict: Dict[str, int] = {}
    for tokens in refDict.values():
        for t in tokens:
            tokenFreqDict[t] = tokenFreqDict.get(t, 0) + 1

    vocab, vectors = build_one_hot_vectors(
        refDict, tokenFreqDict,
        minBlkTokenLen=args.minLen,
        excludeNumericBlocks=True,
        beta=args.beta,
    )
    print(f"Vocabulary size: {len(vocab)} | Records: {len(vectors)}")

    # ---- Configure and run pipeline ----
    cfg = PipelineConfig(
        ann=ANNConfig(
            M=args.M,
            efConstruction=args.efConstruction,
            efSearch=args.efSearch,
            top_k=args.topK,
        ),
        similarity=SimilarityConfig(threshold=args.threshold),
        visualization=VisualizationConfig(output_dir=args.outputDir),
    )

    out = run_pipeline(vectors, cfg, visualize=not args.noViz)
    print_cluster_report(out)
