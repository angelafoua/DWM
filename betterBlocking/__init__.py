"""
betterBlocking — ANN-based Entity Resolution blocking library.

Phases
------
Phase 1  BB10_OneHotEncoding   — sparse one-hot vector construction (pre-existing)
Phase 2  ann.normalizer        — L2 normalisation
Phase 3  ann.index             — FAISS HNSW index
Phase 4  ann.retrieval         — ANN neighbour retrieval
Phase 5  similarity.cosine     — exact cosine refinement
Phase 6  graph.builder         — similarity graph
Phase 7  clustering.extractor  — cluster extraction
Phase 8  clustering.transitivity — transitivity analysis
Phase 9  visualization.plots   — matplotlib artefacts
Phase 10 utils.profiling       — runtime/memory profiling
Phase 11 BB_Pipeline           — end-to-end orchestrator
"""
