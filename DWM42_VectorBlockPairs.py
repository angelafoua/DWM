#!/usr/bin/env python
# coding: utf-8

"""
DWM42_VectorBlockPairs — Vector-based candidate pair generation.

This module is a drop-in replacement for DWM42_BuildBlockPairs.buildBlockPairs.
It produces an identical output: a deduplicated, sorted list of candidate
pairs in "refID_A|refID_B" string format, ready to be consumed by DWM55.

The difference is in HOW those pairs are discovered:

  DWM42 (classic)
  ---------------
  Two records become a candidate pair only if they share the exact same
  blocking-token string.  Blocking is binary: same string → same block,
  different string → never paired.

  DWM42_VectorBlockPairs (this module)
  -------------------------------------
  Blocking tokens are encoded as one-hot vectors and aggregated into a
  multi-hot record vector.  An HNSW approximate nearest-neighbour index
  finds records whose blocking-token vectors are close in cosine space.
  A configurable similarity threshold replaces the binary exact-match rule.
  This means records with similar-but-not-identical blocking tokens (e.g.
  slight spelling variants) can still become candidate pairs.

  Everything downstream (DWM55 scoring, DWM80 transitive closure) is
  unchanged: pairs are handed off as the same list[str] format.

Pipeline stages executed here
------------------------------
  Stage 1  Select blocking tokens per record          (same filters as DWM42)
  Stage 2  Build one-hot vectors                      (BB10_OneHotEncoding)
  Stage 3  Aggregate to multi-hot + L2-normalise      (betterBlocking normalizer)
  Stage 4  Build HNSW index                           (betterBlocking ANNIndex)
  Stage 5  Retrieve approximate neighbours            (betterBlocking retrieval)
  Stage 6  Exact cosine refinement + threshold filter (betterBlocking cosine)
  Stage 7  Convert (src_idx, dst_idx) → "A|B" strings and deduplicate

Parameters read from DWM10_Parms
----------------------------------
  beta                   — maximum blocking token frequency (same as DWM42)
  minBlkTokenLen         — minimum token length           (same as DWM42)
  excludeNumericBlocks   — drop all-digit tokens          (same as DWM42)
  vectorBlockThreshold   — cosine similarity threshold for candidate pairs
                           (new; replaces the implicit 1.0 of exact matching)
                           Default: 0.5 if the attribute is absent.
  vectorBlockTopK        — how many ANN neighbours to retrieve per record
                           (new)  Default: 10 if the attribute is absent.
"""

import sys
import DWM10_Parms
from betterBlocking.BB10_OneHotEncoding import build_one_hot_vectors
from betterBlocking.ann.normalizer import normalize_vectors
from betterBlocking.ann.index import ANNIndex
from betterBlocking.ann.retrieval import retrieve_neighbors
from betterBlocking.similarity.cosine import refine_candidates
from betterBlocking.utils.config import (
    ANNConfig,
    NormalizationConfig,
    SimilarityConfig,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_parm(name, default):
    """Read an optional parameter from DWM10_Parms, falling back to default."""
    return getattr(DWM10_Parms, name, default)


def _pairs_from_refinement(refinement_result, record_ids):
    """
    Convert RefinedEdge objects into DWM55-compatible "refID_A|refID_B" strings.

    Rules (identical to DWM42):
      - The smaller refID always comes first (lexicographic order).
      - Duplicate pairs (same pair reached from both directions) are removed.
      - The final list is sorted.

    Parameters
    ----------
    refinement_result : RefinementResult
        Output of betterBlocking.similarity.cosine.refine_candidates.
    record_ids : list[str]
        Maps integer row indices back to string refIDs.

    Returns
    -------
    list[str]
        Deduplicated, sorted candidate pair strings.
    """
    seen = set()
    pair_list = []

    for edge in refinement_result.edges:
        id_a = record_ids[edge.src_idx]
        id_b = record_ids[edge.dst_idx]

        # Enforce consistent ordering (same convention as DWM42 lines 108-110)
        if id_a > id_b:
            id_a, id_b = id_b, id_a

        key = f"{id_a}|{id_b}"
        if key not in seen:
            seen.add(key)
            pair_list.append(key)

    pair_list.sort()
    return pair_list


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def buildVectorBlockPairs(refDict, linkIndex, tokenFreqDict):
    """
    Build candidate record pairs using vector-based approximate blocking.

    Signature and return value are identical to DWM42.buildBlockPairs so that
    the two functions are interchangeable in the driver.

    Parameters
    ----------
    refDict       : dict {refID: list[str]}   — tokenised records
    linkIndex     : dict {refID: list}         — already-linked records are
                                                 skipped (same logic as DWM42)
    tokenFreqDict : dict {token: int}          — corpus frequency per token

    Returns
    -------
    list[str]
        Deduplicated, sorted candidate pairs: ["refID_A|refID_B", ...]
        Ready to pass directly to DWM55.linkBlockPairs().
    """
    logFile = DWM10_Parms.logFile

    print('\n>>Starting DWM42_VectorBlockPairs')
    print('\n>>Starting DWM42_VectorBlockPairs', file=logFile)

    # ------------------------------------------------------------------
    # Read parameters (DWM42-compatible ones + new vector-specific ones)
    # ------------------------------------------------------------------
    beta                 = DWM10_Parms.beta
    minBlkTokenLen       = DWM10_Parms.minBlkTokenLen
    excludeNumericBlocks = DWM10_Parms.excludeNumericBlocks

    # New parameters — fall back to sensible defaults if not in parms file
    cosine_threshold = _read_parm('vectorBlockThreshold', 0.5)
    top_k            = _read_parm('vectorBlockTopK', 10)

    print(f'beta = {beta}')
    print(f'beta = {beta}', file=logFile)
    print(f'min blocking token length = {minBlkTokenLen}')
    print(f'min blocking token length = {minBlkTokenLen}', file=logFile)
    print(f'exclude numeric blocking tokens = {excludeNumericBlocks}')
    print(f'exclude numeric blocking tokens = {excludeNumericBlocks}', file=logFile)
    print(f'vector cosine threshold = {cosine_threshold}')
    print(f'vector cosine threshold = {cosine_threshold}', file=logFile)
    print(f'ANN top-k neighbours = {top_k}')
    print(f'ANN top-k neighbours = {top_k}', file=logFile)

    # ------------------------------------------------------------------
    # Stage 1 — restrict to unlinked records (mirrors DWM42 lines 39-41)
    # ------------------------------------------------------------------
    unlinked_refDict = {
        key: refDict[key]
        for key in linkIndex
        if len(linkIndex[key]) == 0 and key in refDict
    }

    selectCnt = len(unlinked_refDict)
    print(f'Total Records Selected for Reprocessing = {selectCnt}')
    print(f'Total Records Selected for Reprocessing = {selectCnt}', file=logFile)

    if selectCnt == 0:
        print('No unlinked records — returning empty pair list.')
        print('No unlinked records — returning empty pair list.', file=logFile)
        return []

    # ------------------------------------------------------------------
    # Stage 2 — build one-hot vectors from blocking tokens
    # BB10_OneHotEncoding applies the same three token filters as DWM42:
    #   length >= minBlkTokenLen, not all-digit, 2 <= freq <= beta
    # ------------------------------------------------------------------
    print('Stage 2: building one-hot vectors...')
    print('Stage 2: building one-hot vectors...', file=logFile)

    vocab, vectors = build_one_hot_vectors(
        unlinked_refDict,
        tokenFreqDict,
        minBlkTokenLen=minBlkTokenLen,
        excludeNumericBlocks=excludeNumericBlocks,
        beta=beta,
    )

    vocab_size = len(vocab)
    print(f'Vocabulary size (blocking token dimensions) = {vocab_size}')
    print(f'Vocabulary size (blocking token dimensions) = {vocab_size}', file=logFile)

    if vocab_size == 0:
        print('No qualifying blocking tokens found — returning empty pair list.')
        print('No qualifying blocking tokens found — returning empty pair list.', file=logFile)
        return []

    # Count how many records have at least one blocking token
    records_with_tokens = sum(1 for m in vectors.values() if m.shape[0] > 0)
    print(f'Records with at least one blocking token = {records_with_tokens}')
    print(f'Records with at least one blocking token = {records_with_tokens}', file=logFile)

    # ------------------------------------------------------------------
    # Stage 3 — aggregate per-token rows into one multi-hot vector per
    #           record, then L2-normalise for cosine similarity via dot
    #           product in the HNSW index
    # ------------------------------------------------------------------
    print('Stage 3: aggregating and normalising vectors...')
    print('Stage 3: aggregating and normalising vectors...', file=logFile)

    normalized = normalize_vectors(vectors, NormalizationConfig())

    # ------------------------------------------------------------------
    # Stage 4 — build HNSW approximate nearest-neighbour index
    # ------------------------------------------------------------------
    print('Stage 4: building HNSW index...')
    print('Stage 4: building HNSW index...', file=logFile)

    ann_cfg = ANNConfig(top_k=top_k)
    ann_index = ANNIndex(ann_cfg)
    ann_index.build(normalized.dense_matrix)

    # ------------------------------------------------------------------
    # Stage 5 — retrieve approximate neighbours for every record
    # ------------------------------------------------------------------
    print('Stage 5: retrieving approximate neighbours...')
    print('Stage 5: retrieving approximate neighbours...', file=logFile)

    neighbor_result = retrieve_neighbors(ann_index, normalized, ann_cfg)

    print(f'Total ANN candidate pairs (before refinement) = {neighbor_result.n_candidates}')
    print(f'Total ANN candidate pairs (before refinement) = {neighbor_result.n_candidates}', file=logFile)

    # ------------------------------------------------------------------
    # Stage 6 — exact cosine refinement: recompute cosine similarity for
    #           each ANN candidate and keep only pairs >= cosine_threshold.
    #           This removes ANN false positives.
    # ------------------------------------------------------------------
    print(f'Stage 6: exact cosine refinement at threshold = {cosine_threshold}...')
    print(f'Stage 6: exact cosine refinement at threshold = {cosine_threshold}...', file=logFile)

    sim_cfg = SimilarityConfig(threshold=cosine_threshold)
    refinement_result = refine_candidates(neighbor_result, normalized, sim_cfg)

    print(f'Pairs surviving refinement = {refinement_result.n_edges_out}')
    print(f'Pairs surviving refinement = {refinement_result.n_edges_out}', file=logFile)
    print(f'Mean ANN approximation error = {refinement_result.mean_ann_error:.4f}')
    print(f'Mean ANN approximation error = {refinement_result.mean_ann_error:.4f}', file=logFile)

    # ------------------------------------------------------------------
    # Stage 7 — convert (src_idx, dst_idx) integer edges to the
    #           "refID_A|refID_B" string format that DWM55 expects,
    #           then deduplicate and sort
    # ------------------------------------------------------------------
    block_pair_list = _pairs_from_refinement(
        refinement_result,
        normalized.record_ids,
    )

    print(f'Total Unduplicated Pairs = {len(block_pair_list)}')
    print(f'Total Unduplicated Pairs = {len(block_pair_list)}', file=logFile)

    return block_pair_list
