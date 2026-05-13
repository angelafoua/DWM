#!/usr/bin/env python
"""
Standalone entry point for the ANN Entity Resolution pipeline.

Run from anywhere:
    python run_er.py S12PX.txt --hasHeader --threshold 0.80

This avoids the package-import mechanics of `python -m betterBlocking.BB_Pipeline`
and works regardless of which directory you are in.
"""

import argparse
import os
import re
import sys

# Insert the directory containing this script so that `betterBlocking` is
# importable regardless of where the user runs the command from.
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from betterBlocking.BB10_OneHotEncoding import build_one_hot_vectors
from betterBlocking.BB_Pipeline import run_pipeline, print_cluster_report
from betterBlocking.utils.config import (
    PipelineConfig, ANNConfig, SimilarityConfig, VisualizationConfig,
)

parser = argparse.ArgumentParser(
    description="ANN-based Entity Resolution Pipeline"
)
parser.add_argument("inputFile",       help="CSV input file (refID in first column)")
parser.add_argument("--delimiter",     default=",")
parser.add_argument("--hasHeader",     action="store_true")
parser.add_argument("--beta",          type=int,   default=2)
parser.add_argument("--minLen",        type=int,   default=4)
parser.add_argument("--threshold",     type=float, default=0.80)
parser.add_argument("--topK",          type=int,   default=10)
parser.add_argument("--M",             type=int,   default=32)
parser.add_argument("--efConstruction",type=int,   default=200)
parser.add_argument("--efSearch",      type=int,   default=128)
parser.add_argument("--outputDir",     default="artifacts")
parser.add_argument("--noViz",         action="store_true")
args = parser.parse_args()

# ── Phase 1: tokenise ────────────────────────────────────────────────────────
refDict = {}
with open(args.inputFile) as fh:
    if args.hasHeader:
        fh.readline()
    for line in fh:
        parts = line.strip().split(args.delimiter)
        if not parts or not parts[0]:
            continue
        refID = parts[0]
        body  = " ".join(parts[1:])
        refDict[refID] = re.sub(r"\W", " ", body.lower()).split()

tokenFreqDict = {}
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

# ── Phases 2-12: pipeline ────────────────────────────────────────────────────
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
