#!/usr/bin/env python
"""
Parameter Sweep Runner for DWM Empirical Study (Paper 1).

Generates parameter combinations for beta, sigma, mu, epsilon and runs DWM
non-interactively for each combination, collecting Precision/Recall/F1 results.

Usage:
    # Single dataset sweep (uses default parameter grid)
    python parameter_sweep.py --data S7GX.txt --truth truthABCgoodDQ.txt

    # Custom grid
    python parameter_sweep.py --data S7GX.txt --truth truthABCgoodDQ.txt \
        --beta 2,5,10,15 --sigma 12,25,41 --mu 0.50:0.95:0.05 --epsilon 0.05:0.50:0.05

    # Batch mode: sweep multiple datasets
    python parameter_sweep.py --batch batch_sweep.txt

    # Resume an interrupted sweep
    python parameter_sweep.py --data S7GX.txt --truth truthABCgoodDQ.txt --resume

batch_sweep.txt format (one per line):
    S7GX.txt,truthABCgoodDQ.txt
    S8P.txt,truthABCpoorDQ.txt

Parameter range syntax:
    --mu 0.50:0.95:0.05    means start:stop:step (inclusive)
    --beta 2,5,10,15       means explicit list
"""

import argparse
import csv
import datetime
import io
import itertools
import os
import re
import sys
import time
import contextlib

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import importlib
import DWM10_Parms
import DWM14_BuildRefDict
import DWM15_BuildLinkIndex
import DWM16_BuildTokenFreqDict
import DWM42_BuildBlockPairs
import DWM55_LinkBlockPairs
import DWM80_TransitiveClosure
import DWM90_IterateClusters
import DWM99_ERmetrics


RESULT_COLUMNS = [
    "dataset",
    "beta",
    "sigma",
    "mu",
    "mu_iterate",
    "epsilon",
    "epsilon_iterate",
    "precision",
    "recall",
    "f1",
    "true_pairs",
    "linked_pairs",
    "expected_pairs",
    "num_records",
    "total_tokens",
    "unique_tokens",
    "token_freq_min",
    "token_freq_max",
    "token_freq_mean",
    "token_freq_std",
    "token_len_min",
    "token_len_max",
    "token_len_avg",
    "token_len_std",
    "runtime_seconds",
]


def parse_range(spec):
    """Parse '0.50:0.95:0.05' or '2,5,10,15' into a list of values."""
    if ":" in spec:
        parts = spec.split(":")
        start = float(parts[0])
        stop = float(parts[1])
        step = float(parts[2])
        vals = []
        v = start
        while v <= stop + 1e-9:
            vals.append(round(v, 4))
            v += step
        return vals
    else:
        return [float(x) for x in spec.split(",")]


def reset_parms():
    """Reset DWM10_Parms to defaults before each run."""
    DWM10_Parms.inputFileName = ""
    DWM10_Parms.delimiter = ","
    DWM10_Parms.hasHeader = False
    DWM10_Parms.tokenizerType = "Splitter"
    DWM10_Parms.truthFileName = ""
    DWM10_Parms.runIterationProfile = False
    DWM10_Parms.addRefsToLinkIndex = False
    DWM10_Parms.runGlobalCorrection = False
    DWM10_Parms.globalCorrectionDetail = False
    DWM10_Parms.minFreqStdToken = 5
    DWM10_Parms.minLenStdToken = 3
    DWM10_Parms.maxFreqErrToken = 3
    DWM10_Parms.beta = 2
    DWM10_Parms.blockByPairs = True
    DWM10_Parms.minBlkTokenLen = 4
    DWM10_Parms.excludeNumericBlocks = True
    DWM10_Parms.useVectorBlocking = False
    DWM10_Parms.blockCorrection = False
    DWM10_Parms.blockCorrectionDetail = False
    DWM10_Parms.epsilon = 0.50
    DWM10_Parms.epsilonIterate = 0.00
    DWM10_Parms.mu = 0.50
    DWM10_Parms.muIterate = 0.10
    DWM10_Parms.comparator = "ScoringMatrixStd"
    DWM10_Parms.matrixNumTokenRule = False
    DWM10_Parms.matrixInitialRule = False
    DWM10_Parms.sigma = 12
    DWM10_Parms.removeDuplicateTokens = False
    DWM10_Parms.removeExcludedBlkTokens = True
    DWM10_Parms.fatalError = False
    DWM10_Parms.precision = 0.0
    DWM10_Parms.recall = 0.0
    DWM10_Parms.fMeasure = 0.0
    DWM10_Parms.truePairs = 0
    DWM10_Parms.linkedPairs = 0
    DWM10_Parms.expectedPairs = 0
    DWM10_Parms.dataList = []
    DWM10_Parms.workbook = None
    DWM10_Parms.worksheet = None
    DWM10_Parms.startRow = 0


def load_parms_from_file(parm_file):
    """Load a base parameter file to pick up dataset-specific settings."""
    devnull = open(os.devnull, "w")
    DWM10_Parms.getParms(parm_file, devnull)
    devnull.close()


def run_single_dwm(data_file, truth_file, beta, sigma, mu, mu_iterate,
                    epsilon, epsilon_iterate, base_parm_file=None, quiet=True):
    """
    Run one DWM configuration and return a results dict.
    Uses base_parm_file for dataset-specific settings (delimiter, header, etc.)
    then overrides the sweep parameters.
    """
    reset_parms()

    # Load base settings from a parameter file if provided
    if base_parm_file and os.path.exists(base_parm_file):
        devnull = open(os.devnull, "w")
        DWM10_Parms.logFile = devnull
        DWM10_Parms.getParms(base_parm_file, devnull)

    # Override with sweep values
    DWM10_Parms.inputFileName = data_file
    DWM10_Parms.inputPrefix = data_file[: data_file.rfind(".")]
    DWM10_Parms.truthFileName = truth_file
    DWM10_Parms.beta = int(beta)
    DWM10_Parms.sigma = int(sigma)
    DWM10_Parms.mu = float(mu)
    DWM10_Parms.muIterate = float(mu_iterate)
    DWM10_Parms.epsilon = float(epsilon)
    DWM10_Parms.epsilonIterate = float(epsilon_iterate)
    DWM10_Parms.muStart = float(mu)
    DWM10_Parms.epsilonStart = float(epsilon)
    DWM10_Parms.runIterationProfile = False

    # Validate constraints
    if beta < 2:
        return None
    if sigma <= beta:
        return None
    if mu <= 0.0 or mu > 1.0:
        return None
    if epsilon <= 0.0 or epsilon > 1.0:
        return None

    # Redirect output to suppress DWM's print statements
    log_buf = io.StringIO()
    DWM10_Parms.logFile = log_buf

    t0 = time.time()

    if quiet:
        out_capture = io.StringIO()
    else:
        out_capture = sys.stdout

    try:
        with contextlib.redirect_stdout(out_capture):
            refDict = DWM14_BuildRefDict.tokenizeInput()
            linkIndex = DWM15_BuildLinkIndex.buildLinkIndex(refDict)
            tokenFreqDict = DWM16_BuildTokenFreqDict.buildTokenFreqDict(refDict)

            current_mu = DWM10_Parms.mu
            more = True
            while more:
                blockPairList = DWM42_BuildBlockPairs.buildBlockPairs(
                    refDict, linkIndex, tokenFreqDict
                )
                if len(blockPairList) == 0:
                    break

                linkedPairList = DWM55_LinkBlockPairs.linkBlockPairs(
                    blockPairList, refDict, tokenFreqDict
                )
                if len(linkedPairList) == 0:
                    break

                clusterList = DWM80_TransitiveClosure.transitiveClosure(linkedPairList)
                if len(clusterList) == 0:
                    break

                DWM90_IterateClusters.iterateClusters(
                    clusterList, refDict, linkIndex
                )

                current_mu += DWM10_Parms.muIterate
                current_mu = round(current_mu, 2)
                DWM10_Parms.mu = current_mu

                current_eps = DWM10_Parms.epsilon + DWM10_Parms.epsilonIterate
                current_eps = round(current_eps, 2)
                DWM10_Parms.epsilon = current_eps

                if current_mu > 1.0:
                    more = False

            # Compute metrics
            if truth_file:
                DWM99_ERmetrics.generateMetrics(linkIndex)

    except Exception as e:
        sys.stderr.write(f"Error running config beta={beta} sigma={sigma} "
                         f"mu={mu} eps={epsilon}: {e}\n")
        return None

    elapsed = round(time.time() - t0, 2)

    precision = DWM10_Parms.precision
    recall = DWM10_Parms.recall
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    return {
        "dataset": os.path.basename(data_file),
        "beta": int(beta),
        "sigma": int(sigma),
        "mu": round(float(DWM10_Parms.muStart), 2),
        "mu_iterate": round(float(mu_iterate), 2),
        "epsilon": round(float(DWM10_Parms.epsilonStart), 2),
        "epsilon_iterate": round(float(epsilon_iterate), 2),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_pairs": DWM10_Parms.truePairs,
        "linked_pairs": DWM10_Parms.linkedPairs,
        "expected_pairs": DWM10_Parms.expectedPairs,
        "num_records": DWM10_Parms.refCnt,
        "total_tokens": DWM10_Parms.tokenCnt,
        "unique_tokens": getattr(DWM10_Parms, "uniqueTokenCnt", 0),
        "token_freq_min": DWM10_Parms.minFreq,
        "token_freq_max": DWM10_Parms.maxFreq,
        "token_freq_mean": DWM10_Parms.avgFreq,
        "token_freq_std": DWM10_Parms.stdFreq,
        "token_len_min": DWM10_Parms.minLen,
        "token_len_max": DWM10_Parms.maxLen,
        "token_len_avg": DWM10_Parms.avgLen,
        "token_len_std": DWM10_Parms.stdDevLen,
        "runtime_seconds": elapsed,
    }


def generate_grid(beta_vals, sigma_vals, mu_vals, epsilon_vals):
    """Generate valid (beta, sigma, mu, epsilon) combinations."""
    combos = []
    for b, s, m, e in itertools.product(beta_vals, sigma_vals, mu_vals, epsilon_vals):
        if s <= b:
            continue
        if m <= 0.0 or m > 1.0:
            continue
        if e <= 0.0 or e > 1.0:
            continue
        combos.append((int(b), int(s), round(m, 4), round(e, 4)))
    return combos


def find_base_parm_file(data_file):
    """Try to find an existing parameter file for this dataset."""
    base = os.path.basename(data_file)
    parm_dir = os.path.dirname(data_file) or "."
    for pf in os.listdir(parm_dir):
        if not pf.endswith("-parms.txt"):
            continue
        try:
            with open(os.path.join(parm_dir, pf), "r") as fh:
                content = fh.read()
                if f"inputFileName={base}" in content:
                    return os.path.join(parm_dir, pf)
        except:
            pass
    return None


def run_sweep(data_file, truth_file, grid, mu_iterate=0.05, epsilon_iterate=0.0,
              output_file=None, resume=False):
    """Run the full parameter sweep for one dataset."""
    base_parm = find_base_parm_file(data_file)
    if base_parm:
        print(f"  Using base parameter file: {base_parm}")

    existing_keys = set()
    if resume and output_file and os.path.exists(output_file):
        with open(output_file, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = (row["dataset"], row["beta"], row["sigma"], row["mu"], row["epsilon"])
                existing_keys.add(key)
        print(f"  Resuming: {len(existing_keys)} configurations already completed")

    total = len(grid)
    results = []
    skipped = 0

    file_mode = "a" if resume and existing_keys else "w"
    csv_fh = None
    csv_writer = None
    if output_file:
        csv_fh = open(output_file, file_mode, newline="")
        csv_writer = csv.DictWriter(csv_fh, fieldnames=RESULT_COLUMNS)
        if file_mode == "w":
            csv_writer.writeheader()

    for idx, (beta, sigma, mu, epsilon) in enumerate(grid):
        key = (os.path.basename(data_file), str(beta), str(sigma), str(mu), str(epsilon))
        if key in existing_keys:
            skipped += 1
            continue

        progress = f"[{idx + 1}/{total}]"
        print(f"  {progress} beta={beta} sigma={sigma} mu={mu} eps={epsilon} ...", end="", flush=True)

        result = run_single_dwm(
            data_file, truth_file, beta, sigma, mu, mu_iterate,
            epsilon, epsilon_iterate, base_parm, quiet=True,
        )

        if result:
            results.append(result)
            print(f" P={result['precision']:.4f} R={result['recall']:.4f} F1={result['f1']:.4f} ({result['runtime_seconds']}s)")
            if csv_writer:
                csv_writer.writerow(result)
                csv_fh.flush()
        else:
            print(" SKIPPED (invalid or error)")

    if csv_fh:
        csv_fh.close()

    return results


def find_best(results):
    """Find the configuration with the highest F1."""
    if not results:
        return None
    return max(results, key=lambda r: r["f1"])


def print_summary(results, dataset_name):
    """Print a summary of the sweep results."""
    if not results:
        print(f"\n  No valid results for {dataset_name}")
        return

    f1s = [r["f1"] for r in results]
    best = find_best(results)

    print(f"\n{'='*70}")
    print(f"  SWEEP SUMMARY: {dataset_name}")
    print(f"{'='*70}")
    print(f"  Total configurations tested: {len(results)}")
    print(f"  F1 range: [{min(f1s):.4f}, {max(f1s):.4f}]")
    print(f"  F1 mean:  {sum(f1s)/len(f1s):.4f}")
    print(f"\n  BEST CONFIGURATION:")
    print(f"    beta*    = {best['beta']}")
    print(f"    sigma*   = {best['sigma']}")
    print(f"    mu*      = {best['mu']}")
    print(f"    epsilon* = {best['epsilon']}")
    print(f"    Precision = {best['precision']:.4f}")
    print(f"    Recall    = {best['recall']:.4f}")
    print(f"    F1        = {best['f1']:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Parameter Sweep for DWM Empirical Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Range syntax for parameters:
  --mu 0.50:0.95:0.05     start:stop:step (inclusive of stop)
  --beta 2,5,10,15        explicit comma-separated list
        """,
    )
    parser.add_argument("--data", help="Input dataset CSV file")
    parser.add_argument("--truth", help="Truth file")
    parser.add_argument(
        "--batch", help="Batch file: each line is datafile,truthfile"
    )
    parser.add_argument(
        "--beta", default="2,5,10,15",
        help="Beta values (default: 2,5,10,15)",
    )
    parser.add_argument(
        "--sigma", default="6,12,25,50",
        help="Sigma values (default: 6,12,25,50)",
    )
    parser.add_argument(
        "--mu", default="0.50:0.95:0.05",
        help="Mu values (default: 0.50:0.95:0.05)",
    )
    parser.add_argument(
        "--epsilon", default="0.05:0.50:0.05",
        help="Epsilon values (default: 0.05:0.50:0.05)",
    )
    parser.add_argument(
        "--mu-iterate", type=float, default=0.05,
        help="Mu iteration step (default: 0.05)",
    )
    parser.add_argument(
        "--epsilon-iterate", type=float, default=0.0,
        help="Epsilon iteration step (default: 0.0)",
    )
    parser.add_argument("--output", help="Output CSV file for results")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted sweep (skip already-completed configs)",
    )
    args = parser.parse_args()

    beta_vals = [int(x) for x in parse_range(args.beta)]
    sigma_vals = [int(x) for x in parse_range(args.sigma)]
    mu_vals = parse_range(args.mu)
    epsilon_vals = parse_range(args.epsilon)

    grid = generate_grid(beta_vals, sigma_vals, mu_vals, epsilon_vals)
    print(f"Parameter grid: {len(beta_vals)} beta x {len(sigma_vals)} sigma "
          f"x {len(mu_vals)} mu x {len(epsilon_vals)} epsilon")
    print(f"Valid combinations (sigma > beta): {len(grid)}")

    datasets = []
    if args.batch:
        with open(args.batch, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",", 1)
                datasets.append((parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""))
    elif args.data:
        datasets.append((args.data, args.truth or ""))
    else:
        parser.print_help()
        sys.exit(1)

    all_results = []
    for data_file, truth_file in datasets:
        print(f"\n{'='*70}")
        print(f"  Sweeping: {data_file}")
        print(f"{'='*70}")

        out_file = args.output
        if args.batch and args.output:
            base = os.path.splitext(os.path.basename(data_file))[0]
            out_file = f"{os.path.splitext(args.output)[0]}_{base}.csv"

        results = run_sweep(
            data_file, truth_file, grid,
            mu_iterate=args.mu_iterate,
            epsilon_iterate=args.epsilon_iterate,
            output_file=out_file,
            resume=args.resume,
        )
        all_results.extend(results)
        print_summary(results, os.path.basename(data_file))

    if len(datasets) > 1:
        print(f"\n{'='*70}")
        print(f"  OVERALL: {len(all_results)} total configurations across {len(datasets)} datasets")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
