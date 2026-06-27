"""
DWM sweep execution engine.

Owns the single-run DWM invocation, the outer evaluation loop, CSV
result logging, and resume-from-checkpoint logic.  Receives parameter
configurations from an external generator — contains no parameter
generation logic itself.
"""

import contextlib
import csv
import io
import os
import sys
import time
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

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


def _reset_parms() -> None:
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


def run_single_dwm(
    data_file: str,
    truth_file: str,
    beta: int,
    sigma: int,
    mu: float,
    mu_iterate: float,
    epsilon: float,
    epsilon_iterate: float,
    base_parm_file: Optional[str] = None,
    quiet: bool = True,
) -> Optional[Dict[str, Any]]:
    """Run one DWM configuration and return a results dict (or None on failure)."""
    _reset_parms()

    if base_parm_file and os.path.exists(base_parm_file):
        devnull = open(os.devnull, "w")
        DWM10_Parms.logFile = devnull
        DWM10_Parms.getParms(base_parm_file, devnull)

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

    if beta < 2:
        return None
    if sigma <= beta:
        return None
    if mu <= 0.0 or mu > 1.0:
        return None
    if epsilon <= 0.0 or epsilon > 1.0:
        return None

    log_buf = io.StringIO()
    DWM10_Parms.logFile = log_buf

    t0 = time.time()
    out_capture = io.StringIO() if quiet else sys.stdout

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

                DWM90_IterateClusters.iterateClusters(clusterList, refDict, linkIndex)

                current_mu += DWM10_Parms.muIterate
                current_mu = round(current_mu, 2)
                DWM10_Parms.mu = current_mu

                current_eps = DWM10_Parms.epsilon + DWM10_Parms.epsilonIterate
                current_eps = round(current_eps, 2)
                DWM10_Parms.epsilon = current_eps

                if current_mu > 1.0:
                    more = False

            if truth_file:
                DWM99_ERmetrics.generateMetrics(linkIndex)

    except Exception as e:
        sys.stderr.write(
            f"Error running config beta={beta} sigma={sigma} "
            f"mu={mu} eps={epsilon}: {e}\n"
        )
        return None

    elapsed = round(time.time() - t0, 2)

    precision = DWM10_Parms.precision
    recall = DWM10_Parms.recall
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if (precision + recall) > 0
        else 0.0
    )

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


def _load_existing_keys(output_file: str) -> Set[Tuple[str, str, str, str, str]]:
    """Read an existing results CSV and return the set of completed config keys."""
    keys: Set[Tuple[str, str, str, str, str]] = set()
    if not os.path.exists(output_file):
        return keys
    with open(output_file, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            keys.add(
                (row["dataset"], row["beta"], row["sigma"], row["mu"], row["epsilon"])
            )
    return keys


def run_sweep(
    data_file: str,
    truth_file: str,
    configs: Iterable[Tuple[int, int, float, float]],
    n_total: int,
    mu_iterate: float = 0.05,
    epsilon_iterate: float = 0.0,
    output_file: Optional[str] = None,
    resume: bool = False,
    base_parm_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run the full parameter sweep for one dataset.

    Args:
        data_file:      Path to the input dataset CSV.
        truth_file:     Path to the truth file.
        configs:        Iterable of (beta, sigma, mu, epsilon) tuples.
        n_total:        Expected number of configurations (for progress display).
        mu_iterate:     Mu iteration step per DWM iteration.
        epsilon_iterate: Epsilon iteration step per DWM iteration.
        output_file:    CSV file to write results to.
        resume:         If True, skip configurations already in *output_file*.
        base_parm_file: Optional path to a base DWM parameter file.

    Returns:
        List of result dicts for all successful configurations.
    """
    if base_parm_file:
        print(f"  Using base parameter file: {base_parm_file}")

    existing_keys: Set[Tuple[str, str, str, str, str]] = set()
    if resume and output_file:
        existing_keys = _load_existing_keys(output_file)
        if existing_keys:
            print(f"  Resuming: {len(existing_keys)} configurations already completed")

    results: List[Dict[str, Any]] = []

    file_mode = "a" if resume and existing_keys else "w"
    csv_fh = None
    csv_writer = None
    if output_file:
        csv_fh = open(output_file, file_mode, newline="")
        csv_writer = csv.DictWriter(csv_fh, fieldnames=RESULT_COLUMNS)
        if file_mode == "w":
            csv_writer.writeheader()

    for idx, (beta, sigma, mu, epsilon) in enumerate(configs):
        key = (
            os.path.basename(data_file),
            str(beta),
            str(sigma),
            str(mu),
            str(epsilon),
        )
        if key in existing_keys:
            continue

        progress = f"[{idx + 1}/{n_total}]"
        print(
            f"  {progress} beta={beta} sigma={sigma} mu={mu} eps={epsilon} ...",
            end="",
            flush=True,
        )

        result = run_single_dwm(
            data_file,
            truth_file,
            beta,
            sigma,
            mu,
            mu_iterate,
            epsilon,
            epsilon_iterate,
            base_parm_file,
            quiet=True,
        )

        if result:
            results.append(result)
            print(
                f" P={result['precision']:.4f} R={result['recall']:.4f} "
                f"F1={result['f1']:.4f} ({result['runtime_seconds']}s)"
            )
            if csv_writer:
                csv_writer.writerow(result)
                csv_fh.flush()
        else:
            print(" SKIPPED (invalid or error)")

    if csv_fh:
        csv_fh.close()

    return results
