"""
Experiment I/O: CLI argument parsing, batch file loading, and dataset handling.
"""

import argparse
import os
from typing import List, Optional, Tuple


def parse_range(spec: str) -> List[float]:
    """Parse '0.50:0.95:0.05' (start:stop:step) or '2,5,10,15' (explicit list)."""
    if ":" in spec:
        parts = spec.split(":")
        start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
        vals: List[float] = []
        v = start
        while v <= stop + 1e-9:
            vals.append(round(v, 4))
            v += step
        return vals
    return [float(x) for x in spec.split(",")]


def find_base_parm_file(data_file: str) -> Optional[str]:
    """Locate an existing *-parms.txt whose inputFileName matches *data_file*."""
    base = os.path.basename(data_file)
    parm_dir = os.path.dirname(data_file) or "."
    try:
        entries = os.listdir(parm_dir)
    except OSError:
        return None
    for pf in entries:
        if not pf.endswith("-parms.txt"):
            continue
        try:
            with open(os.path.join(parm_dir, pf), "r") as fh:
                if f"inputFileName={base}" in fh.read():
                    return os.path.join(parm_dir, pf)
        except OSError:
            pass
    return None


def load_batch_file(path: str) -> List[Tuple[str, str]]:
    """Read a batch file and return a list of (data_file, truth_file) pairs."""
    datasets: List[Tuple[str, str]] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 1)
            data = parts[0].strip()
            truth = parts[1].strip() if len(parts) > 1 else ""
            datasets.append((data, truth))
    return datasets


def build_cli_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the sweep CLI."""
    parser = argparse.ArgumentParser(
        description="Parameter Sweep for DWM Empirical Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Range syntax for parameters:
  --mu 0.50:0.95:0.05     start:stop:step (inclusive of stop)
  --beta 2,5,10,15        explicit comma-separated list

Sampling mode (default):
  --n-samples 1000        number of random configurations to evaluate
  --seed 42               RNG seed for reproducibility
        """,
    )
    parser.add_argument("--data", help="Input dataset CSV file")
    parser.add_argument("--truth", help="Truth file")
    parser.add_argument(
        "--batch", help="Batch file: each line is datafile,truthfile"
    )
    parser.add_argument(
        "--beta",
        default="2,150",
        help="Beta range 'low,high' for random sampling (default: 2,150)",
    )
    parser.add_argument(
        "--sigma",
        default="6,150",
        help="Sigma range 'low,high' for random sampling (default: 6,150)",
    )
    parser.add_argument(
        "--mu",
        default=None,
        help="Mu values as explicit list or range (default: uniform in (0,1])",
    )
    parser.add_argument(
        "--epsilon",
        default=None,
        help="Epsilon values as explicit list or range (default: uniform in (0,1])",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of random configurations to sample (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducibility",
    )
    parser.add_argument(
        "--mu-iterate",
        type=float,
        default=0.05,
        help="Mu iteration step (default: 0.05)",
    )
    parser.add_argument(
        "--epsilon-iterate",
        type=float,
        default=0.0,
        help="Epsilon iteration step (default: 0.0)",
    )
    parser.add_argument("--output", help="Output CSV file for results")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted sweep (skip already-completed configs)",
    )
    return parser
