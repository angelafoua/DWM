#!/usr/bin/env python
"""
CLI entry point for the DWM parameter sweep system.

Orchestrates the modular pipeline:
  experiment_io   -> CLI parsing, batch/dataset loading
  parameter_search -> streaming random-sample generator
  sweep_runner     -> DWM execution loop + CSV logging
  metrics_utils    -> summary reporting
"""

import os
import sys

from sweep.experiment_io import build_cli_parser, find_base_parm_file, load_batch_file, parse_range
from sweep.metrics_utils import print_summary
from sweep.parameter_search import generate_search_space
from sweep.sweep_runner import run_sweep
from typing import List, Optional, Tuple


def main() -> None:
    parser = build_cli_parser()
    args = parser.parse_args()

    # --- resolve datasets ---------------------------------------------------
    datasets: List[Tuple[str, str]] = []
    if args.batch:
        datasets = load_batch_file(args.batch)
    elif args.data:
        datasets.append((args.data, args.truth or ""))
    else:
        parser.print_help()
        sys.exit(1)

    # --- resolve mu / epsilon discrete lists (None → continuous uniform) -----
    mu_vals: Optional[List[float]] = None
    if args.mu is not None:
        mu_vals = parse_range(args.mu)

    epsilon_vals: Optional[List[float]] = None
    if args.epsilon is not None:
        epsilon_vals = parse_range(args.epsilon)

    # --- resolve beta / sigma ranges ----------------------------------------
    beta_parts = [int(x) for x in args.beta.split(",")]
    if len(beta_parts) == 2:
        beta_range = (beta_parts[0], beta_parts[1])
    else:
        beta_range = (min(beta_parts), max(beta_parts))

    sigma_parts = [int(x) for x in args.sigma.split(",")]
    if len(sigma_parts) == 2:
        sigma_range = (sigma_parts[0], sigma_parts[1])
    else:
        sigma_range = (min(sigma_parts), max(sigma_parts))

    n_samples = args.n_samples
    seed = args.seed

    print(
        f"Random sampling: {n_samples} configurations  "
        f"beta∈[{beta_range[0]},{beta_range[1]}]  "
        f"sigma∈[{sigma_range[0]},{sigma_range[1]}]  "
        f"seed={seed}"
    )
    if mu_vals:
        print(f"  mu drawn from {len(mu_vals)} discrete values")
    else:
        print("  mu drawn uniformly from (0, 1]")
    if epsilon_vals:
        print(f"  epsilon drawn from {len(epsilon_vals)} discrete values")
    else:
        print("  epsilon drawn uniformly from (0, 1]")

    # --- sweep each dataset -------------------------------------------------
    all_results = []
    for data_file, truth_file in datasets:
        print(f"\n{'=' * 70}")
        print(f"  Sweeping: {data_file}")
        print(f"{'=' * 70}")

        base_parm = find_base_parm_file(data_file)

        out_file = args.output
        if args.batch and args.output:
            base = os.path.splitext(os.path.basename(data_file))[0]
            out_file = f"{os.path.splitext(args.output)[0]}_{base}.csv"

        configs = generate_search_space(
            beta_range=beta_range,
            sigma_range=sigma_range,
            mu_vals=mu_vals,
            epsilon_vals=epsilon_vals,
            n_samples=n_samples,
            seed=seed,
        )

        results = run_sweep(
            data_file,
            truth_file,
            configs,
            n_total=n_samples,
            mu_iterate=args.mu_iterate,
            epsilon_iterate=args.epsilon_iterate,
            output_file=out_file,
            resume=args.resume,
            base_parm_file=base_parm,
        )
        all_results.extend(results)
        print_summary(results, os.path.basename(data_file))

    if len(datasets) > 1:
        print(f"\n{'=' * 70}")
        print(
            f"  OVERALL: {len(all_results)} total configurations "
            f"across {len(datasets)} datasets"
        )
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
