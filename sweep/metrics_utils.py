"""
Post-sweep summary statistics and best-configuration selection.
"""

import os
from typing import Any, Dict, List, Optional


def find_best(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the configuration with the highest F1 score, or None."""
    if not results:
        return None
    return max(results, key=lambda r: r["f1"])


def print_summary(results: List[Dict[str, Any]], dataset_name: str) -> None:
    """Print a human-readable summary of sweep results for one dataset."""
    if not results:
        print(f"\n  No valid results for {dataset_name}")
        return

    f1s = [r["f1"] for r in results]
    best = find_best(results)

    print(f"\n{'=' * 70}")
    print(f"  SWEEP SUMMARY: {dataset_name}")
    print(f"{'=' * 70}")
    print(f"  Total configurations tested: {len(results)}")
    print(f"  F1 range: [{min(f1s):.4f}, {max(f1s):.4f}]")
    print(f"  F1 mean:  {sum(f1s) / len(f1s):.4f}")
    print(f"\n  BEST CONFIGURATION:")
    print(f"    beta*    = {best['beta']}")
    print(f"    sigma*   = {best['sigma']}")
    print(f"    mu*      = {best['mu']}")
    print(f"    epsilon* = {best['epsilon']}")
    print(f"    Precision = {best['precision']:.4f}")
    print(f"    Recall    = {best['recall']:.4f}")
    print(f"    F1        = {best['f1']:.4f}")
    print()
