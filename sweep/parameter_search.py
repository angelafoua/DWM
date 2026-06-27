"""
Stochastic parameter search space generator for DWM empirical studies.

Replaces exhaustive grid search with streaming random sampling.
Generates (beta, sigma, mu, epsilon) tuples one at a time via a Python
generator, keeping memory constant regardless of sample count.
"""

import random
from typing import Generator, List, Optional, Tuple, Union


def generate_search_space(
    beta_range: Tuple[int, int] = (2, 150),
    sigma_range: Tuple[int, int] = (6, 150),
    mu_vals: Optional[List[float]] = None,
    epsilon_vals: Optional[List[float]] = None,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> Generator[Tuple[int, int, float, float], None, None]:
    """Yield random (beta, sigma, mu, epsilon) configurations one at a time.

    Each call draws beta and sigma uniformly from their integer ranges,
    and mu/epsilon either from provided discrete lists or uniformly from
    (0, 1].  Only valid configurations (sigma > beta) are emitted; invalid
    draws are silently re-sampled so exactly *n_samples* tuples are yielded.

    Args:
        beta_range:   Inclusive (low, high) for uniform integer sampling.
        sigma_range:  Inclusive (low, high) for uniform integer sampling.
        mu_vals:      Explicit list to sample from.  If None, drawn
                      uniformly from (0.01, 1.0].
        epsilon_vals: Explicit list to sample from.  If None, drawn
                      uniformly from (0.01, 1.0].
        n_samples:    Total configurations to yield.
        seed:         RNG seed for reproducibility.  None = non-deterministic.

    Yields:
        (beta, sigma, mu, epsilon) tuples compatible with the downstream
        ``for idx, (beta, sigma, mu, epsilon) in enumerate(...)`` loop.
    """
    rng = random.Random(seed)
    beta_lo, beta_hi = beta_range
    sigma_lo, sigma_hi = sigma_range

    emitted = 0
    while emitted < n_samples:
        beta = rng.randint(beta_lo, beta_hi)
        sigma = rng.randint(sigma_lo, sigma_hi)

        if sigma <= beta:
            continue

        if mu_vals is not None:
            mu = rng.choice(mu_vals)
        else:
            mu = round(rng.uniform(0.01, 1.0), 4)

        if epsilon_vals is not None:
            epsilon = rng.choice(epsilon_vals)
        else:
            epsilon = round(rng.uniform(0.01, 1.0), 4)

        if mu <= 0.0 or mu > 1.0:
            continue
        if epsilon <= 0.0 or epsilon > 1.0:
            continue

        emitted += 1
        yield (int(beta), int(sigma), round(mu, 4), round(epsilon, 4))
