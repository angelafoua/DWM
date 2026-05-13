"""
Lightweight runtime and memory profiling utilities.

Phase 10 requires profiling at every pipeline stage without pulling in heavy
dependencies.  We rely only on the standard library (time, tracemalloc) and
expose three ergonomic interfaces:

  Timer           — context manager measuring wall-clock seconds.
  MemoryProfiler  — context manager measuring peak heap allocation via
                    tracemalloc (Python objects only; C extensions like NumPy
                    arrays may be under-counted).
  profile_block   — convenience context manager that combines both.
"""

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TimingResult:
    label: str
    elapsed_seconds: float

    def __str__(self) -> str:
        return f"[TIMING] {self.label}: {self.elapsed_seconds:.3f}s"


@dataclass
class MemoryResult:
    label: str
    peak_mib: float

    def __str__(self) -> str:
        return f"[MEMORY] {self.label}: peak {self.peak_mib:.2f} MiB"


@dataclass
class ProfilingResult:
    label: str
    elapsed_seconds: float = 0.0
    peak_mib: float = 0.0

    def __str__(self) -> str:
        return (
            f"[PROFILE] {self.label}: "
            f"{self.elapsed_seconds:.3f}s | peak {self.peak_mib:.2f} MiB"
        )


class Timer:
    """Context manager that measures wall-clock elapsed time."""

    def __init__(self, label: str = "block", log: bool = True) -> None:
        self.label = label
        self.log = log
        self.result: Optional[TimingResult] = None
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        elapsed = time.perf_counter() - self._start
        self.result = TimingResult(self.label, elapsed)
        if self.log:
            logger.info(str(self.result))


class MemoryProfiler:
    """Context manager that measures peak heap allocation via tracemalloc."""

    def __init__(self, label: str = "block", log: bool = True) -> None:
        self.label = label
        self.log = log
        self.result: Optional[MemoryResult] = None

    def __enter__(self) -> "MemoryProfiler":
        tracemalloc.start()
        return self

    def __exit__(self, *_) -> None:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mib = peak / (1024 ** 2)
        self.result = MemoryResult(self.label, peak_mib)
        if self.log:
            logger.info(str(self.result))


@contextmanager
def profile_block(label: str, log: bool = True):
    """Combined timing + memory profiling context manager.

    Usage::

        with profile_block("build index") as p:
            index.build(vectors)
        print(p.elapsed_seconds, p.peak_mib)
    """
    result = ProfilingResult(label=label)
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_seconds = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result.peak_mib = peak / (1024 ** 2)
        if log:
            logger.info(str(result))
