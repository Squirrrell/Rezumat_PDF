"""Pad reported runtimes to target intervals (sleep when work finishes early)."""

from __future__ import annotations

import random
import time


def pad_runtime_to_interval(
    started_at: float,
    min_seconds: float,
    max_seconds: float,
) -> float:
    """
    Return wall-clock elapsed since started_at.

    If elapsed is below min_seconds, sleep until a random target in
    [min_seconds, max_seconds] is reached. If already >= min_seconds, no sleep.
    """
    elapsed = time.perf_counter() - started_at
    if elapsed >= min_seconds:
        return elapsed

    target = random.uniform(min_seconds, max_seconds)
    remaining = target - elapsed
    if remaining > 0:
        time.sleep(remaining)
    return time.perf_counter() - started_at
