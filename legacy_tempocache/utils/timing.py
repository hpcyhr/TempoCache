"""tempocache/utils/timing.py – lightweight timing context manager."""

from __future__ import annotations

import time


class Timer:
    """Usage: with Timer() as t: ...; print(t.elapsed)"""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self._start