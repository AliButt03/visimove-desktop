from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator


@dataclass
class PerformanceMonitor:
    frame_count: int = 0
    total_frame_time_ms: float = 0.0
    last_frame_time_ms: float = 0.0
    _started_at: float = field(default_factory=perf_counter)

    @contextmanager
    def measure_frame(self) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (perf_counter() - started) * 1000
            self.frame_count += 1
            self.last_frame_time_ms = elapsed_ms
            self.total_frame_time_ms += elapsed_ms

    @property
    def average_frame_time_ms(self) -> float:
        if self.frame_count == 0:
            return 0.0
        return self.total_frame_time_ms / self.frame_count

    @property
    def fps(self) -> float:
        elapsed = perf_counter() - self._started_at
        if elapsed <= 0:
            return 0.0
        return self.frame_count / elapsed

