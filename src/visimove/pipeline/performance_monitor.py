from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator

STAGE_ORDER = ("camera", "detector", "gaze", "blink", "calibration", "smoothing", "cursor")


@dataclass
class StageStats:
    calls: int = 0
    total_ms: float = 0.0
    last_ms: float = 0.0

    @property
    def average_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0


@dataclass
class PipelinePerformanceMonitor:
    stage_stats: dict[str, StageStats] = field(default_factory=dict)
    frame_count: int = 0
    _started_at: float = field(default_factory=perf_counter)
    _last_report_at: float = field(default_factory=perf_counter)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (perf_counter() - started) * 1000
            stats = self.stage_stats.setdefault(name, StageStats())
            stats.calls += 1
            stats.last_ms = elapsed_ms
            stats.total_ms += elapsed_ms

    def mark_frame(self) -> None:
        self.frame_count += 1

    @property
    def fps(self) -> float:
        elapsed = perf_counter() - self._started_at
        return self.frame_count / elapsed if elapsed > 0 else 0.0

    def should_report(self, interval_seconds: float) -> bool:
        now = perf_counter()
        if now - self._last_report_at >= interval_seconds:
            self._last_report_at = now
            return True
        return False

    def format_report(self) -> str:
        reported = set()
        ordered_parts: list[str] = []
        for name in STAGE_ORDER:
            stats = self.stage_stats.get(name)
            ordered_parts.append(f"{name}: {(stats.last_ms if stats else 0.0):.1f}ms")
            reported.add(name)
        extra_parts = [
            f"{name}: {stats.last_ms:.1f}ms"
            for name, stats in sorted(self.stage_stats.items())
            if name not in reported
        ]
        stages = " | ".join([*ordered_parts, *extra_parts])
        return f"FPS: {self.fps:.1f} | {stages}"

    def format_report_legacy(self) -> str:
        stages = " | ".join(
            f"{name}: {stats.last_ms:.1f}ms"
            for name, stats in sorted(self.stage_stats.items())
        )
        return f"FPS: {self.fps:.1f}" + (f" | {stages}" if stages else "")
