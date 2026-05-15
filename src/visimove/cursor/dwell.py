from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from visimove.types import ScreenPoint


@dataclass
class DwellSelector:
    dwell_time_ms: int = 900
    radius_px: int = 35
    _anchor: ScreenPoint | None = None
    _started_at: float | None = None

    def update(self, point: ScreenPoint) -> bool:
        now = perf_counter()
        if self._anchor is None or self._distance(point, self._anchor) > self.radius_px:
            self._anchor = point
            self._started_at = now
            return False
        if self._started_at is None:
            self._started_at = now
            return False
        if (now - self._started_at) * 1000 >= self.dwell_time_ms:
            self._started_at = now
            return True
        return False

    @staticmethod
    def _distance(a: ScreenPoint, b: ScreenPoint) -> float:
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

