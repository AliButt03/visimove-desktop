from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FixationFilter:
    fixation_duration_ms: int = 180
    fixation_radius: float = 18.0
    max_jump_pixels: float | None = None
    _anchor: tuple[float, float] | None = None
    _anchor_started_at: float | None = None
    _last_output: tuple[float, float] | None = None

    def update(
        self,
        x: float | None,
        y: float | None,
        timestamp: float,
    ) -> tuple[float, float] | None:
        if x is None or y is None:
            return self._last_output

        target = (float(x), float(y))
        if self._anchor is None:
            self._anchor = target
            self._anchor_started_at = timestamp
            self._last_output = target
            return target

        if self._distance(self._anchor, target) > self.fixation_radius:
            self._anchor = target
            self._anchor_started_at = timestamp
            self._last_output = self._clamp_jump(self._last_output, target)
            return self._last_output

        if self._anchor_started_at is not None:
            elapsed_ms = (timestamp - self._anchor_started_at) * 1000
            if elapsed_ms >= self.fixation_duration_ms:
                self._last_output = self._clamp_jump(self._last_output, self._anchor)
        return self._last_output

    def reset(self) -> None:
        self._anchor = None
        self._anchor_started_at = None
        self._last_output = None

    def _clamp_jump(
        self,
        origin: tuple[float, float] | None,
        target: tuple[float, float],
    ) -> tuple[float, float]:
        if origin is None or self.max_jump_pixels is None or self.max_jump_pixels <= 0:
            return target
        distance = self._distance(origin, target)
        if distance <= self.max_jump_pixels:
            return target
        scale = self.max_jump_pixels / distance
        return origin[0] + (target[0] - origin[0]) * scale, origin[1] + (target[1] - origin[1]) * scale

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return (dx * dx + dy * dy) ** 0.5

