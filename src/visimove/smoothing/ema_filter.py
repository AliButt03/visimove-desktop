from __future__ import annotations

from dataclasses import dataclass


PointValue = tuple[float, float] | None


@dataclass
class EmaFilter:
    alpha: float = 0.35
    max_jump_pixels: float | None = None
    _last: tuple[float, float] | None = None

    def update(self, x: float | None, y: float | None, timestamp: float) -> PointValue:
        if x is None or y is None:
            return self._last

        target = (float(x), float(y))
        if self._last is None:
            self._last = target
            return self._last

        target = self._clamp_jump(self._last, target)
        smoothed = (
            self.alpha * target[0] + (1.0 - self.alpha) * self._last[0],
            self.alpha * target[1] + (1.0 - self.alpha) * self._last[1],
        )
        self._last = smoothed
        return smoothed

    def reset(self) -> None:
        self._last = None

    def _clamp_jump(
        self,
        origin: tuple[float, float],
        target: tuple[float, float],
    ) -> tuple[float, float]:
        if self.max_jump_pixels is None or self.max_jump_pixels <= 0:
            return target
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        distance = (dx * dx + dy * dy) ** 0.5
        if distance <= self.max_jump_pixels:
            return target
        scale = self.max_jump_pixels / distance
        return origin[0] + dx * scale, origin[1] + dy * scale

