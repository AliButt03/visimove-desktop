from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeadzoneFilter:
    radius: float = 8.0
    max_jump_pixels: float | None = None
    _last: tuple[float, float] | None = None

    def update(
        self,
        x: float | None,
        y: float | None,
        timestamp: float,
    ) -> tuple[float, float] | None:
        if x is None or y is None:
            return self._last

        target = (float(x), float(y))
        if self._last is None:
            self._last = target
            return self._last

        distance = self._distance(self._last, target)
        if distance <= self.radius:
            return self._last

        self._last = self._clamp_jump(self._last, target)
        return self._last

    def reset(self) -> None:
        self._last = None

    def _clamp_jump(
        self,
        origin: tuple[float, float],
        target: tuple[float, float],
    ) -> tuple[float, float]:
        if self.max_jump_pixels is None or self.max_jump_pixels <= 0:
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

