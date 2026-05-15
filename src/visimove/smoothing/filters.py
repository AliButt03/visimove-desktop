from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from visimove.smoothing.ema_filter import EmaFilter
from visimove.types import ScreenPoint


class SmoothingFilter(Protocol):
    def update(self, point: ScreenPoint) -> ScreenPoint: ...

    def reset(self) -> None: ...


@dataclass
class ExponentialSmoothingFilter:
    alpha: float = 0.35
    max_jump_pixels: float | None = None
    _filter: EmaFilter | None = None

    def update(self, point: ScreenPoint) -> ScreenPoint:
        if self._filter is None:
            self._filter = EmaFilter(alpha=self.alpha, max_jump_pixels=self.max_jump_pixels)
        output = self._filter.update(point.x, point.y, monotonic())
        if output is None:
            return point
        return ScreenPoint(x=round(output[0]), y=round(output[1]))

    def reset(self) -> None:
        if self._filter is not None:
            self._filter.reset()
