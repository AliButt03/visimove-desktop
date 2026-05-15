from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from visimove.types import Point


@dataclass(frozen=True)
class GazeResult:
    raw_x: float
    raw_y: float
    screen_x: int | None = None
    screen_y: int | None = None
    gaze_vector: tuple[float, float, float] | None = None
    confidence: float = 0.0
    inference_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def point(self) -> Point:
        return Point(self.raw_x, self.raw_y)

