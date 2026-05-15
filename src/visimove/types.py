from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np


Frame = np.ndarray


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class ScreenPoint:
    x: int
    y: int


@dataclass(frozen=True)
class EyeRegion:
    left: tuple[int, int, int, int] | None = None
    right: tuple[int, int, int, int] | None = None
    landmarks: dict[str, Any] | None = None


@dataclass(frozen=True)
class DetectionResult:
    found: bool
    face_box: tuple[int, int, int, int] | None = None
    eyes: EyeRegion | None = None
    confidence: float = 0.0


@dataclass(frozen=True)
class GazeEstimate:
    point: Point
    confidence: float


@dataclass(frozen=True)
class BlinkState:
    is_blinking: bool
    confidence: float


@dataclass
class PipelineState:
    frame_index: int = 0
    started_at: float = 0.0

    @classmethod
    def started(cls) -> "PipelineState":
        return cls(started_at=perf_counter())

