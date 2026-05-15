from __future__ import annotations

import math
from typing import Any

from visimove.gaze.gaze_output import GazeResult
from visimove.types import DetectionResult, Frame


class MovingDummyGazeModel:
    """Produces a smooth fake normalized gaze point for end-to-end testing."""

    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        self._frame_index = 0
        self._metadata = metadata or {}

    def estimate(self, frame: Frame, detection: DetectionResult) -> GazeResult:
        metadata = {"backend": "dummy", **self._metadata}
        if not detection.found:
            return GazeResult(raw_x=0.5, raw_y=0.5, confidence=0.0, metadata=metadata)

        self._frame_index += 1
        t = self._frame_index / 30.0
        x = 0.5 + 0.28 * math.sin(t)
        y = 0.5 + 0.18 * math.sin(t * 0.7)
        return GazeResult(raw_x=x, raw_y=y, confidence=0.85, metadata=metadata)
