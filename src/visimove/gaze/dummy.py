from __future__ import annotations

from visimove.types import DetectionResult, Frame, GazeEstimate, Point


class DummyGazeModel:
    def estimate(self, frame: Frame, detection: DetectionResult) -> GazeEstimate:
        if not detection.found:
            return GazeEstimate(point=Point(0.5, 0.5), confidence=0.0)
        return GazeEstimate(point=Point(0.5, 0.5), confidence=detection.confidence)

