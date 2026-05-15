from __future__ import annotations

from visimove.types import BlinkState, DetectionResult, Frame


class DummyBlinkModel:
    def infer(self, frame: Frame, detection: DetectionResult) -> BlinkState:
        return BlinkState(is_blinking=False, confidence=1.0 if detection.found else 0.0)

