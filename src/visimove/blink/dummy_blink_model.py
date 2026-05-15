from __future__ import annotations

from dataclasses import dataclass

from visimove.blink.base_blink_model import BlinkResult
from visimove.types import DetectionResult, Frame


@dataclass
class KeyboardDummyBlinkModel:
    """Dummy blink model that can be triggered by keyboard input from the preview loop."""

    fake_probability: float = 0.05
    trigger_probability: float = 0.95
    _triggered: bool = False

    def trigger(self) -> None:
        self._triggered = True

    def infer(self, frame: Frame, detection: DetectionResult) -> BlinkResult:
        if not detection.found:
            return BlinkResult(0.0, 0.0, 0.0, 0.0, 0.0)
        if self._triggered:
            self._triggered = False
            return BlinkResult(
                left_closed_probability=self.trigger_probability,
                right_closed_probability=self.trigger_probability,
                combined_closed_probability=self.trigger_probability,
                confidence=1.0,
                inference_time_ms=0.0,
            )
        return BlinkResult(
            left_closed_probability=self.fake_probability,
            right_closed_probability=self.fake_probability,
            combined_closed_probability=self.fake_probability,
            confidence=1.0,
            inference_time_ms=0.0,
        )
