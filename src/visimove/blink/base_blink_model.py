from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from visimove.types import Frame


@dataclass(frozen=True)
class BlinkResult:
    left_closed_probability: float
    right_closed_probability: float
    combined_closed_probability: float
    confidence: float
    inference_time_ms: float = 0.0

    @property
    def is_blinking(self) -> bool:
        return self.combined_closed_probability >= 0.5


class BaseBlinkModel(Protocol):
    def infer(self, frame: Frame, detection: Any) -> BlinkResult: ...


def empty_blink_result() -> BlinkResult:
    return BlinkResult(
        left_closed_probability=0.0,
        right_closed_probability=0.0,
        combined_closed_probability=0.0,
        confidence=0.0,
        inference_time_ms=0.0,
    )

