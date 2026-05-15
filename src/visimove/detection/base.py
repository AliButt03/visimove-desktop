from __future__ import annotations

from typing import Protocol

from visimove.types import DetectionResult, Frame


class FaceEyeDetector(Protocol):
    def detect(self, frame: Frame) -> DetectionResult: ...

