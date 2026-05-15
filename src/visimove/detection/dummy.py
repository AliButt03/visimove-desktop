from __future__ import annotations

from visimove.types import DetectionResult, EyeRegion, Frame


class DummyFaceEyeDetector:
    def detect(self, frame: Frame) -> DetectionResult:
        height, width = frame.shape[:2]
        face = (width // 4, height // 5, width // 2, height // 2)
        eye_width = width // 10
        eye_height = height // 14
        eyes = EyeRegion(
            left=(width // 3, height // 3, eye_width, eye_height),
            right=(width // 2, height // 3, eye_width, eye_height),
        )
        return DetectionResult(found=True, face_box=face, eyes=eyes, confidence=1.0)

