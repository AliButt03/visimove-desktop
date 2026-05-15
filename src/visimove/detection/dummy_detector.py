from __future__ import annotations

from visimove.detection.base_detector import DetectorResult, safe_crop
from visimove.types import Frame


class DummyDetector:
    """Lightweight placeholder detector returning stable face and eye boxes."""

    def detect(self, frame: Frame, timestamp: float | None = None) -> DetectorResult:
        height, width = frame.shape[:2]
        face_width = width // 2
        face_height = height // 2
        face_x = (width - face_width) // 2
        face_y = height // 5

        eye_width = max(12, width // 10)
        eye_height = max(8, height // 16)
        eye_y = face_y + face_height // 3

        left_eye = (face_x + face_width // 4 - eye_width // 2, eye_y, eye_width, eye_height)
        right_eye = (face_x + (face_width * 3) // 4 - eye_width // 2, eye_y, eye_width, eye_height)
        return DetectorResult(
            found=True,
            face_box=(face_x, face_y, face_width, face_height),
            left_eye_box=left_eye,
            right_eye_box=right_eye,
            left_eye_crop=safe_crop(frame, left_eye),
            right_eye_crop=safe_crop(frame, right_eye),
            landmarks={"source": "dummy"},
            confidence=1.0,
        )
