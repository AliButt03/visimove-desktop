from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from visimove.types import EyeRegion, Frame


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectorResult:
    found: bool
    face_box: Box | None = None
    left_eye_box: Box | None = None
    right_eye_box: Box | None = None
    left_eye_crop: Frame | None = None
    right_eye_crop: Frame | None = None
    landmarks: dict[str, Any] | None = None
    confidence: float = 0.0
    reused_previous: bool = False

    @property
    def eyes(self) -> EyeRegion | None:
        if self.left_eye_box is None and self.right_eye_box is None and self.landmarks is None:
            return None
        return EyeRegion(left=self.left_eye_box, right=self.right_eye_box, landmarks=self.landmarks)


class BaseDetector(Protocol):
    def detect(self, frame: Frame, timestamp: float | None = None) -> DetectorResult: ...


def safe_crop(frame: Frame, box: Box | None) -> Frame | None:
    if box is None:
        return None
    height, width = frame.shape[:2]
    x, y, w, h = box
    x1 = max(0, min(width, x))
    y1 = max(0, min(height, y))
    x2 = max(0, min(width, x + max(0, w)))
    y2 = max(0, min(height, y + max(0, h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def clamp_box(box: Box, frame_width: int, frame_height: int) -> Box | None:
    x, y, w, h = box
    x1 = max(0, min(frame_width - 1, x))
    y1 = max(0, min(frame_height - 1, y))
    x2 = max(0, min(frame_width, x + max(0, w)))
    y2 = max(0, min(frame_height, y + max(0, h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def expanded_box(box: Box, scale: float, frame_width: int, frame_height: int) -> Box | None:
    x, y, w, h = box
    center_x = x + w / 2
    center_y = y + h / 2
    new_w = w * scale
    new_h = h * scale
    return clamp_box(
        (
            round(center_x - new_w / 2),
            round(center_y - new_h / 2),
            round(new_w),
            round(new_h),
        ),
        frame_width,
        frame_height,
    )

