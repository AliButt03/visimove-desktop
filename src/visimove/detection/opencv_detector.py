from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from visimove.detection.base_detector import BaseDetector, Box, DetectorResult, clamp_box, expanded_box, safe_crop
from visimove.types import Frame


@dataclass
class OpenCvHaarDetector:
    min_face_size: tuple[int, int] = (80, 80)
    reuse_frames: int = 2
    eye_padding_scale: float = 1.6
    _face_cascade: cv2.CascadeClassifier | None = None
    _eye_cascade: cv2.CascadeClassifier | None = None
    _last_result: DetectorResult | None = None
    _reuse_count: int = 0

    def __post_init__(self) -> None:
        haar_dir = Path(cv2.data.haarcascades)
        self._face_cascade = cv2.CascadeClassifier(str(haar_dir / "haarcascade_frontalface_default.xml"))
        self._eye_cascade = cv2.CascadeClassifier(str(haar_dir / "haarcascade_eye.xml"))
        if self._face_cascade.empty():
            raise RuntimeError("OpenCV face Haar cascade could not be loaded.")
        if self._eye_cascade.empty():
            raise RuntimeError("OpenCV eye Haar cascade could not be loaded.")

    def detect(self, frame: Frame, timestamp: float | None = None) -> DetectorResult:
        if self._last_result is not None and self._reuse_count < self.reuse_frames:
            self._reuse_count += 1
            return self._with_reused_crops(frame, self._last_result)

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(  # type: ignore[union-attr]
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_face_size,
        )
        if len(faces) == 0:
            self._last_result = None
            self._reuse_count = 0
            return DetectorResult(found=False)

        face_box = self._largest_box([tuple(map(int, face)) for face in faces])
        face_box = clamp_box(face_box, width, height)
        if face_box is None:
            return DetectorResult(found=False)

        left_eye_box, right_eye_box = self._detect_eyes(gray, face_box, width, height)
        detection = DetectorResult(
            found=True,
            face_box=face_box,
            left_eye_box=left_eye_box,
            right_eye_box=right_eye_box,
            left_eye_crop=safe_crop(frame, left_eye_box),
            right_eye_crop=safe_crop(frame, right_eye_box),
            landmarks=None,
            confidence=0.75,
        )
        self._last_result = detection
        self._reuse_count = 0
        return detection

    def _detect_eyes(
        self,
        gray: Frame,
        face_box: Box,
        frame_width: int,
        frame_height: int,
    ) -> tuple[Box | None, Box | None]:
        face_x, face_y, face_w, face_h = face_box
        upper_face_h = max(1, face_h // 2)
        roi = gray[face_y:face_y + upper_face_h, face_x:face_x + face_w]
        eyes = self._eye_cascade.detectMultiScale(roi, scaleFactor=1.08, minNeighbors=4)  # type: ignore[union-attr]
        eye_boxes = [
            expanded_box(
                (face_x + int(x), face_y + int(y), int(w), int(h)),
                self.eye_padding_scale,
                frame_width,
                frame_height,
            )
            for x, y, w, h in eyes
        ]
        eye_boxes = [box for box in eye_boxes if box is not None]
        if len(eye_boxes) >= 2:
            eye_boxes = sorted(eye_boxes, key=lambda box: box[0])[:2]
            return eye_boxes[0], eye_boxes[1]

        fallback_y = face_y + face_h // 3
        eye_w = max(12, face_w // 5)
        eye_h = max(8, face_h // 8)
        left = clamp_box((face_x + face_w // 4 - eye_w // 2, fallback_y, eye_w, eye_h), frame_width, frame_height)
        right = clamp_box((face_x + (face_w * 3) // 4 - eye_w // 2, fallback_y, eye_w, eye_h), frame_width, frame_height)
        return left, right

    @staticmethod
    def _largest_box(boxes: list[Box]) -> Box:
        return max(boxes, key=lambda box: box[2] * box[3])

    @staticmethod
    def _with_reused_crops(frame: Frame, result: DetectorResult) -> DetectorResult:
        return DetectorResult(
            found=result.found,
            face_box=result.face_box,
            left_eye_box=result.left_eye_box,
            right_eye_box=result.right_eye_box,
            left_eye_crop=safe_crop(frame, result.left_eye_box),
            right_eye_crop=safe_crop(frame, result.right_eye_box),
            landmarks=result.landmarks,
            confidence=result.confidence,
            reused_previous=True,
        )

