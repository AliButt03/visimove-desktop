from __future__ import annotations

from dataclasses import dataclass

import cv2

from visimove.camera.base import CameraConfig
from visimove.types import Frame


@dataclass(frozen=True)
class WebcamFrame:
    frame: Frame
    original_width: int
    original_height: int


class WebcamCamera:
    """OpenCV webcam capture with clear errors and one optional resize per frame."""

    def __init__(self, config: CameraConfig, resize_width: int | None = None) -> None:
        self.config = config
        self.resize_width = resize_width
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        capture = cv2.VideoCapture(self.config.index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(self.config.index)

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)

        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open webcam index {self.config.index}. "
                "Check camera permissions, index, and whether another app is using it."
            )

        self._capture = capture

    def read(self) -> WebcamFrame | None:
        if self._capture is None:
            raise RuntimeError("WebcamCamera.read() called before open().")

        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None

        original_height, original_width = frame.shape[:2]
        if self.resize_width and 0 < self.resize_width < original_width:
            scale = self.resize_width / original_width
            resized_height = max(1, round(original_height * scale))
            frame = cv2.resize(frame, (self.resize_width, resized_height), interpolation=cv2.INTER_AREA)

        return WebcamFrame(
            frame=frame,
            original_width=original_width,
            original_height=original_height,
        )

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

