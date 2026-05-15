from __future__ import annotations

import cv2

from visimove.camera.base import CameraConfig
from visimove.types import Frame


class OpenCVCamera:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        capture = cv2.VideoCapture(self.config.index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera index {self.config.index}")
        self._capture = capture

    def read(self) -> Frame | None:
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        return frame if ok else None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

