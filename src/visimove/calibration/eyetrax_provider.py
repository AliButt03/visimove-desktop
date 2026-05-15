from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import cv2

from visimove.gaze.eyetrax_adapter import EyeTraxAdapter
from visimove.gaze.gaze_output import GazeResult


class EyeTraxCalibrationGazeProvider:
    """Reads webcam frames and returns real EyeTrax gaze estimates for calibration."""

    def __init__(self, adapter: EyeTraxAdapter, camera_index: int) -> None:
        self.adapter = adapter
        self.camera_index = camera_index
        self._capture: Any = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Could not open camera index {camera_index} for EyeTrax calibration.")

    def read_gaze(self) -> GazeResult | None:
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        detection_hint = SimpleNamespace(found=True, eyes=True)
        gaze = self.adapter.estimate(frame, detection_hint)
        if gaze.confidence <= 0.0:
            return None
        return gaze

    def close(self) -> None:
        self._capture.release()
