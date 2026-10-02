from __future__ import annotations

from typing import Any

import cv2

from visimove.detection.base_detector import BaseDetector
from visimove.gaze.gaze_output import GazeResult
from visimove.gaze.mobilegaze_adapter import MobileGazeAdapter


class MobileGazeCalibrationGazeProvider:
    """Reads webcam frames and returns real MobileGaze ONNX estimates for calibration."""

    def __init__(self, adapter: MobileGazeAdapter, detector: BaseDetector, camera_index: int) -> None:
        self.adapter = adapter
        self.detector = detector
        self.camera_index = camera_index
        self._capture: Any = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Could not open camera index {camera_index} for MobileGaze calibration.")

    def read_gaze(self) -> GazeResult | None:
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        detection = self.detector.detect(frame)
        if not detection.found:
            return None
        gaze = self.adapter.estimate(frame, detection)
        if gaze.confidence <= 0.0:
            return None
        return gaze

    def close(self) -> None:
        self._capture.release()
