from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from visimove.blink import BlinkModel, DummyBlinkModel
from visimove.calibration import CalibrationMapper
from visimove.camera import CameraConfig, OpenCVCamera
from visimove.camera.base import Camera
from visimove.cursor import CursorController, DryRunCursorController, PyAutoGuiCursorController
from visimove.detection import DummyFaceEyeDetector, FaceEyeDetector
from visimove.gaze import DummyGazeModel, GazeModel
from visimove.smoothing import ExponentialSmoothingFilter, SmoothingFilter


@dataclass(frozen=True)
class PipelineRuntime:
    camera: Camera
    detector: FaceEyeDetector
    gaze_model: GazeModel
    blink_model: BlinkModel
    mapper: CalibrationMapper
    smoother: SmoothingFilter
    cursor: CursorController


def build_runtime(config: dict[str, Any]) -> PipelineRuntime:
    camera_cfg = CameraConfig(**config.get("camera", {}))
    dry_run = bool(config.get("pipeline", {}).get("dry_run", True))
    cursor_enabled = bool(config.get("cursor", {}).get("enabled", False))
    alpha = float(config.get("smoothing", {}).get("alpha", 0.35))

    cursor: CursorController
    if dry_run or not cursor_enabled:
        cursor = DryRunCursorController()
    else:
        cursor = PyAutoGuiCursorController()

    return PipelineRuntime(
        camera=OpenCVCamera(camera_cfg),
        detector=DummyFaceEyeDetector(),
        gaze_model=DummyGazeModel(),
        blink_model=DummyBlinkModel(),
        mapper=CalibrationMapper(),
        smoother=ExponentialSmoothingFilter(alpha=alpha),
        cursor=cursor,
    )
