from __future__ import annotations

from pathlib import Path
from typing import Any

from visimove.external_backends import require_external_repo
from visimove.gaze.base_gaze_model import GazeBackendUnavailable
from visimove.gaze.gaze_output import GazeResult
from visimove.types import Frame


class MobileGazeAdapter:
    """Adapter shell for `external/mobilegaze`.

    Expected setup:
    - repo: `external/mobilegaze/`
    - ONNX weights: `models/gaze/mobilegaze/*.onnx`
    - PyTorch weights for experiments only: `models/gaze/mobilegaze/*.pt`

    TODO:
    - Prefer ONNX Runtime for desktop inference.
    - Add preprocessing for face/eye/head-pose inputs required by the selected model.
    - Convert gaze vector output into normalized raw gaze coordinates.
    """

    backend_name = "mobilegaze"

    def __init__(self, model_path: str | None = None) -> None:
        self.repo_path = require_external_repo(self.backend_name)
        self.model_path = self._validate_model_path(model_path)

    def estimate(self, frame: Frame, detection: Any) -> GazeResult:
        if not getattr(detection, "found", False):
            return GazeResult(raw_x=0.5, raw_y=0.5, confidence=0.0, metadata={"backend": self.backend_name})
        raise GazeBackendUnavailable(
            "MobileGaze adapter is prepared but exact external inference wiring is still TODO."
        )

    @staticmethod
    def _validate_model_path(model_path: str | None) -> Path:
        if not model_path:
            raise GazeBackendUnavailable(
                "MobileGaze model_path is missing. Expected: models/gaze/mobilegaze/<model>.onnx."
            )
        path = Path(model_path)
        if not path.exists():
            raise GazeBackendUnavailable(f"MobileGaze model file not found: {path}")
        return path

