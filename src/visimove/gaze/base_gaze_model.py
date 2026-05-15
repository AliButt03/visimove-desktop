from __future__ import annotations

from typing import Any, Protocol

from visimove.gaze.gaze_output import GazeResult
from visimove.types import Frame


class BaseGazeModel(Protocol):
    def estimate(self, frame: Frame, detection: Any) -> GazeResult: ...


class GazeBackendUnavailable(RuntimeError):
    """Raised when an optional gaze backend is selected but not ready."""

