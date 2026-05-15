from __future__ import annotations

from visimove.external_backends import require_external_repo
from visimove.types import DetectionResult, Frame, GazeEstimate


class GazeFollowerAdapter:
    """Adapter shell for the optional gazefollower backend."""

    backend_name = "gazefollower"

    def __init__(self) -> None:
        self.repo_path = require_external_repo(self.backend_name)

    def estimate(self, frame: Frame, detection: DetectionResult) -> GazeEstimate:
        raise NotImplementedError("Wire gazefollower inference here after setup is documented.")


class EyeTraxAdapter:
    """Adapter shell for the optional eyetrax backend."""

    backend_name = "eyetrax"

    def __init__(self) -> None:
        self.repo_path = require_external_repo(self.backend_name)

    def estimate(self, frame: Frame, detection: DetectionResult) -> GazeEstimate:
        raise NotImplementedError("Wire eyetrax inference here after setup is documented.")


class MobileGazeAdapter:
    """Adapter shell for the optional mobilegaze backend."""

    backend_name = "mobilegaze"

    def __init__(self) -> None:
        self.repo_path = require_external_repo(self.backend_name)

    def estimate(self, frame: Frame, detection: DetectionResult) -> GazeEstimate:
        raise NotImplementedError("Wire mobilegaze inference here after setup is documented.")

