from __future__ import annotations

from visimove.external_backends import require_external_repo
from visimove.types import DetectionResult, Frame


class OcecDetectorAdapter:
    """Adapter shell for an optional OCEC detector backend."""

    backend_name = "ocec"

    def __init__(self) -> None:
        self.repo_path = require_external_repo(self.backend_name)

    def detect(self, frame: Frame) -> DetectionResult:
        raise NotImplementedError("Wire OCEC detection here after the external repo is installed.")

