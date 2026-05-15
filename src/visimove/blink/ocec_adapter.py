from __future__ import annotations

from time import perf_counter
from typing import Any

import cv2
import numpy as np

from visimove.blink.base_blink_model import BlinkResult, empty_blink_result
from visimove.external_backends import require_external_repo
from visimove.types import Frame


class OcecBlinkAdapter:
    """Adapter shell for an optional OCEC blink backend.

    The external repository remains isolated in `external/ocec`. Until its exact
    API is wired, this adapter provides a deterministic crop-based placeholder
    that can be replaced without changing the pipeline.
    """

    backend_name = "ocec"

    def __init__(self) -> None:
        self.repo_path = require_external_repo(self.backend_name)

    def infer(self, frame: Frame, detection: Any) -> BlinkResult:
        left_crop = getattr(detection, "left_eye_crop", None)
        right_crop = getattr(detection, "right_eye_crop", None)
        if not getattr(detection, "found", False) or left_crop is None or right_crop is None:
            return empty_blink_result()

        started = perf_counter()
        left = self._brightness_closed_probability(left_crop)
        right = self._brightness_closed_probability(right_crop)
        combined = max(left, right)
        return BlinkResult(
            left_closed_probability=left,
            right_closed_probability=right,
            combined_closed_probability=combined,
            confidence=0.25,
            inference_time_ms=(perf_counter() - started) * 1000,
        )

    @staticmethod
    def _brightness_closed_probability(crop: Frame) -> float:
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        darkness = 1.0 - float(np.mean(gray) / 255.0)
        return float(np.clip(darkness, 0.0, 1.0))

