from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from visimove.blink.base_blink_model import BlinkResult, empty_blink_result
from visimove.types import Frame


@dataclass(frozen=True)
class OnnxBlinkModelConfig:
    model_path: str
    input_width: int = 64
    input_height: int = 64
    normalize_mean: float = 0.5
    normalize_std: float = 0.5


class OnnxBlinkModel:
    def __init__(self, config: OnnxBlinkModelConfig) -> None:
        self.config = config
        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Blink model file not found: {model_path}")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is not installed.") from exc

        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def infer(self, frame: Frame, detection: Any) -> BlinkResult:
        left_crop = getattr(detection, "left_eye_crop", None)
        right_crop = getattr(detection, "right_eye_crop", None)
        if not getattr(detection, "found", False) or left_crop is None or right_crop is None:
            return empty_blink_result()

        started = perf_counter()
        left_probability = self._infer_eye(left_crop)
        right_probability = self._infer_eye(right_crop)
        elapsed_ms = (perf_counter() - started) * 1000
        combined = max(left_probability, right_probability)
        confidence = 1.0 if left_probability >= 0.0 and right_probability >= 0.0 else 0.0
        return BlinkResult(
            left_closed_probability=left_probability,
            right_closed_probability=right_probability,
            combined_closed_probability=combined,
            confidence=confidence,
            inference_time_ms=elapsed_ms,
        )

    def _infer_eye(self, crop: Frame) -> float:
        tensor = self.preprocess_eye_crop(crop)
        outputs = self._session.run(None, {self._input_name: tensor})
        output = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if output.size == 1:
            return float(np.clip(output[0], 0.0, 1.0))
        probabilities = self._softmax(output)
        return float(probabilities[-1])

    def preprocess_eye_crop(self, crop: Frame | None) -> np.ndarray:
        if crop is None or crop.size == 0:
            raise ValueError("Eye crop is missing or empty.")
        resized = cv2.resize(
            crop,
            (self.config.input_width, self.config.input_height),
            interpolation=cv2.INTER_AREA,
        )
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self.config.normalize_mean) / max(self.config.normalize_std, 1e-6)
        return np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exp = np.exp(shifted)
        return exp / np.sum(exp)

