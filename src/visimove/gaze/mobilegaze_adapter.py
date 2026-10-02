from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from importlib import util
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from visimove.config import PROJECT_ROOT
from visimove.detection.base_detector import clamp_box, expanded_box, safe_crop
from visimove.external_backends import ExternalBackendUnavailable, require_external_repo
from visimove.gaze.base_gaze_model import GazeBackendUnavailable
from visimove.gaze.gaze_output import GazeResult
from visimove.types import Frame


@dataclass(frozen=True)
class MobileGazeSetupStatus:
    ready: bool
    reason: str
    repo_path: Path | None = None
    model_path: Path | None = None
    missing_dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class MobileGazeAdapterConfig:
    repo_path: Path = PROJECT_ROOT / "external" / "mobilegaze"
    model_path: Path = PROJECT_ROOT / "external" / "mobilegaze" / "weights" / "mobileone_s0_gaze.onnx"
    yaw_range_deg: float = 45.0
    pitch_range_deg: float = 35.0
    face_crop_scale: float = 1.15
    temporal_median_window: int = 5
    providers: tuple[str, ...] = ("CPUExecutionProvider",)


class _RawGazeMedianFilter:
    def __init__(self, window_size: int = 5) -> None:
        normalized_size = max(1, int(window_size))
        if normalized_size % 2 == 0:
            normalized_size += 1
        self.window_size = normalized_size
        self._samples: deque[tuple[float, float]] = deque(maxlen=normalized_size)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def update(self, raw_x: float, raw_y: float) -> tuple[float, float]:
        self._samples.append((float(raw_x), float(raw_y)))
        return (
            float(median(sample[0] for sample in self._samples)),
            float(median(sample[1] for sample in self._samples)),
        )

    def reset(self) -> None:
        self._samples.clear()


class MobileGazeAdapter:
    """ONNX adapter for `external/mobilegaze`.

    MobileGaze predicts gaze angles, not screen coordinates. This adapter keeps
    the external repository isolated, runs the ONNX model against the face crop
    supplied by VisiMove's detector, then exposes yaw/pitch as normalized raw
    gaze values for preview and later VisiMove calibration.
    """

    backend_name = "mobilegaze"
    preprocessing_version = "raw_median_v1"

    def __init__(
        self,
        model_path: str | Path | None = None,
        repo_path: str | Path | None = None,
        yaw_range_deg: float = 45.0,
        pitch_range_deg: float = 35.0,
        face_crop_scale: float = 1.15,
        temporal_median_window: int = 5,
        providers: tuple[str, ...] | None = None,
    ) -> None:
        self.config = MobileGazeAdapterConfig(
            repo_path=_resolve_path(repo_path or PROJECT_ROOT / "external" / "mobilegaze"),
            model_path=_resolve_optional_path(model_path)
            or PROJECT_ROOT / "external" / "mobilegaze" / "weights" / "mobileone_s0_gaze.onnx",
            yaw_range_deg=max(float(yaw_range_deg), 1.0),
            pitch_range_deg=max(float(pitch_range_deg), 1.0),
            face_crop_scale=max(float(face_crop_scale), 1.0),
            temporal_median_window=max(1, int(temporal_median_window)),
            providers=providers or ("CPUExecutionProvider",),
        )
        self._raw_gaze_filter = _RawGazeMedianFilter(self.config.temporal_median_window)
        status = self.check_setup(
            repo_path=self.config.repo_path,
            model_path=self.config.model_path,
        )
        if not status.ready:
            raise GazeBackendUnavailable(status.reason)

        self.repo_path = status.repo_path or self.config.repo_path
        self.model_path = status.model_path or self.config.model_path
        self._session: Any | None = None
        self._input_name = ""
        self._input_size = (448, 448)
        self._output_names: list[str] = []
        self._idx_tensor = np.arange(90, dtype=np.float32)
        self._initialized = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MobileGazeAdapter:
        providers_value = config.get("providers", ("CPUExecutionProvider",))
        if isinstance(providers_value, str):
            providers = tuple(part.strip() for part in providers_value.split(",") if part.strip())
        else:
            providers = tuple(providers_value)
        return cls(
            repo_path=config.get("repo_path", "external/mobilegaze"),
            model_path=config.get("model_path"),
            yaw_range_deg=float(config.get("yaw_range_deg", 45.0)),
            pitch_range_deg=float(config.get("pitch_range_deg", 35.0)),
            face_crop_scale=float(config.get("face_crop_scale", 1.15)),
            temporal_median_window=int(config.get("temporal_median_window", 5)),
            providers=providers or ("CPUExecutionProvider",),
        )

    @classmethod
    def check_setup(
        cls,
        repo_path: str | Path = PROJECT_ROOT / "external" / "mobilegaze",
        model_path: str | Path | None = None,
    ) -> MobileGazeSetupStatus:
        repo = _resolve_path(repo_path)
        try:
            repo = require_external_repo(cls.backend_name) if repo == PROJECT_ROOT / "external" / cls.backend_name else repo
        except ExternalBackendUnavailable:
            return MobileGazeSetupStatus(
                ready=False,
                reason=(
                    "MobileGaze backend selected, but external/mobilegaze is not set up. "
                    "Please clone/install MobileGaze first."
                ),
                repo_path=repo,
            )

        if not repo.exists():
            return MobileGazeSetupStatus(
                ready=False,
                reason=(
                    "MobileGaze backend selected, but external/mobilegaze is not set up. "
                    "Please clone/install MobileGaze first."
                ),
                repo_path=repo,
            )

        required_files = (
            repo / "onnx_inference.py",
            repo / "models" / "__init__.py",
            repo / "utils" / "helpers.py",
        )
        missing_files = [str(path) for path in required_files if not path.exists()]
        if missing_files:
            return MobileGazeSetupStatus(
                ready=False,
                reason="MobileGaze repo is incomplete. Missing: " + ", ".join(missing_files),
                repo_path=repo,
            )

        resolved_model_path = _resolve_optional_path(model_path) or repo / "weights" / "mobileone_s0_gaze.onnx"
        if not resolved_model_path.exists():
            return MobileGazeSetupStatus(
                ready=False,
                reason=f"MobileGaze ONNX model file not found: {resolved_model_path}",
                repo_path=repo,
                model_path=resolved_model_path,
            )

        missing_dependencies = tuple(name for name in ("onnxruntime", "cv2") if util.find_spec(name) is None)
        if missing_dependencies:
            return MobileGazeSetupStatus(
                ready=False,
                reason=(
                    "MobileGaze dependencies are missing: "
                    + ", ".join(missing_dependencies)
                    + ". Install them in the selected virtual environment before using this backend."
                ),
                repo_path=repo,
                model_path=resolved_model_path,
                missing_dependencies=missing_dependencies,
            )

        return MobileGazeSetupStatus(
            ready=True,
            reason="ready",
            repo_path=repo,
            model_path=resolved_model_path,
        )

    def estimate(self, frame: Frame, detection: Any) -> GazeResult:
        started = perf_counter()
        if not getattr(detection, "found", False):
            return self._empty_result(started, "no face from VisiMove detector")

        face_crop = self._crop_face(frame, getattr(detection, "face_box", None))
        if face_crop is None or face_crop.size == 0:
            return self._empty_result(started, "MobileGaze face crop is unavailable")

        try:
            self._initialize()
            input_tensor = self._preprocess(face_crop)
            outputs = self._session.run(self._output_names, {self._input_name: input_tensor})  # type: ignore[union-attr]
            yaw_rad, pitch_rad = self._decode(outputs[0], outputs[1])
        except Exception as exc:  # pragma: no cover - depends on optional ONNX runtime
            return self._empty_result(started, f"MobileGaze inference failed: {exc}")

        unfiltered_raw_x, unfiltered_raw_y = self._angles_to_raw(
            yaw_rad,
            pitch_rad,
            yaw_range_deg=self.config.yaw_range_deg,
            pitch_range_deg=self.config.pitch_range_deg,
        )
        raw_x, raw_y = self._raw_gaze_filter.update(unfiltered_raw_x, unfiltered_raw_y)
        gaze_vector = _angles_to_vector(yaw_rad, pitch_rad)
        return GazeResult(
            raw_x=raw_x,
            raw_y=raw_y,
            gaze_vector=gaze_vector,
            confidence=0.75,
            inference_time_ms=(perf_counter() - started) * 1000.0,
            metadata={
                "backend": self.backend_name,
                "fallback": False,
                "initialized": self._initialized,
                "model_path": str(self.model_path),
                "native_prediction": (float(yaw_rad), float(pitch_rad)),
                "native_prediction_units": "radians_yaw_pitch",
                "unfiltered_raw_prediction": (unfiltered_raw_x, unfiltered_raw_y),
                "temporal_median_window": self._raw_gaze_filter.window_size,
                "temporal_sample_count": self._raw_gaze_filter.sample_count,
                "normalization_method": "angle_range_linear",
                "yaw_range_deg": self.config.yaw_range_deg,
                "pitch_range_deg": self.config.pitch_range_deg,
                "face_crop_shape": tuple(int(v) for v in face_crop.shape[:2]),
            },
        )

    def _initialize(self) -> None:
        if self._initialized:
            return
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(self.model_path), providers=list(self.config.providers))
        input_cfg = self._session.get_inputs()[0]
        self._input_name = input_cfg.name
        shape = input_cfg.shape
        if len(shape) >= 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self._input_size = (int(shape[3]), int(shape[2]))
        self._output_names = [output.name for output in self._session.get_outputs()]
        if len(self._output_names) != 2:
            raise GazeBackendUnavailable(
                f"MobileGaze ONNX model should expose 2 outputs, got {len(self._output_names)}."
            )
        self._initialized = True

    def _crop_face(self, frame: Frame, face_box: tuple[int, int, int, int] | None) -> Frame | None:
        if face_box is None:
            return None
        height, width = frame.shape[:2]
        box = expanded_box(face_box, self.config.face_crop_scale, width, height)
        box = box or clamp_box(face_box, width, height)
        crop = safe_crop(frame, box)
        return None if crop is None else crop

    def _preprocess(self, image: Frame) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self._input_size)
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        return np.expand_dims(image, axis=0).astype(np.float32)

    def _decode(self, yaw_logits: np.ndarray, pitch_logits: np.ndarray) -> tuple[float, float]:
        yaw_probs = _softmax(np.asarray(yaw_logits, dtype=np.float32))
        pitch_probs = _softmax(np.asarray(pitch_logits, dtype=np.float32))
        yaw_deg = np.sum(yaw_probs * self._idx_tensor, axis=1) * 4.0 - 180.0
        pitch_deg = np.sum(pitch_probs * self._idx_tensor, axis=1) * 4.0 - 180.0
        return float(np.radians(yaw_deg[0])), float(np.radians(pitch_deg[0]))

    def _empty_result(self, started: float, reason: str) -> GazeResult:
        self._raw_gaze_filter.reset()
        return GazeResult(
            raw_x=0.5,
            raw_y=0.5,
            confidence=0.0,
            inference_time_ms=(perf_counter() - started) * 1000.0,
            metadata={
                "backend": self.backend_name,
                "fallback": False,
                "initialized": self._initialized,
                "unavailable_reason": reason,
                "model_path": str(self.model_path),
            },
        )

    @staticmethod
    def _angles_to_raw(
        yaw_rad: float,
        pitch_rad: float,
        *,
        yaw_range_deg: float = 45.0,
        pitch_range_deg: float = 35.0,
    ) -> tuple[float, float]:
        yaw_range = np.radians(max(float(yaw_range_deg), 1.0))
        pitch_range = np.radians(max(float(pitch_range_deg), 1.0))
        raw_x = 0.5 + float(yaw_rad) / (2.0 * yaw_range)
        raw_y = 0.5 - float(pitch_rad) / (2.0 * pitch_range)
        return _clamp01(raw_x), _clamp01(raw_y)


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    exp = np.exp(values - np.max(values, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _angles_to_vector(yaw_rad: float, pitch_rad: float) -> tuple[float, float, float]:
    x = -np.sin(yaw_rad) * np.cos(pitch_rad)
    y = -np.sin(pitch_rad)
    z = -np.cos(yaw_rad) * np.cos(pitch_rad)
    return float(x), float(y), float(z)


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def _resolve_optional_path(path: str | Path | None) -> Path | None:
    if path in {None, "", "null"}:
        return None
    return _resolve_path(Path(str(path)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
