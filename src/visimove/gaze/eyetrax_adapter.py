from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from visimove.config import PROJECT_ROOT
from visimove.gaze.base_gaze_model import GazeBackendUnavailable
from visimove.gaze.gaze_output import GazeResult
from visimove.types import Frame
from visimove.utils.screen import get_screen_bounds


EYETRAX_SETUP_MESSAGE = (
    "EyeTrax backend selected, but external/eyetrax is not set up. "
    "Please clone/install EyeTrax first."
)


@dataclass(frozen=True)
class EyeTraxAdapterConfig:
    repo_path: Path = PROJECT_ROOT / "external" / "eyetrax"
    model_path: Path = PROJECT_ROOT / "models" / "gaze" / "eyetrax"
    face_landmarker_model_path: Path | None = PROJECT_ROOT / "models" / "detection" / "face_landmarker.task"
    use_gpu: bool = False
    input_size: int | None = None


@dataclass(frozen=True)
class EyeTraxSetupStatus:
    ready: bool
    reason: str = "ready"
    repo_path: Path | None = None
    model_path: Path | None = None
    face_landmarker_model_path: Path | None = None


@dataclass(frozen=True)
class EyeTraxTrace:
    face_detected: bool
    landmark_count: int | None
    feature_count: int
    feature_preview: tuple[float, ...]
    blink_detected: bool
    prediction: tuple[float, float] | None
    gaze_result: GazeResult
    invalid_reason: str | None = None


class EyeTraxAdapter:
    """Adapter for the optional EyeTrax repository in `external/eyetrax`.

    EyeTrax predicts screen coordinates from features extracted by its own
    `GazeEstimator`. This adapter only returns real EyeTrax output after the
    external package, a calibrated gaze model, and a local FaceLandmarker task
    model are all available.
    """

    backend_name = "eyetrax"

    def __init__(
        self,
        model_path: str | Path | None = None,
        repo_path: str | Path | None = None,
        face_landmarker_model_path: str | Path | None = None,
        use_gpu: bool = False,
        input_size: int | None = None,
    ) -> None:
        self.config = EyeTraxAdapterConfig(
            repo_path=resolve_project_path(repo_path or "external/eyetrax"),
            model_path=resolve_project_path(model_path or "models/gaze/eyetrax"),
            face_landmarker_model_path=resolve_optional_project_path(
                face_landmarker_model_path or "models/detection/face_landmarker.task"
            ),
            use_gpu=use_gpu,
            input_size=input_size,
        )
        status = self.check_setup(
            repo_path=self.config.repo_path,
            model_path=self.config.model_path,
            face_landmarker_model_path=self.config.face_landmarker_model_path,
        )
        if not status.ready:
            raise GazeBackendUnavailable(status.reason)

        self.model_file = status.model_path
        self.face_landmarker_model_path = status.face_landmarker_model_path
        self._estimator: Any | None = None
        self._screen = get_screen_bounds()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EyeTraxAdapter":
        return cls(
            model_path=config.get("model_path"),
            repo_path=config.get("repo_path"),
            face_landmarker_model_path=config.get("face_landmarker_model_path"),
            use_gpu=bool(config.get("use_gpu", False)),
            input_size=config.get("input_size"),
        )

    @classmethod
    def check_setup(
        cls,
        repo_path: str | Path | None = None,
        model_path: str | Path | None = None,
        face_landmarker_model_path: str | Path | None = None,
    ) -> EyeTraxSetupStatus:
        resolved_repo = resolve_project_path(repo_path or "external/eyetrax")
        if not resolved_repo.exists() or not resolved_repo.is_dir():
            return EyeTraxSetupStatus(False, EYETRAX_SETUP_MESSAGE, repo_path=resolved_repo)

        package_dir = resolved_repo / "src" / "eyetrax"
        if not (package_dir / "__init__.py").exists() or not (package_dir / "gaze.py").exists():
            return EyeTraxSetupStatus(False, EYETRAX_SETUP_MESSAGE, repo_path=resolved_repo)

        resolved_model = resolve_model_file(model_path or "models/gaze/eyetrax")
        if resolved_model is None:
            return EyeTraxSetupStatus(
                False,
                "EyeTrax model file not found. Expected a calibrated model such as "
                "models/gaze/eyetrax/gaze_model.pkl.",
                repo_path=resolved_repo,
            )

        resolved_face_model = resolve_optional_project_path(
            face_landmarker_model_path or "models/detection/face_landmarker.task"
        )
        if resolved_face_model is None or not resolved_face_model.exists():
            return EyeTraxSetupStatus(
                False,
                "EyeTrax FaceLandmarker model not found. Expected "
                "models/detection/face_landmarker.task. VisiMove will not let EyeTrax "
                "auto-download this model during tracking.",
                repo_path=resolved_repo,
                model_path=resolved_model,
                face_landmarker_model_path=resolved_face_model,
            )

        import_error = cls._validate_importable(resolved_repo)
        if import_error is not None:
            return EyeTraxSetupStatus(
                False,
                f"EyeTrax dependencies are not importable: {import_error}",
                repo_path=resolved_repo,
                model_path=resolved_model,
                face_landmarker_model_path=resolved_face_model,
            )

        return EyeTraxSetupStatus(
            True,
            repo_path=resolved_repo,
            model_path=resolved_model,
            face_landmarker_model_path=resolved_face_model,
        )

    @staticmethod
    def _validate_importable(repo_path: Path) -> str | None:
        src_path = str(repo_path / "src")
        added = False
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            added = True
        try:
            module = importlib.import_module("eyetrax")
            getattr(module, "GazeEstimator")
        except Exception as exc:
            return str(exc)
        finally:
            if added:
                try:
                    sys.path.remove(src_path)
                except ValueError:
                    pass
        return None

    def estimate(self, frame: Frame, detection: Any) -> GazeResult:
        return self.trace_frame(frame, detection, include_landmark_count=False).gaze_result

    def trace_frame(
        self,
        frame: Frame,
        detection: Any,
        include_landmark_count: bool = True,
    ) -> EyeTraxTrace:
        started = perf_counter()
        if not getattr(detection, "found", False):
            result = self._empty_result(started, "no face detected")
            return EyeTraxTrace(
                face_detected=False,
                landmark_count=None,
                feature_count=0,
                feature_preview=(),
                blink_detected=False,
                prediction=None,
                gaze_result=result,
                invalid_reason="no face detected",
            )
        try:
            estimator = self._ensure_estimator()
        except GazeBackendUnavailable as exc:
            result = self._empty_result(started, str(exc))
            return EyeTraxTrace(
                face_detected=False,
                landmark_count=None,
                feature_count=0,
                feature_preview=(),
                blink_detected=False,
                prediction=None,
                gaze_result=result,
                invalid_reason=str(exc),
            )
        landmark_count = self._landmark_count(frame) if include_landmark_count else None
        features, blink = estimator.extract_features(frame)
        if features is None:
            result = self._empty_result(started, "EyeTrax features unavailable")
            return EyeTraxTrace(
                face_detected=False,
                landmark_count=landmark_count,
                feature_count=0,
                feature_preview=(),
                blink_detected=False,
                prediction=None,
                gaze_result=result,
                invalid_reason="EyeTrax features unavailable",
            )
        if blink:
            result = self._empty_result(started, "EyeTrax detected blink")
            return EyeTraxTrace(
                face_detected=True,
                landmark_count=landmark_count,
                feature_count=len(features),
                feature_preview=tuple(float(value) for value in features[:8]),
                blink_detected=True,
                prediction=None,
                gaze_result=result,
                invalid_reason="EyeTrax detected blink",
            )

        prediction = np.asarray(estimator.predict([features]), dtype=float)
        if prediction.size < 2:
            result = self._empty_result(started, "EyeTrax prediction did not contain x,y")
            return EyeTraxTrace(
                face_detected=True,
                landmark_count=landmark_count,
                feature_count=len(features),
                feature_preview=tuple(float(value) for value in features[:8]),
                blink_detected=False,
                prediction=None,
                gaze_result=result,
                invalid_reason="EyeTrax prediction did not contain x,y",
            )
        screen_x = float(prediction.reshape(-1, 2)[0, 0])
        screen_y = float(prediction.reshape(-1, 2)[0, 1])
        raw_x = clamp01(screen_x / max(1, self._screen.width - 1))
        raw_y = clamp01(screen_y / max(1, self._screen.height - 1))
        elapsed_ms = (perf_counter() - started) * 1000
        result = GazeResult(
            raw_x=raw_x,
            raw_y=raw_y,
            screen_x=round(screen_x),
            screen_y=round(screen_y),
            confidence=1.0,
            inference_time_ms=elapsed_ms,
            metadata={
                "backend": self.backend_name,
                "initialized": True,
                "fallback": False,
                "screen_prediction": (screen_x, screen_y),
                "model_path": str(self.model_file),
            },
        )
        return EyeTraxTrace(
            face_detected=True,
            landmark_count=landmark_count,
            feature_count=len(features),
            feature_preview=tuple(float(value) for value in features[:8]),
            blink_detected=False,
            prediction=(screen_x, screen_y),
            gaze_result=result,
        )

    def _landmark_count(self, frame: Frame) -> int | None:
        if self._estimator is None:
            return None
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb = np.ascontiguousarray(image_rgb)
            mp_image = self._estimator._mp.Image(
                image_format=self._estimator._mp.ImageFormat.SRGB,
                data=image_rgb,
            )
            ts_ms = int(perf_counter() * 1000)
            if ts_ms <= self._estimator._mp_last_ts_ms:
                ts_ms = self._estimator._mp_last_ts_ms + 1
            self._estimator._mp_last_ts_ms = ts_ms
            result = self._estimator._face_landmarker.detect_for_video(mp_image, ts_ms)
            if not result.face_landmarks:
                return 0
            return len(result.face_landmarks[0])
        except Exception:
            return None

    def _ensure_estimator(self) -> Any:
        if self._estimator is not None:
            return self._estimator

        src_path = str(self.config.repo_path / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        try:
            from eyetrax import GazeEstimator  # type: ignore
        except Exception as exc:
            raise GazeBackendUnavailable(f"EyeTrax import failed during initialization: {exc}") from exc

        try:
            estimator = GazeEstimator(face_landmarker_model=str(self.face_landmarker_model_path))
            estimator.load_model(str(self.model_file))
        except Exception as exc:
            raise GazeBackendUnavailable(f"EyeTrax initialization failed: {exc}") from exc

        self._estimator = estimator
        print("EyeTrax backend initialized successfully.")
        return estimator

    def _empty_result(self, started: float, reason: str) -> GazeResult:
        elapsed_ms = (perf_counter() - started) * 1000
        return GazeResult(
            raw_x=0.5,
            raw_y=0.5,
            confidence=0.0,
            inference_time_ms=elapsed_ms,
            metadata={
                "backend": self.backend_name,
                "initialized": self._estimator is not None,
                "fallback": False,
                "unavailable_reason": reason,
            },
        )


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_optional_project_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    return resolve_project_path(path_value)


def resolve_model_file(path_value: str | Path) -> Path | None:
    path = resolve_project_path(path_value)
    if path.is_file():
        return path
    if path.is_dir():
        candidate = path / "gaze_model.pkl"
        return candidate if candidate.exists() else None
    if path.suffix:
        return path if path.exists() else None
    candidate = path / "gaze_model.pkl"
    return candidate if candidate.exists() else None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
