from __future__ import annotations

from dataclasses import dataclass
from importlib import util
import math
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from typing import Any

import numpy as np

from visimove.config import PROJECT_ROOT
from visimove.external_backends import ExternalBackendUnavailable, require_external_repo
from visimove.gaze.base_gaze_model import GazeBackendUnavailable
from visimove.gaze.gaze_output import GazeResult
from visimove.types import Frame


@dataclass(frozen=True)
class GazeFollowerSetupStatus:
    ready: bool
    reason: str
    repo_path: Path | None = None
    model_path: Path | None = None
    face_model_path: Path | None = None
    missing_dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class GazeFollowerAdapterConfig:
    repo_path: Path = PROJECT_ROOT / "external" / "gazefollower"
    model_path: Path | None = None
    face_model_path: Path | None = None
    face_alignment_backend: str = "blazeface"
    use_calibrated_output: bool = False
    native_output_mode: str = "model_coordinates"
    native_coordinate_scale_x: float = 10.0
    native_coordinate_scale_y: float = 10.0


class GazeFollowerAdapter:
    """GazeFollower backend wrapper.

    GazeFollower is kept isolated under `external/gazefollower`. This adapter
    validates the repo, bundled MNN weights, and runtime dependencies before
    importing its code. If setup is incomplete, callers should fall back to the
    dummy gaze model rather than pretending real gaze is available.
    """

    backend_name = "gazefollower"

    def __init__(
        self,
        model_path: str | Path | None = None,
        repo_path: str | Path | None = None,
        face_model_path: str | Path | None = None,
        face_alignment_backend: str = "blazeface",
        use_calibrated_output: bool = False,
        native_output_mode: str = "model_coordinates",
        native_coordinate_scale_x: float = 10.0,
        native_coordinate_scale_y: float = 10.0,
    ) -> None:
        self.config = GazeFollowerAdapterConfig(
            repo_path=_resolve_path(repo_path or PROJECT_ROOT / "external" / "gazefollower"),
            model_path=_resolve_optional_path(model_path),
            face_model_path=_resolve_optional_path(face_model_path),
            face_alignment_backend=face_alignment_backend.lower(),
            use_calibrated_output=use_calibrated_output,
            native_output_mode=native_output_mode,
            native_coordinate_scale_x=max(float(native_coordinate_scale_x), 1e-6),
            native_coordinate_scale_y=max(float(native_coordinate_scale_y), 1e-6),
        )
        status = self.check_setup(
            repo_path=self.config.repo_path,
            model_path=self.config.model_path,
            face_model_path=self.config.face_model_path,
            face_alignment_backend=self.config.face_alignment_backend,
        )
        if not status.ready:
            raise GazeBackendUnavailable(status.reason)

        self.repo_path = status.repo_path or self.config.repo_path
        self.model_path = status.model_path
        self.face_model_path = status.face_model_path
        self._face_alignment: Any | None = None
        self._gaze_estimator: Any | None = None
        self._initialized = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GazeFollowerAdapter:
        return cls(
            repo_path=config.get("repo_path", "external/gazefollower"),
            model_path=config.get("model_path"),
            face_model_path=config.get("face_model_path"),
            face_alignment_backend=str(config.get("face_alignment_backend", "blazeface")),
            use_calibrated_output=bool(config.get("use_calibrated_output", False)),
            native_output_mode=str(config.get("native_output_mode", "model_coordinates")),
            native_coordinate_scale_x=float(config.get("native_coordinate_scale_x", 10.0)),
            native_coordinate_scale_y=float(config.get("native_coordinate_scale_y", 10.0)),
        )

    @classmethod
    def check_setup(
        cls,
        repo_path: str | Path = PROJECT_ROOT / "external" / "gazefollower",
        model_path: str | Path | None = None,
        face_model_path: str | Path | None = None,
        face_alignment_backend: str = "blazeface",
    ) -> GazeFollowerSetupStatus:
        repo = _resolve_path(repo_path)
        try:
            repo = require_external_repo(cls.backend_name) if repo == PROJECT_ROOT / "external" / cls.backend_name else repo
        except ExternalBackendUnavailable as exc:
            return GazeFollowerSetupStatus(
                ready=False,
                reason=(
                    "GazeFollower backend selected, but external/gazefollower is not set up. "
                    "Please clone/install GazeFollower first."
                ),
                repo_path=repo,
            )

        if not repo.exists():
            return GazeFollowerSetupStatus(
                ready=False,
                reason=(
                    "GazeFollower backend selected, but external/gazefollower is not set up. "
                    "Please clone/install GazeFollower first."
                ),
                repo_path=repo,
            )

        package_dir = repo / "gazefollower"
        required_files = (
            package_dir / "__init__.py",
            package_dir / "GazeFollower.py",
            package_dir / "gaze_estimator" / "MGazeNetGazeEstimator.py",
        )
        missing_files = [str(path) for path in required_files if not path.exists()]
        if missing_files:
            return GazeFollowerSetupStatus(
                ready=False,
                reason="GazeFollower repo is incomplete. Missing: " + ", ".join(missing_files),
                repo_path=repo,
            )

        resolved_model_path = _resolve_optional_path(model_path) or package_dir / "res" / "model_weights" / "base.mnn"
        if not resolved_model_path.exists():
            return GazeFollowerSetupStatus(
                ready=False,
                reason=f"GazeFollower MNN model file not found: {resolved_model_path}",
                repo_path=repo,
                model_path=resolved_model_path,
            )

        backend = face_alignment_backend.lower()
        resolved_face_model = _resolve_optional_path(face_model_path)
        if backend == "blazeface":
            resolved_face_model = resolved_face_model or package_dir / "res" / "model_weights" / "blaze_face.mnn"
            if not resolved_face_model.exists():
                return GazeFollowerSetupStatus(
                    ready=False,
                    reason=f"GazeFollower BlazeFace model file not found: {resolved_face_model}",
                    repo_path=repo,
                    model_path=resolved_model_path,
                    face_model_path=resolved_face_model,
                )
        elif backend != "mediapipe":
            return GazeFollowerSetupStatus(
                ready=False,
                reason=f"Unsupported GazeFollower face_alignment_backend: {face_alignment_backend}",
                repo_path=repo,
                model_path=resolved_model_path,
                face_model_path=resolved_face_model,
            )

        missing_dependencies = tuple(
            name for name in ("MNN", "pygame", "screeninfo", "pandas") if util.find_spec(name) is None
        )
        if missing_dependencies:
            return GazeFollowerSetupStatus(
                ready=False,
                reason=(
                    "GazeFollower dependencies are missing: "
                    + ", ".join(missing_dependencies)
                    + ". Install them in the selected virtual environment before using this backend."
                ),
                repo_path=repo,
                model_path=resolved_model_path,
                face_model_path=resolved_face_model,
                missing_dependencies=missing_dependencies,
            )

        return GazeFollowerSetupStatus(
            ready=True,
            reason="ready",
            repo_path=repo,
            model_path=resolved_model_path,
            face_model_path=resolved_face_model,
        )

    def estimate(self, frame: Frame, detection: Any) -> GazeResult:
        started = perf_counter()
        if not getattr(detection, "found", False):
            return self._empty_result(started, "no face from VisiMove detector")

        try:
            self._initialize()
            face_info = self._face_alignment.detect(int(perf_counter() * 1000), frame)
            gaze_info = self._gaze_estimator.detect(frame, face_info)
        except Exception as exc:  # pragma: no cover - depends on optional external runtime
            return self._empty_result(started, f"GazeFollower inference failed: {exc}")

        if not getattr(gaze_info, "status", False):
            tracking_state = getattr(getattr(gaze_info, "tracking_state", None), "name", "unknown")
            return self._empty_result(started, f"GazeFollower tracking_state={tracking_state}")

        native = np.asarray(getattr(gaze_info, "raw_gaze_coordinates", []), dtype=float).reshape(-1)
        if native.size < 2 or not np.isfinite(native[:2]).all():
            return self._empty_result(started, "GazeFollower returned invalid raw coordinates")

        raw_x, raw_y, units = self._normalize_coordinates(
            float(native[0]),
            float(native[1]),
            scale_x=self.config.native_coordinate_scale_x,
            scale_y=self.config.native_coordinate_scale_y,
        )
        features = np.asarray(getattr(gaze_info, "features", []), dtype=float).reshape(-1)
        metadata = {
            "backend": self.backend_name,
            "fallback": False,
            "initialized": self._initialized,
            "model_path": str(self.model_path),
            "face_model_path": str(self.face_model_path) if self.face_model_path else "none",
            "face_alignment_backend": self.config.face_alignment_backend,
            "native_prediction": (float(native[0]), float(native[1])),
            "native_prediction_units": units,
            "normalization_method": "centered_tanh",
            "normalization_scale": (
                self.config.native_coordinate_scale_x,
                self.config.native_coordinate_scale_y,
            ),
            "feature_vector_size": int(features.size),
            "tracking_state": getattr(getattr(gaze_info, "tracking_state", None), "name", "unknown"),
            "left_openness": float(getattr(gaze_info, "left_openness", 0.0)),
            "right_openness": float(getattr(gaze_info, "right_openness", 0.0)),
        }
        return GazeResult(
            raw_x=raw_x,
            raw_y=raw_y,
            confidence=0.8,
            inference_time_ms=(perf_counter() - started) * 1000.0,
            metadata=metadata,
        )

    def _initialize(self) -> None:
        if self._initialized:
            return
        repo = str(self.repo_path)
        if repo not in sys.path:
            sys.path.insert(0, repo)

        if self.config.face_alignment_backend == "blazeface":
            self._prepare_external_imports()
            BlazeFaceAlignment = self._load_blazeface_alignment_class()
            self._face_alignment = BlazeFaceAlignment(model_path=str(self.face_model_path))
        else:
            from gazefollower.face_alignment.MediaPipeFaceAlignment import MediaPipeFaceAlignment

            self._face_alignment = MediaPipeFaceAlignment()

        from gazefollower.gaze_estimator.MGazeNetGazeEstimator import MGazeNetGazeEstimator

        self._gaze_estimator = MGazeNetGazeEstimator(model_path=str(self.model_path))
        self._initialized = True

    def _prepare_external_imports(self) -> None:
        repo = str(self.repo_path)
        if repo not in sys.path:
            sys.path.insert(0, repo)

        package_dir = self.repo_path / "gazefollower"
        root_package = sys.modules.get("gazefollower")
        if root_package is None:
            root_package = ModuleType("gazefollower")
            root_package.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
            sys.modules["gazefollower"] = root_package

        self._prepare_misc_import_shim(package_dir)

        face_alignment_package = ModuleType("gazefollower.face_alignment")
        face_alignment_package.__path__ = [str(package_dir / "face_alignment")]  # type: ignore[attr-defined]
        face_alignment_class = self._load_external_class(
            "gazefollower.face_alignment.FaceAlignment",
            package_dir / "face_alignment" / "FaceAlignment.py",
            "FaceAlignment",
        )
        face_alignment_package.FaceAlignment = face_alignment_class
        sys.modules["gazefollower.face_alignment"] = face_alignment_package

    def _prepare_misc_import_shim(self, package_dir: Path) -> None:
        misc_package = ModuleType("gazefollower.misc")
        misc_package.__path__ = [str(package_dir / "misc")]  # type: ignore[attr-defined]
        sys.modules["gazefollower.misc"] = misc_package

        enumeration_module = self._load_external_module(
            "gazefollower.misc.Enumeration",
            package_dir / "misc" / "Enumeration.py",
        )
        face_info_class = self._load_external_class(
            "gazefollower.misc.FaceInfo",
            package_dir / "misc" / "FaceInfo.py",
            "FaceInfo",
        )
        gaze_info_class = self._load_external_class(
            "gazefollower.misc.GazeInfo",
            package_dir / "misc" / "GazeInfo.py",
            "GazeInfo",
        )

        misc_package.FaceInfo = face_info_class
        misc_package.GazeInfo = gaze_info_class
        misc_package.TrackingState = getattr(enumeration_module, "TrackingState")
        misc_package.EyeMovementEvent = getattr(enumeration_module, "EyeMovementEvent")
        misc_package.clip_patch = _clip_patch

    def _load_blazeface_alignment_class(self) -> type[Any]:
        return self._load_external_class(
            "gazefollower.face_alignment.BlazeFaceAlignment",
            self.repo_path / "gazefollower" / "face_alignment" / "BlazeFaceAlignment.py",
            "BlazeFaceAlignment",
        )

    @staticmethod
    def _load_external_class(module_name: str, path: Path, class_name: str) -> type[Any]:
        module = GazeFollowerAdapter._load_external_module(module_name, path)
        return getattr(module, class_name)

    @staticmethod
    def _load_external_module(module_name: str, path: Path) -> ModuleType:
        loaded = sys.modules.get(module_name)
        if loaded is not None:
            return loaded
        spec = util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise GazeBackendUnavailable(f"Could not load GazeFollower module: {path}")
        module = util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _empty_result(self, started: float, reason: str) -> GazeResult:
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
                "model_path": str(self.model_path) if self.model_path else "none",
                "face_model_path": str(self.face_model_path) if self.face_model_path else "none",
            },
        )

    @staticmethod
    def _normalize_coordinates(
        x: float,
        y: float,
        *,
        scale_x: float = 10.0,
        scale_y: float = 10.0,
    ) -> tuple[float, float, str]:
        """Map GazeFollower's uncalibrated model output into VisiMove's 0..1 raw space.

        In the upstream GazeFollower code, `raw_gaze_coordinates` is `res[:2]`
        from the MNN model output. GazeFollower's own screen output is produced
        later by SVR calibration over the full feature vector, so these native
        coordinates must not be treated as Windows screen pixels.
        """
        safe_scale_x = max(float(scale_x), 1e-6)
        safe_scale_y = max(float(scale_y), 1e-6)
        raw_x = 0.5 + 0.5 * math.tanh(float(x) / safe_scale_x)
        raw_y = 0.5 + 0.5 * math.tanh(float(y) / safe_scale_y)
        return _clamp01(raw_x), _clamp01(raw_y), "gazefollower_model_coordinates"


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def _resolve_optional_path(path: str | Path | None) -> Path | None:
    if path in {None, "", "null"}:
        return None
    return _resolve_path(Path(str(path)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clip_patch(frame: Any, rect: Any) -> Any | None:
    x, y, w, h = rect
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return None
    if x >= frame.shape[1] or y >= frame.shape[0]:
        return None
    x_end = min(x + w, frame.shape[1])
    y_end = min(y + h, frame.shape[0])
    return frame[y:y_end, x:x_end].copy()
