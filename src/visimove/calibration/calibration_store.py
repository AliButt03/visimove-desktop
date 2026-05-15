from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from visimove.calibration.calibration_points import CalibrationPoint


@dataclass(frozen=True)
class CalibrationSample:
    raw_gaze: tuple[float, float]
    target_screen: tuple[int, int]
    timestamp: float
    confidence: float | None = None


@dataclass(frozen=True)
class CalibrationProfile:
    timestamp: str
    screen_width: int
    screen_height: int
    camera_index: int
    calibration_points: list[CalibrationPoint]
    raw_gaze_samples: list[CalibrationSample]
    target_screen_coordinates: list[tuple[int, int]]
    mapping_model_type: str
    mapping_parameters: dict[str, Any]
    gaze_backend: str = "unknown"
    detector_backend: str = "unknown"
    blink_backend: str = "unknown"
    calibration_point_layout: str = "unknown"
    backend_metadata: dict[str, Any] | None = None
    sample_count_per_point: list[int] | None = None
    confidence_statistics: dict[str, float | None] | None = None
    calibration_quality: str = "unknown"
    calibration_warnings: list[str] | None = None
    quality_metrics: dict[str, Any] | None = None
    raw_x_min: float | None = None
    raw_x_max: float | None = None
    raw_y_min: float | None = None
    raw_y_max: float | None = None
    raw_x_range: float | None = None
    raw_y_range: float | None = None


def new_profile(
    screen_width: int,
    screen_height: int,
    camera_index: int,
    calibration_points: list[CalibrationPoint],
    samples: list[CalibrationSample],
    mapping_model_type: str,
    mapping_parameters: dict[str, Any],
    gaze_backend: str = "unknown",
    detector_backend: str = "unknown",
    blink_backend: str = "unknown",
    calibration_point_layout: str = "unknown",
    backend_metadata: dict[str, Any] | None = None,
    sample_count_per_point: list[int] | None = None,
    confidence_statistics: dict[str, float | None] | None = None,
    calibration_quality: str = "unknown",
    calibration_warnings: list[str] | None = None,
    quality_metrics: dict[str, Any] | None = None,
    raw_x_min: float | None = None,
    raw_x_max: float | None = None,
    raw_y_min: float | None = None,
    raw_y_max: float | None = None,
    raw_x_range: float | None = None,
    raw_y_range: float | None = None,
) -> CalibrationProfile:
    raw_domain = _raw_domain_from_samples(samples)
    return CalibrationProfile(
        timestamp=datetime.now(timezone.utc).isoformat(),
        screen_width=screen_width,
        screen_height=screen_height,
        camera_index=camera_index,
        calibration_points=calibration_points,
        raw_gaze_samples=samples,
        target_screen_coordinates=[sample.target_screen for sample in samples],
        mapping_model_type=mapping_model_type,
        mapping_parameters=mapping_parameters,
        gaze_backend=gaze_backend,
        detector_backend=detector_backend,
        blink_backend=blink_backend,
        calibration_point_layout=calibration_point_layout,
        backend_metadata=backend_metadata or {},
        sample_count_per_point=sample_count_per_point,
        confidence_statistics=confidence_statistics,
        calibration_quality=calibration_quality,
        calibration_warnings=calibration_warnings or [],
        quality_metrics=quality_metrics or {},
        raw_x_min=raw_x_min if raw_x_min is not None else raw_domain["raw_x_min"],
        raw_x_max=raw_x_max if raw_x_max is not None else raw_domain["raw_x_max"],
        raw_y_min=raw_y_min if raw_y_min is not None else raw_domain["raw_y_min"],
        raw_y_max=raw_y_max if raw_y_max is not None else raw_domain["raw_y_max"],
        raw_x_range=raw_x_range if raw_x_range is not None else raw_domain["raw_x_range"],
        raw_y_range=raw_y_range if raw_y_range is not None else raw_domain["raw_y_range"],
    )


def save_calibration_profile(profile: CalibrationProfile, path: str | Path) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(_profile_to_json(profile), handle, indent=2)


def load_calibration_profile(path: str | Path) -> CalibrationProfile:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return _profile_from_json(data)


def _profile_to_json(profile: CalibrationProfile) -> dict[str, Any]:
    return asdict(profile)


def _profile_from_json(data: dict[str, Any]) -> CalibrationProfile:
    points = [CalibrationPoint(**point) for point in data["calibration_points"]]
    samples = [
        CalibrationSample(
            raw_gaze=tuple(sample["raw_gaze"]),
            target_screen=tuple(sample["target_screen"]),
            timestamp=float(sample["timestamp"]),
            confidence=(
                None
                if sample.get("confidence") is None
                else float(sample.get("confidence"))
            ),
        )
        for sample in data["raw_gaze_samples"]
    ]
    targets = [tuple(target) for target in data["target_screen_coordinates"]]
    return CalibrationProfile(
        timestamp=str(data["timestamp"]),
        screen_width=int(data["screen_width"]),
        screen_height=int(data["screen_height"]),
        camera_index=int(data["camera_index"]),
        calibration_points=points,
        raw_gaze_samples=samples,
        target_screen_coordinates=targets,
        mapping_model_type=str(data["mapping_model_type"]),
        mapping_parameters=dict(data["mapping_parameters"]),
        gaze_backend=str(data.get("gaze_backend", "unknown")),
        detector_backend=str(data.get("detector_backend", "unknown")),
        blink_backend=str(data.get("blink_backend", "unknown")),
        calibration_point_layout=str(data.get("calibration_point_layout", "unknown")),
        backend_metadata=dict(data.get("backend_metadata") or {}),
        sample_count_per_point=(
            [int(count) for count in data["sample_count_per_point"]]
            if data.get("sample_count_per_point") is not None
            else None
        ),
        confidence_statistics=dict(data.get("confidence_statistics") or {}),
        calibration_quality=str(data.get("calibration_quality", "unknown")),
        calibration_warnings=list(data.get("calibration_warnings") or []),
        quality_metrics=dict(data.get("quality_metrics") or {}),
        raw_x_min=_optional_float(data.get("raw_x_min", data.get("quality_metrics", {}).get("raw_x_min"))),
        raw_x_max=_optional_float(data.get("raw_x_max", data.get("quality_metrics", {}).get("raw_x_max"))),
        raw_y_min=_optional_float(data.get("raw_y_min", data.get("quality_metrics", {}).get("raw_y_min"))),
        raw_y_max=_optional_float(data.get("raw_y_max", data.get("quality_metrics", {}).get("raw_y_max"))),
        raw_x_range=_optional_float(data.get("raw_x_range", data.get("quality_metrics", {}).get("raw_x_range"))),
        raw_y_range=_optional_float(data.get("raw_y_range", data.get("quality_metrics", {}).get("raw_y_range"))),
    )


def _raw_domain_from_samples(samples: list[CalibrationSample]) -> dict[str, float | None]:
    if not samples:
        return {
            "raw_x_min": None,
            "raw_x_max": None,
            "raw_y_min": None,
            "raw_y_max": None,
            "raw_x_range": None,
            "raw_y_range": None,
        }
    raw_x = [sample.raw_gaze[0] for sample in samples]
    raw_y = [sample.raw_gaze[1] for sample in samples]
    raw_x_min = min(raw_x)
    raw_x_max = max(raw_x)
    raw_y_min = min(raw_y)
    raw_y_max = max(raw_y)
    return {
        "raw_x_min": raw_x_min,
        "raw_x_max": raw_x_max,
        "raw_y_min": raw_y_min,
        "raw_y_max": raw_y_max,
        "raw_x_range": raw_x_max - raw_x_min,
        "raw_y_range": raw_y_max - raw_y_min,
    }


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
