from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any


@dataclass(frozen=True)
class DemoTargetSample:
    name: str
    target_x: int
    target_y: int
    observed_x: float
    observed_y: float


@dataclass(frozen=True)
class DemoTuningProfile:
    version: int
    gaze_backend: str
    calibration_profile_path: str
    screen_width: int
    screen_height: int
    samples: list[DemoTargetSample]
    calibration_overrides: dict[str, float | bool | int] = field(default_factory=dict)
    smoothing_overrides: dict[str, float | int | str] = field(default_factory=dict)
    cursor_overrides: dict[str, float | int | str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    quality: str = "good"
    quality_warnings: list[str] = field(default_factory=list)


def save_demo_tuning_profile(profile: DemoTuningProfile, path: str | Path) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(_profile_to_json(profile), handle, indent=2)


def load_demo_tuning_profile(path: str | Path) -> DemoTuningProfile:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return _profile_from_json(data)


def apply_demo_tuning_profile(config: dict[str, Any], profile: DemoTuningProfile) -> None:
    quality_warnings = validate_demo_tuning_profile(profile)
    if profile.quality != "good" or quality_warnings:
        details = "; ".join(quality_warnings or profile.quality_warnings or ["quality is not good"])
        raise ValueError(f"unsafe demo tuning profile: {details}")

    calibration_config = config.setdefault("calibration", {})
    smoothing_config = config.setdefault("smoothing", {})
    cursor_config = config.setdefault("cursor", {})

    calibration_config.update(profile.calibration_overrides)
    smoothing_config.update(profile.smoothing_overrides)
    cursor_config.update(profile.cursor_overrides)
    config.setdefault("demo_tuning", {})["profile_path"] = profile.calibration_profile_path
    config["demo_tuning"]["loaded"] = True


def create_demo_tuning_profile(
    *,
    gaze_backend: str,
    calibration_profile_path: str,
    screen_width: int,
    screen_height: int,
    samples: list[DemoTargetSample],
    current_calibration: dict[str, Any] | None = None,
    current_smoothing: dict[str, Any] | None = None,
    current_cursor: dict[str, Any] | None = None,
    max_offset_px: float = 180.0,
) -> DemoTuningProfile:
    calibration = current_calibration or {}
    smoothing = current_smoothing or {}
    cursor = current_cursor or {}
    notes: list[str] = []

    center_samples = [sample for sample in samples if sample.name == "center"]
    quality_warnings = _quality_warnings(samples, screen_width, screen_height)
    quality = "needs_review" if quality_warnings else "good"
    existing_x_gain = float(calibration.get("horizontal_gain", 1.0) or 1.0)
    existing_y_gain = float(calibration.get("vertical_gain", 1.0) or 1.0)
    existing_x_offset = float(calibration.get("horizontal_offset", 0.0) or 0.0)
    existing_y_offset = float(calibration.get("vertical_offset", 0.0) or 0.0)

    x_correction = None
    y_correction = None
    if quality == "good":
        x_correction = _fit_axis_correction(
            samples,
            observed_name="observed_x",
            target_name="target_x",
            axis_center=max(0.0, float(screen_width - 1)) / 2.0,
            axis_span=max(1.0, float(screen_width - 1)),
            max_offset=max_offset_px,
        )
        y_correction = _fit_axis_correction(
            samples,
            observed_name="observed_y",
            target_name="target_y",
            axis_center=max(0.0, float(screen_height - 1)) / 2.0,
            axis_span=max(1.0, float(screen_height - 1)),
            max_offset=max_offset_px,
        )

    is_multi_target = len({sample.name for sample in samples}) >= 3
    if quality == "good" and is_multi_target and (x_correction is None or y_correction is None):
        quality_warnings.append("multi-target gain/offset correction could not be fitted safely")
        quality = "needs_review"

    if quality == "good" and x_correction is not None and y_correction is not None:
        x_gain, x_offset = x_correction
        y_gain, y_offset = y_correction
        horizontal_gain = existing_x_gain * x_gain
        vertical_gain = existing_y_gain * y_gain
        horizontal_offset = existing_x_offset * x_gain + x_offset
        vertical_offset = existing_y_offset * y_gain + y_offset
        notes.append("Five-point settled samples were used for measured axis gain and offset correction.")
    else:
        horizontal_gain = existing_x_gain
        vertical_gain = existing_y_gain
        horizontal_offset = existing_x_offset
        vertical_offset = existing_y_offset
        if quality == "good":
            offset_basis = center_samples or samples
            raw_offset_x, raw_offset_y = _median_offsets(offset_basis)
            if abs(raw_offset_x) >= max_offset_px or abs(raw_offset_y) >= max_offset_px:
                quality_warnings.append(
                    "center-only correction reaches the offset safety limit; capture is not usable"
                )
                quality = "needs_review"
            else:
                offset_x = _clamp_small(raw_offset_x, max_offset_px)
                offset_y = _clamp_small(raw_offset_y, max_offset_px)
                horizontal_offset += offset_x
                vertical_offset += offset_y
                if not center_samples:
                    notes.append("No center sample was available, so all samples were used for offset estimation.")

    notes.extend(quality_warnings)
    calibration_overrides: dict[str, float | bool | int] = {
        "horizontal_gain": round(horizontal_gain, 4),
        "vertical_gain": round(vertical_gain, 4),
        "horizontal_offset": round(horizontal_offset, 1),
        "vertical_offset": round(vertical_offset, 1),
        "edge_reach_enabled": bool(calibration.get("edge_reach_enabled", True)),
        "edge_boost_enabled": False,
    }

    filter_name = str(smoothing.get("filter", "one_euro")).lower()
    smoothing_overrides: dict[str, float | int | str] = {"filter": filter_name}
    for key, value in smoothing.items():
        if key.startswith(f"{filter_name}_") and isinstance(value, (int, float, str)):
            smoothing_overrides[key] = value

    cursor_overrides: dict[str, float | int | str] = {
        "movement_duration_sec": 0.0,
        "max_speed_px_per_sec": float(cursor.get("max_speed_px_per_sec", 3000) or 3000),
    }
    if not samples:
        notes.append("No samples were recorded; the profile keeps baseline tuning values.")

    return DemoTuningProfile(
        version=3,
        gaze_backend=gaze_backend,
        calibration_profile_path=calibration_profile_path,
        screen_width=screen_width,
        screen_height=screen_height,
        samples=samples,
        calibration_overrides=calibration_overrides,
        smoothing_overrides=smoothing_overrides,
        cursor_overrides=cursor_overrides,
        notes=notes,
        quality=quality,
        quality_warnings=quality_warnings,
    )

def validate_demo_tuning_profile(profile: DemoTuningProfile) -> list[str]:
    return _quality_warnings(profile.samples, profile.screen_width, profile.screen_height)


def _quality_warnings(
    samples: list[DemoTargetSample],
    screen_width: int,
    screen_height: int,
) -> list[str]:
    center_samples = [sample for sample in samples if sample.name == "center"]
    warnings: list[str] = []
    if len(center_samples) < 8:
        warnings.append(f"center capture has only {len(center_samples)} samples; at least 8 are required")
        return warnings

    max_x = max(0.0, float(screen_width - 1))
    max_y = max(0.0, float(screen_height - 1))
    saturated = [
        sample
        for sample in center_samples
        if sample.observed_x <= 1.0
        or sample.observed_x >= max_x - 1.0
        or sample.observed_y <= 1.0
        or sample.observed_y >= max_y - 1.0
    ]
    saturation_ratio = len(saturated) / len(center_samples)
    if saturation_ratio > 0.35:
        warnings.append(f"center capture edge saturation is {saturation_ratio:.0%}; maximum allowed is 35%")

    jitter = _robust_jitter(center_samples)
    max_jitter = max(220.0, min(float(screen_width), float(screen_height)) * 0.22)
    if jitter > max_jitter:
        warnings.append(f"center capture jitter is {jitter:.1f}px; maximum allowed is {max_jitter:.1f}px")

    grouped: dict[str, list[DemoTargetSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.name, []).append(sample)
    if len(grouped) >= 3:
        short_targets = sorted(name for name, values in grouped.items() if len(values) < 8)
        if short_targets:
            warnings.append(f"targets with fewer than 8 samples: {', '.join(short_targets)}")

        medians = {
            name: (
                float(median([sample.observed_x for sample in values])),
                float(median([sample.observed_y for sample in values])),
            )
            for name, values in grouped.items()
            if len(values) >= 8
        }
        center_median = medians.get("center")
        if center_median is not None and (
            center_median[0] < max_x * 0.20
            or center_median[0] > max_x * 0.80
            or center_median[1] < max_y * 0.20
            or center_median[1] > max_y * 0.80
        ):
            warnings.append("center target median is outside the central 60% of the screen")

        horizontal_minimum = max_x * 0.15
        vertical_minimum = max_y * 0.15
        for left_name, right_name in (("top_left", "top_right"), ("bottom_left", "bottom_right")):
            if left_name in medians and right_name in medians:
                separation = medians[right_name][0] - medians[left_name][0]
                if separation < horizontal_minimum:
                    warnings.append(f"{left_name}/{right_name} horizontal separation is not usable")
        for top_name, bottom_name in (("top_left", "bottom_left"), ("top_right", "bottom_right")):
            if top_name in medians and bottom_name in medians:
                separation = medians[bottom_name][1] - medians[top_name][1]
                if separation < vertical_minimum:
                    warnings.append(f"{top_name}/{bottom_name} vertical separation is not usable")

        for name, values in grouped.items():
            target_jitter = _robust_jitter(values)
            if target_jitter > max_jitter:
                warnings.append(
                    f"{name} capture jitter is {target_jitter:.1f}px; maximum allowed is {max_jitter:.1f}px"
                )
    return warnings


def _fit_axis_correction(
    samples: list[DemoTargetSample],
    *,
    observed_name: str,
    target_name: str,
    axis_center: float,
    axis_span: float,
    max_offset: float,
) -> tuple[float, float] | None:
    grouped: dict[str, list[DemoTargetSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.name, []).append(sample)

    points: list[tuple[float, float]] = []
    for target_samples in grouped.values():
        if len(target_samples) < 8:
            continue
        observed = median([float(getattr(sample, observed_name)) for sample in target_samples])
        target = median([float(getattr(sample, target_name)) for sample in target_samples])
        points.append((observed, target))

    if len(points) < 3:
        return None
    observed_values = [point[0] for point in points]
    target_values = [point[1] for point in points]
    if max(target_values) - min(target_values) < axis_span * 0.50:
        return None
    if max(observed_values) - min(observed_values) < axis_span * 0.25:
        return None

    observed_mean = sum(observed_values) / len(observed_values)
    target_mean = sum(target_values) / len(target_values)
    variance = sum((value - observed_mean) ** 2 for value in observed_values)
    if variance <= 1e-6:
        return None
    gain = sum(
        (observed - observed_mean) * (target - target_mean)
        for observed, target in points
    ) / variance
    intercept = target_mean - gain * observed_mean
    offset = intercept - axis_center * (1.0 - gain)

    # Large corrections indicate a bad capture or changed camera posture, not tuning.
    if not 0.70 <= gain <= 1.30 or abs(offset) > max_offset:
        return None
    return float(gain), _clamp_small(float(offset), max_offset)

def _median_offsets(samples: list[DemoTargetSample]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    x_errors = [float(sample.target_x) - float(sample.observed_x) for sample in samples]
    y_errors = [float(sample.target_y) - float(sample.observed_y) for sample in samples]
    return float(median(x_errors)), float(median(y_errors))


def _robust_jitter(samples: list[DemoTargetSample]) -> float:
    if len(samples) < 3:
        return 0.0
    center_x = median([sample.observed_x for sample in samples])
    center_y = median([sample.observed_y for sample in samples])
    distances = sorted(
        ((sample.observed_x - center_x) ** 2 + (sample.observed_y - center_y) ** 2) ** 0.5
        for sample in samples
    )
    index = min(len(distances) - 1, round((len(distances) - 1) * 0.90))
    return float(distances[index])


def _clamp_small(value: float, max_abs: float, deadzone: float = 8.0) -> float:
    if abs(value) < deadzone:
        return 0.0
    return max(-max_abs, min(max_abs, float(value)))


def _profile_to_json(profile: DemoTuningProfile) -> dict[str, Any]:
    return asdict(profile)


def _profile_from_json(data: dict[str, Any]) -> DemoTuningProfile:
    samples = [DemoTargetSample(**sample) for sample in data.get("samples", [])]
    screen_width = int(data.get("screen_width", 0))
    screen_height = int(data.get("screen_height", 0))
    derived_warnings = _quality_warnings(samples, screen_width, screen_height)
    quality_warnings = [str(warning) for warning in data.get("quality_warnings", derived_warnings)]
    quality = str(data.get("quality", "needs_review" if derived_warnings else "good"))
    return DemoTuningProfile(
        version=int(data.get("version", 1)),
        gaze_backend=str(data.get("gaze_backend", "unknown")),
        calibration_profile_path=str(data.get("calibration_profile_path", "")),
        screen_width=screen_width,
        screen_height=screen_height,
        samples=samples,
        calibration_overrides=dict(data.get("calibration_overrides") or {}),
        smoothing_overrides=dict(data.get("smoothing_overrides") or {}),
        cursor_overrides=dict(data.get("cursor_overrides") or {}),
        notes=[str(note) for note in data.get("notes", [])],
        quality=quality,
        quality_warnings=quality_warnings,
    )