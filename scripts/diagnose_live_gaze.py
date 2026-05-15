from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys
from statistics import mean
from time import monotonic
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visimove.calibration import AxisAdjustmentConfig, LiveTrackingQualityMonitor, apply_axis_adjustment
from visimove.camera.base import CameraConfig
from visimove.camera.webcam import WebcamCamera
from visimove.config import load_config
from visimove.pipeline.realtime_pipeline import (
    build_calibration_mapper,
    build_detector,
    build_gaze_backend_config,
    build_gaze_model,
    build_smoothing_filter,
)


TARGETS = ("left", "center", "right", "top", "bottom")


@dataclass
class LiveSample:
    raw_x: float
    raw_y: float
    mapped_x: int
    mapped_y: int
    adjusted_x: int
    adjusted_y: int
    smoothed_x: int
    smoothed_y: int
    confidence: float
    domain_violation: bool
    raw_x_saturated: bool


@dataclass
class TargetStats:
    name: str
    samples: list[LiveSample] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    def values(self, attr: str) -> list[float]:
        return [float(getattr(sample, attr)) for sample in self.samples]

    def mean(self, attr: str) -> float:
        values = self.values(attr)
        return mean(values) if values else 0.0

    def ratio(self, attr: str) -> float:
        if not self.samples:
            return 0.0
        return sum(1 for sample in self.samples if getattr(sample, attr)) / len(self.samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose live VisiMove gaze coverage and cursor bias.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--gaze-backend", choices=("dummy", "eyetrax", "gazefollower", "mobilegaze"), default=None)
    parser.add_argument("--detector-backend", choices=("dummy", "opencv", "mediapipe", "yolo"), default=None)
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--calibration-profile", default=None)
    parser.add_argument("--guided", action="store_true", help="Collect left/center/right/top/bottom guided samples.")
    parser.add_argument("--seconds-per-target", type=float, default=2.0)
    parser.add_argument("--horizontal-gain", type=float, default=None)
    parser.add_argument("--vertical-gain", type=float, default=None)
    parser.add_argument("--horizontal-offset", type=float, default=None)
    parser.add_argument("--vertical-offset", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config, "config/performance.yaml")
    apply_cli_overrides(config, args)
    targets = TARGETS if args.guided else ("center",)

    mapper = build_calibration_mapper(config.get("calibration", {}))
    camera = WebcamCamera(
        CameraConfig(**config.get("camera", {})),
        resize_width=config.get("performance", {}).get("resize_width"),
    )
    detector = build_detector(config.get("detection", {}))
    gaze_model = build_gaze_model(build_gaze_backend_config(config))
    smoother = build_smoothing_filter(config.get("smoothing", {}))
    axis_config = build_axis_config(config.get("calibration", {}))
    live_quality_monitor = LiveTrackingQualityMonitor(
        window_size=int(config.get("calibration", {}).get("live_domain_window_size", 30)),
        unsafe_ratio_threshold=float(
            config.get("calibration", {}).get("live_domain_violation_ratio_threshold", 0.30)
        ),
    )

    print("Live gaze diagnostics")
    print(f"  gaze_backend: {config.get('gaze', {}).get('gaze_backend', config.get('gaze', {}).get('backend'))}")
    print(f"  detector_backend: {config.get('detection', {}).get('detector_backend')}")
    print(f"  calibration profile: {config.get('calibration', {}).get('profile_path', 'none')}")
    print(f"  screen size: {mapper.screen_width}x{mapper.screen_height}")
    print(
        "  axis tuning: "
        f"horizontal_gain={axis_config.horizontal_gain:.2f}, "
        f"vertical_gain={axis_config.vertical_gain:.2f}, "
        f"horizontal_offset={axis_config.horizontal_offset:.1f}, "
        f"vertical_offset={axis_config.vertical_offset:.1f}"
    )

    all_stats: list[TargetStats] = []
    camera.open()
    try:
        for target in targets:
            if args.guided:
                input(f"\nLook at the {target.upper()} side/area, then press Enter to collect samples...")
            reset_smoother(smoother)
            stats = collect_target_stats(
                name=target,
                seconds=args.seconds_per_target,
                camera=camera,
                detector=detector,
                gaze_model=gaze_model,
                mapper=mapper,
                smoother=smoother,
                axis_config=axis_config,
                live_quality_monitor=live_quality_monitor,
            )
            all_stats.append(stats)
            print_target_stats(stats)
    finally:
        camera.close()

    print_diagnosis(all_stats, mapper.screen_width)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.camera_index is not None:
        config.setdefault("camera", {})["index"] = args.camera_index
    if args.detector_backend is not None:
        config.setdefault("detection", {})["detector_backend"] = args.detector_backend
        config.setdefault("detection", {})["backend"] = args.detector_backend
    if args.gaze_backend is not None:
        config.setdefault("gaze", {})["gaze_backend"] = args.gaze_backend
        config.setdefault("gaze", {})["backend"] = args.gaze_backend

    gaze_backend = str(config.get("gaze", {}).get("gaze_backend", "dummy")).lower()
    calibration_config = config.setdefault("calibration", {})
    if args.calibration_profile:
        calibration_config["profile_path"] = str(resolve_project_path(args.calibration_profile))
    elif gaze_backend == "eyetrax":
        eyetrax_profile = PROJECT_ROOT / "data" / "calibration" / "user_profile_eyetrax.json"
        if eyetrax_profile.exists():
            calibration_config["profile_path"] = str(eyetrax_profile)

    if calibration_config.get("profile_path"):
        calibration_config["profile_path"] = str(resolve_project_path(calibration_config["profile_path"]))
    if args.horizontal_gain is not None:
        calibration_config["horizontal_gain"] = args.horizontal_gain
    if args.vertical_gain is not None:
        calibration_config["vertical_gain"] = args.vertical_gain
    if args.horizontal_offset is not None:
        calibration_config["horizontal_offset"] = args.horizontal_offset
    if args.vertical_offset is not None:
        calibration_config["vertical_offset"] = args.vertical_offset


def build_axis_config(config: dict[str, Any]) -> AxisAdjustmentConfig:
    return AxisAdjustmentConfig(
        horizontal_gain=float(config.get("horizontal_gain", 1.0)),
        vertical_gain=float(config.get("vertical_gain", 1.0)),
        horizontal_offset=float(config.get("horizontal_offset", 0.0)),
        vertical_offset=float(config.get("vertical_offset", 0.0)),
        enabled=bool(config.get("center_bias_correction", True)),
    )


def collect_target_stats(
    name: str,
    seconds: float,
    camera: WebcamCamera,
    detector: Any,
    gaze_model: Any,
    mapper: Any,
    smoother: Any,
    axis_config: AxisAdjustmentConfig,
    live_quality_monitor: LiveTrackingQualityMonitor,
) -> TargetStats:
    stats = TargetStats(name=name)
    deadline = monotonic() + max(0.5, seconds)
    while monotonic() < deadline:
        now = monotonic()
        webcam_frame = camera.read()
        if webcam_frame is None:
            continue
        frame = webcam_frame.frame
        detection = detector.detect(frame, now)
        gaze = gaze_model.estimate(frame, detection)
        mapped, mapping_debug = mapper.map_with_debug(gaze)
        live_quality_monitor.update(mapping_debug)
        adjustment = apply_axis_adjustment(mapped, mapper.screen_width, mapper.screen_height, axis_config)
        smoothed_xy = smoother.update(adjustment.after.x, adjustment.after.y, now)
        if smoothed_xy is None:
            smoothed_x, smoothed_y = adjustment.after.x, adjustment.after.y
        else:
            smoothed_x, smoothed_y = round(smoothed_xy[0]), round(smoothed_xy[1])
        stats.samples.append(
            LiveSample(
                raw_x=gaze.point.x,
                raw_y=gaze.point.y,
                mapped_x=mapped.x,
                mapped_y=mapped.y,
                adjusted_x=adjustment.after.x,
                adjusted_y=adjustment.after.y,
                smoothed_x=smoothed_x,
                smoothed_y=smoothed_y,
                confidence=gaze.confidence,
                domain_violation=mapping_debug.raw_domain_status == "outside",
                raw_x_saturated=gaze.point.x >= 0.98,
            )
        )
    return stats


def print_target_stats(stats: TargetStats) -> None:
    print(f"\n{stats.name.upper()} samples: {stats.count}")
    for label, attr in (
        ("raw_x", "raw_x"),
        ("raw_y", "raw_y"),
        ("mapped_x", "mapped_x"),
        ("mapped_y", "mapped_y"),
        ("smoothed_x", "smoothed_x"),
        ("smoothed_y", "smoothed_y"),
    ):
        print(f"  {label}: {format_range(stats.values(attr))}")
    print(f"  confidence mean: {stats.mean('confidence'):.3f}")
    print(f"  live domain violations: {stats.ratio('domain_violation'):.0%}")
    print(f"  raw_x saturation: {stats.ratio('raw_x_saturated'):.0%}")


def print_diagnosis(all_stats: list[TargetStats], screen_width: int) -> None:
    stats_by_name = {stats.name: stats for stats in all_stats}
    warnings: list[str] = []
    left = stats_by_name.get("left")
    center = stats_by_name.get("center")
    right = stats_by_name.get("right")
    if left and center and left.count and center.count:
        left_separation = center.mean("mapped_x") - left.mean("mapped_x")
        if left_separation < screen_width * 0.15:
            warnings.append("left gaze separation is weak")
    if left and center and right and left.count and center.count and right.count:
        left_strength = max(1.0, center.mean("mapped_x") - left.mean("mapped_x"))
        right_strength = right.mean("mapped_x") - center.mean("mapped_x")
        if right_strength > left_strength * 1.5:
            warnings.append("x-axis right bias")
    all_mapped_x = [value for stats in all_stats for value in stats.values("mapped_x")]
    if all_mapped_x and max(all_mapped_x) - min(all_mapped_x) < screen_width * 0.60:
        warnings.append("horizontal range too small")
    all_samples = [sample for stats in all_stats for sample in stats.samples]
    if all_samples:
        saturation_ratio = sum(1 for sample in all_samples if sample.raw_x_saturated) / len(all_samples)
        if saturation_ratio > 0.30:
            warnings.append("raw_x saturation")

    print("\nDiagnosis")
    if warnings:
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("  No strong horizontal bias warning detected.")
    print("  Try preview tuning first, for example: --horizontal-gain 1.2 or --horizontal-gain 1.3")


def format_range(values: list[float]) -> str:
    if not values:
        return "min=none max=none mean=none range=0.000"
    minimum = min(values)
    maximum = max(values)
    return f"min={minimum:.3f} max={maximum:.3f} mean={mean(values):.3f} range={maximum - minimum:.3f}"


def reset_smoother(smoother: Any) -> None:
    reset = getattr(smoother, "reset", None)
    if callable(reset):
        reset()


def resolve_project_path(path_value: object) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
