from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys
from statistics import mean
from time import monotonic, sleep
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visimove.calibration import CalibrationMapper, load_calibration_profile
from visimove.camera.base import CameraConfig
from visimove.camera.webcam import WebcamCamera
from visimove.config import load_config
from visimove.detection import DetectorResult
from visimove.gaze.eyetrax_adapter import EyeTraxAdapter
from visimove.pipeline.realtime_pipeline import build_calibration_mapper, build_detector


GUIDED_TARGETS = ("far left", "center", "far right", "top", "bottom")


@dataclass
class TraceSample:
    pipeline_face_detected: bool
    eyetrax_face_detected: bool
    landmark_count: int | None
    feature_count: int
    feature_preview: tuple[float, ...]
    prediction_x: float | None
    prediction_y: float | None
    adapter_raw_x: float
    adapter_raw_y: float
    adapter_screen_x: int | None
    adapter_screen_y: int | None
    mapped_x: int
    mapped_y: int
    confidence: float
    blink_detected: bool
    invalid_reason: str | None


@dataclass
class GuidedStats:
    name: str
    samples: list[TraceSample] = field(default_factory=list)

    def values(self, attr: str) -> list[float]:
        values: list[float] = []
        for sample in self.samples:
            value = getattr(sample, attr)
            if value is not None:
                values.append(float(value))
        return values

    def mean(self, attr: str) -> float | None:
        values = self.values(attr)
        return mean(values) if values else None

    @property
    def valid_samples(self) -> list[TraceSample]:
        return [sample for sample in self.samples if sample.confidence > 0.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace EyeTrax prediction through the VisiMove mapping pipeline.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--detector-backend", choices=("dummy", "opencv", "mediapipe", "yolo"), default=None)
    parser.add_argument("--calibration-profile", default=None)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--delay-ms", type=int, default=50)
    parser.add_argument("--guided", action="store_true")
    parser.add_argument("--seconds-per-target", type=float, default=2.0)
    args = parser.parse_args()

    config = load_config(args.config, "config/performance.yaml")
    config.setdefault("gaze", {})["gaze_backend"] = "eyetrax"
    config.setdefault("gaze", {})["backend"] = "eyetrax"
    if args.camera_index is not None:
        config.setdefault("camera", {})["index"] = args.camera_index
    if args.detector_backend is not None:
        config.setdefault("detection", {})["detector_backend"] = args.detector_backend
        config.setdefault("detection", {})["backend"] = args.detector_backend
    calibration_config = config.setdefault("calibration", {})
    if args.calibration_profile:
        calibration_config["profile_path"] = str(resolve_project_path(args.calibration_profile))
    else:
        eyetrax_profile = PROJECT_ROOT / "data" / "calibration" / "user_profile_eyetrax.json"
        if eyetrax_profile.exists():
            calibration_config["profile_path"] = str(eyetrax_profile)

    adapter = build_eyetrax_adapter(config)
    mapper = build_calibration_mapper(calibration_config)
    detector = build_detector(config.get("detection", {}))
    camera = WebcamCamera(
        CameraConfig(**config.get("camera", {})),
        resize_width=config.get("performance", {}).get("resize_width"),
    )

    print_startup_notes(config, mapper)
    print_calibration_summary(calibration_config.get("profile_path"))

    camera.open()
    try:
        if args.guided:
            all_stats: list[GuidedStats] = []
            for target in GUIDED_TARGETS:
                input(f"\nLook at {target.upper()}, then press Enter to trace EyeTrax...")
                stats = collect_guided_target(
                    target,
                    args.seconds_per_target,
                    camera,
                    detector,
                    adapter,
                    mapper,
                )
                all_stats.append(stats)
                print_guided_stats(stats)
            print_guided_diagnosis(all_stats)
        else:
            for index in range(max(1, args.frames)):
                sample = collect_one(camera, detector, adapter, mapper)
                print_sample(index + 1, sample)
                if args.delay_ms > 0:
                    sleep(args.delay_ms / 1000)
    finally:
        camera.close()


def build_eyetrax_adapter(config: dict[str, Any]) -> EyeTraxAdapter:
    eyetrax_config = config.get("eyetrax", {})
    gaze_config = config.get("gaze", {})
    model_path = gaze_config.get("model_path") or eyetrax_config.get("model_path", "models/gaze/eyetrax")
    status = EyeTraxAdapter.check_setup(
        repo_path=eyetrax_config.get("repo_path", "external/eyetrax"),
        model_path=model_path,
        face_landmarker_model_path=eyetrax_config.get(
            "face_landmarker_model_path",
            "models/detection/face_landmarker.task",
        ),
    )
    if not status.ready:
        raise RuntimeError(status.reason)
    return EyeTraxAdapter(
        repo_path=eyetrax_config.get("repo_path", "external/eyetrax"),
        model_path=model_path,
        face_landmarker_model_path=eyetrax_config.get(
            "face_landmarker_model_path",
            "models/detection/face_landmarker.task",
        ),
        use_gpu=bool(eyetrax_config.get("use_gpu", False)),
        input_size=eyetrax_config.get("input_size"),
    )


def collect_guided_target(
    name: str,
    seconds: float,
    camera: WebcamCamera,
    detector: Any,
    adapter: EyeTraxAdapter,
    mapper: CalibrationMapper,
) -> GuidedStats:
    stats = GuidedStats(name=name)
    deadline = monotonic() + max(0.5, seconds)
    while monotonic() < deadline:
        stats.samples.append(collect_one(camera, detector, adapter, mapper))
    return stats


def collect_one(
    camera: WebcamCamera,
    detector: Any,
    adapter: EyeTraxAdapter,
    mapper: CalibrationMapper,
) -> TraceSample:
    webcam_frame = camera.read()
    if webcam_frame is None:
        return empty_sample("camera frame unavailable")
    frame = webcam_frame.frame
    detection = detector.detect(frame, monotonic())
    forced_detection = SimpleNamespace(found=True)
    trace = adapter.trace_frame(frame, forced_detection, include_landmark_count=True)
    mapped, _debug = mapper.map_with_debug(trace.gaze_result)
    return TraceSample(
        pipeline_face_detected=bool(getattr(detection, "found", False)),
        eyetrax_face_detected=trace.face_detected,
        landmark_count=trace.landmark_count,
        feature_count=trace.feature_count,
        feature_preview=trace.feature_preview,
        prediction_x=trace.prediction[0] if trace.prediction else None,
        prediction_y=trace.prediction[1] if trace.prediction else None,
        adapter_raw_x=trace.gaze_result.raw_x,
        adapter_raw_y=trace.gaze_result.raw_y,
        adapter_screen_x=trace.gaze_result.screen_x,
        adapter_screen_y=trace.gaze_result.screen_y,
        mapped_x=mapped.x,
        mapped_y=mapped.y,
        confidence=trace.gaze_result.confidence,
        blink_detected=trace.blink_detected,
        invalid_reason=trace.invalid_reason,
    )


def empty_sample(reason: str) -> TraceSample:
    return TraceSample(
        pipeline_face_detected=False,
        eyetrax_face_detected=False,
        landmark_count=None,
        feature_count=0,
        feature_preview=(),
        prediction_x=None,
        prediction_y=None,
        adapter_raw_x=0.5,
        adapter_raw_y=0.5,
        adapter_screen_x=None,
        adapter_screen_y=None,
        mapped_x=0,
        mapped_y=0,
        confidence=0.0,
        blink_detected=False,
        invalid_reason=reason,
    )


def print_sample(index: int, sample: TraceSample) -> None:
    print(
        f"frame={index} "
        f"pipeline_face={'yes' if sample.pipeline_face_detected else 'no'} "
        f"eyetrax_face={'yes' if sample.eyetrax_face_detected else 'no'} "
        f"landmarks={sample.landmark_count if sample.landmark_count is not None else 'unknown'} "
        f"features={sample.feature_count} "
        f"feature_preview={format_tuple(sample.feature_preview)} "
        f"model_prediction_px={format_optional_pair(sample.prediction_x, sample.prediction_y)} "
        f"adapter_raw=({sample.adapter_raw_x:.3f},{sample.adapter_raw_y:.3f}) "
        f"adapter_screen={format_optional_pair(sample.adapter_screen_x, sample.adapter_screen_y)} "
        f"mapped=({sample.mapped_x},{sample.mapped_y}) "
        f"confidence={sample.confidence:.2f} "
        f"blink={'yes' if sample.blink_detected else 'no'} "
        f"reason={sample.invalid_reason or 'none'}"
    )


def print_guided_stats(stats: GuidedStats) -> None:
    print(f"\n{stats.name.upper()} trace")
    print(f"  samples: {len(stats.samples)} valid: {len(stats.valid_samples)}")
    print(f"  raw EyeTrax prediction x: {format_range(stats.values('prediction_x'))}")
    print(f"  raw EyeTrax prediction y: {format_range(stats.values('prediction_y'))}")
    print(f"  adapter raw_x: {format_range(stats.values('adapter_raw_x'))}")
    print(f"  adapter raw_y: {format_range(stats.values('adapter_raw_y'))}")
    print(f"  mapped x: {format_range(stats.values('mapped_x'))}")
    print(f"  mapped y: {format_range(stats.values('mapped_y'))}")
    print(f"  confidence mean: {format_mean(stats.values('confidence'))}")
    print(f"  x saturation near 1.0: {ratio(stats.values('adapter_raw_x'), lambda value: value >= 0.98):.0%}")
    print(
        "  y saturation near 0/1: "
        f"{ratio(stats.values('adapter_raw_y'), lambda value: value <= 0.02 or value >= 0.98):.0%}"
    )
    reasons = sorted({sample.invalid_reason for sample in stats.samples if sample.invalid_reason})
    print("  invalid reasons: " + ("none" if not reasons else ", ".join(reasons)))


def print_guided_diagnosis(all_stats: list[GuidedStats]) -> None:
    by_name = {stats.name: stats for stats in all_stats}
    left = by_name.get("far left")
    center = by_name.get("center")
    right = by_name.get("far right")
    warnings: list[str] = []
    if left and center and right:
        left_prediction = left.mean("prediction_x")
        center_prediction = center.mean("prediction_x")
        right_prediction = right.mean("prediction_x")
        left_mapped = left.mean("mapped_x")
        center_mapped = center.mean("mapped_x")
        right_mapped = right.mean("mapped_x")
        if None not in (left_prediction, center_prediction, right_prediction):
            print(
                "\nHorizontal EyeTrax prediction means: "
                f"left={left_prediction:.1f}, center={center_prediction:.1f}, right={right_prediction:.1f}"
            )
            if not (left_prediction < center_prediction < right_prediction):
                warnings.append("EyeTrax x prediction is not ordered left < center < right")
            elif center_prediction - left_prediction < 250:
                warnings.append("EyeTrax left/center separation is weak")
            if right_prediction is not None and center_prediction is not None and left_prediction is not None:
                left_strength = max(1.0, center_prediction - left_prediction)
                right_strength = right_prediction - center_prediction
                if right_strength > left_strength * 1.5:
                    warnings.append("EyeTrax raw prediction has right-side bias")
        if None not in (left_mapped, center_mapped, right_mapped):
            print(
                "Horizontal mapped means: "
                f"left={left_mapped:.1f}, center={center_mapped:.1f}, right={right_mapped:.1f}"
            )
            if not (left_mapped < center_mapped < right_mapped):
                warnings.append("VisiMove mapped x is not ordered left < center < right")

    all_raw_x = [value for stats in all_stats for value in stats.values("adapter_raw_x")]
    if ratio(all_raw_x, lambda value: value >= 0.98) > 0.30:
        warnings.append("adapter raw_x frequently saturates near 1.0")

    print("\nRoot-cause hints")
    if warnings:
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("  No obvious x-ordering or saturation issue detected in this trace.")
    print(
        "  EyeTrax model predictions are screen pixels. The VisiMove adapter normalizes those "
        "pixels to raw_x/raw_y, then VisiMove screen calibration maps them again."
    )
    print(
        "  If raw EyeTrax prediction cannot separate left/center/right, the issue is in the "
        "EyeTrax model/calibration for this setup. If prediction separates well but mapped x "
        "does not, inspect VisiMove calibration/profile pairing."
    )


def print_startup_notes(config: dict[str, Any], mapper: CalibrationMapper) -> None:
    print("EyeTrax trace startup")
    print(f"  camera index: {config.get('camera', {}).get('index', 0)}")
    print(f"  detector backend: {config.get('detection', {}).get('detector_backend', 'auto')}")
    print(f"  calibration profile: {config.get('calibration', {}).get('profile_path', 'none')}")
    print(f"  mapper screen: {mapper.screen_width}x{mapper.screen_height}")
    print("  correction logic: disabled in this trace; no gain/offset is applied")


def print_calibration_summary(profile_path: object) -> None:
    if not profile_path:
        print("  calibration summary: no profile configured")
        return
    path = resolve_project_path(profile_path)
    if not path.exists():
        print(f"  calibration summary: profile missing at {path}")
        return
    profile = load_calibration_profile(path)
    print(f"  calibration quality: {profile.calibration_quality}")
    print(f"  mapping model: {profile.mapping_model_type}")
    print(
        "  calibration raw domain: "
        f"x[{profile.raw_x_min},{profile.raw_x_max}], y[{profile.raw_y_min},{profile.raw_y_max}]"
    )
    expected = [
        (point.screen_x, point.screen_y)
        for point in profile.calibration_points
    ]
    print("  calibration target order: " + " -> ".join(f"({x},{y})" for x, y in expected))


def format_range(values: list[float]) -> str:
    if not values:
        return "none"
    return f"min={min(values):.3f} max={max(values):.3f} mean={mean(values):.3f} range={max(values) - min(values):.3f}"


def format_mean(values: list[float]) -> str:
    return "none" if not values else f"{mean(values):.3f}"


def ratio(values: list[float], predicate: Any) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if predicate(value)) / len(values)


def format_tuple(values: tuple[float, ...]) -> str:
    if not values:
        return "()"
    return "(" + ",".join(f"{value:.3f}" for value in values[:4]) + ("..." if len(values) > 4 else "") + ")"


def format_optional_pair(x: object, y: object) -> str:
    if x is None or y is None:
        return "(none,none)"
    if isinstance(x, float) or isinstance(y, float):
        return f"({float(x):.1f},{float(y):.1f})"
    return f"({x},{y})"


def resolve_project_path(path_value: object) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
