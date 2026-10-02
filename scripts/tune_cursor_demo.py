from __future__ import annotations

import argparse
from pathlib import Path
import sys
from statistics import median
from time import monotonic
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visimove.calibration import AxisAdjustmentConfig, DemoTargetSample, apply_axis_adjustment
from visimove.calibration import create_demo_tuning_profile, load_calibration_profile, save_demo_tuning_profile
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

DEFAULT_TARGETS = ("center", "top_left", "top_right", "bottom_left", "bottom_right")
WINDOW_NAME = "VisiMove Accuracy Tuning"
START_KEYS = {10, 13, 32}
CANCEL_KEYS = {27, ord("q")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect guided samples and create a VisiMove demo tuning profile.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--gaze-backend", choices=("dummy", "eyetrax", "gazefollower", "mobilegaze"), default="mobilegaze")
    parser.add_argument("--detector-backend", choices=("dummy", "opencv", "mediapipe", "yolo"), default=None)
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--calibration-profile", default=None)
    parser.add_argument("--output", default="data/calibration/demo_tuning_mobilegaze.json")
    parser.add_argument("--seconds-per-target", type=float, default=2.0)
    parser.add_argument("--stabilization-seconds", type=float, default=1.0)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS), help="Comma-separated target names.")
    parser.add_argument("--no-preview", action="store_true", help="Collect without the OpenCV preview window.")
    parser.add_argument("--max-offset-px", type=float, default=180.0)
    args = parser.parse_args()

    config = load_config(args.config, "config/performance.yaml")
    apply_cli_overrides(config, args)
    calibration_profile_path = str(config.get("calibration", {}).get("profile_path", ""))
    apply_edge_source_from_profile(config)

    mapper = build_calibration_mapper(config.get("calibration", {}))
    smoothing_config = dict(config.get("smoothing", {}))
    smoothing_config["screen_width"] = mapper.screen_width
    smoothing_config["screen_height"] = mapper.screen_height
    smoother = build_smoothing_filter(smoothing_config)
    camera = WebcamCamera(
        CameraConfig(**config.get("camera", {})),
        resize_width=config.get("performance", {}).get("resize_width"),
    )
    detector = build_detector(config.get("detection", {}))
    gaze_model = build_gaze_model(build_gaze_backend_config(config))
    axis_config = build_axis_config(config.get("calibration", {}))
    targets = parse_targets(args.targets, mapper.screen_width, mapper.screen_height)

    print("VisiMove demo cursor tuning")
    print(f"  gaze_backend: {config.get('gaze', {}).get('gaze_backend')}")
    print(f"  calibration profile: {calibration_profile_path or 'none'}")
    print(f"  screen size: {mapper.screen_width}x{mapper.screen_height}")
    print("  cursor movement: disabled during tuning")

    samples: list[DemoTargetSample] = []
    cancelled = False
    camera.open()
    try:
        if not args.no_preview:
            initialize_preview_window()
        for name, target_x, target_y in targets:
            if not wait_for_target_start(
                name=name,
                target_x=target_x,
                target_y=target_y,
                screen_width=mapper.screen_width,
                screen_height=mapper.screen_height,
                show_preview=not args.no_preview,
            ):
                cancelled = True
                break
            smoother.reset()
            target_samples, target_cancelled = collect_target_samples(
                name=name,
                target_x=target_x,
                target_y=target_y,
                seconds=args.seconds_per_target,
                camera=camera,
                detector=detector,
                gaze_model=gaze_model,
                mapper=mapper,
                axis_config=axis_config,
                smoother=smoother,
                stabilization_seconds=args.stabilization_seconds,
                show_preview=not args.no_preview,
            )
            samples.extend(target_samples)
            print_target_summary(name, target_x, target_y, target_samples)
            if target_cancelled:
                cancelled = True
                break
    finally:
        camera.close()
        if not args.no_preview:
            cv2.destroyAllWindows()

    if cancelled:
        raise SystemExit("Tuning cancelled by user.")

    profile = create_demo_tuning_profile(        gaze_backend=str(config.get("gaze", {}).get("gaze_backend", args.gaze_backend)),
        calibration_profile_path=calibration_profile_path,
        screen_width=mapper.screen_width,
        screen_height=mapper.screen_height,
        samples=samples,
        current_calibration=dict(config.get("calibration", {})),
        current_smoothing=dict(config.get("smoothing", {})),
        current_cursor=dict(config.get("cursor", {})),
        max_offset_px=args.max_offset_px,
    )
    save_demo_tuning_profile(profile, resolve_project_path(args.output))
    print("\nSaved demo tuning profile")
    print(f"  path: {resolve_project_path(args.output)}")
    print(f"  quality: {profile.quality}")
    print("  calibration overrides:")
    for key, value in profile.calibration_overrides.items():
        print(f"    {key}: {value}")
    print("  smoothing overrides:")
    for key, value in profile.smoothing_overrides.items():
        print(f"    {key}: {value}")
    for note in profile.notes:
        print(f"  note: {note}")
    if profile.quality != "good":
        raise SystemExit("Tuning capture needs review. The profile was saved for diagnosis but will not be applied.")
    print("\nUse this command for the tuned demo:")
    print(
        ".\\.venv\\Scripts\\python.exe scripts\\run_tracking.py "
        f"--gaze-backend {args.gaze_backend} --enable-cursor --show-debug "
        f"--tuning-profile {args.output}"
    )


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.camera_index is not None:
        config.setdefault("camera", {})["index"] = args.camera_index
    if args.detector_backend is not None:
        config.setdefault("detection", {})["detector_backend"] = args.detector_backend
        config.setdefault("detection", {})["backend"] = args.detector_backend
    config.setdefault("gaze", {})["gaze_backend"] = args.gaze_backend
    config.setdefault("gaze", {})["backend"] = args.gaze_backend
    calibration = config.setdefault("calibration", {})
    if args.calibration_profile:
        calibration["profile_path"] = str(resolve_project_path(args.calibration_profile))
    else:
        backend_profile = PROJECT_ROOT / "data" / "calibration" / f"user_profile_{args.gaze_backend}.json"
        if backend_profile.exists():
            calibration["profile_path"] = str(backend_profile)
    if calibration.get("profile_path"):
        calibration["profile_path"] = str(resolve_project_path(calibration["profile_path"]))
    config.setdefault("cursor", {})["enabled"] = False
    config.setdefault("pipeline", {})["dry_run"] = True


def apply_edge_source_from_profile(config: dict[str, Any]) -> None:
    profile_path = config.get("calibration", {}).get("profile_path")
    if not profile_path:
        return
    try:
        profile = load_calibration_profile(profile_path)
    except (OSError, KeyError, ValueError):
        return
    if not profile.target_screen_coordinates:
        return
    xs = [float(target[0]) for target in profile.target_screen_coordinates]
    ys = [float(target[1]) for target in profile.target_screen_coordinates]
    calibration = config.setdefault("calibration", {})
    calibration["edge_source_min_x"] = min(xs)
    calibration["edge_source_max_x"] = max(xs)
    calibration["edge_source_min_y"] = min(ys)
    calibration["edge_source_max_y"] = max(ys)


def build_axis_config(config: dict[str, Any]) -> AxisAdjustmentConfig:
    return AxisAdjustmentConfig(
        horizontal_gain=float(config.get("horizontal_gain", 1.0)),
        vertical_gain=float(config.get("vertical_gain", 1.0)),
        horizontal_offset=float(config.get("horizontal_offset", 0.0)),
        vertical_offset=float(config.get("vertical_offset", 0.0)),
        enabled=bool(config.get("center_bias_correction", True)),
        edge_reach_enabled=bool(config.get("edge_reach_enabled", False)),
        edge_margin_px=int(config.get("edge_margin_px", 0)),
        edge_boost_enabled=bool(config.get("edge_boost_enabled", False)),
        edge_boost_gamma=float(config.get("edge_boost_gamma", 0.80)),
        source_min_x=optional_float(config.get("edge_source_min_x")),
        source_max_x=optional_float(config.get("edge_source_max_x")),
        source_min_y=optional_float(config.get("edge_source_min_y")),
        source_max_y=optional_float(config.get("edge_source_max_y")),
    )


def initialize_preview_window() -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except cv2.error:
        # Some OpenCV Windows builds do not support fullscreen window properties.
        pass


def wait_for_target_start(
    *,
    name: str,
    target_x: int,
    target_y: int,
    screen_width: int,
    screen_height: int,
    show_preview: bool,
) -> bool:
    label = name.replace("_", " ").upper()
    if not show_preview:
        input(f"\nLook at {label} ({target_x},{target_y}), then press Enter...")
        return True

    print(f"\nLook at {label} ({target_x},{target_y}); press Space or Enter in the tuning window.")
    while True:
        draw_preview(
            name=name,
            target_x=target_x,
            target_y=target_y,
            screen_width=screen_width,
            screen_height=screen_height,
            phase="ready",
            count=0,
            progress=0.0,
            face_ready=None,
        )
        key = cv2.waitKey(20) & 0xFF
        if key in START_KEYS:
            return True
        if key in CANCEL_KEYS:
            return False


def collect_target_samples(
    *,
    name: str,
    target_x: int,
    target_y: int,
    seconds: float,
    camera: WebcamCamera,
    detector: Any,
    gaze_model: Any,
    mapper: Any,
    axis_config: AxisAdjustmentConfig,
    smoother: Any,
    stabilization_seconds: float,
    show_preview: bool,
) -> tuple[list[DemoTargetSample], bool]:
    samples: list[DemoTargetSample] = []
    stabilization = max(0.0, stabilization_seconds)
    collection_seconds = max(0.5, seconds)
    started_at = monotonic()
    collect_after = started_at + stabilization
    deadline = collect_after + collection_seconds
    while monotonic() < deadline:
        now = monotonic()
        webcam_frame = camera.read()
        if webcam_frame is None:
            if show_preview:
                draw_preview(
                    name=name,
                    target_x=target_x,
                    target_y=target_y,
                    screen_width=mapper.screen_width,
                    screen_height=mapper.screen_height,
                    phase="camera unavailable",
                    count=len(samples),
                    progress=0.0,
                    face_ready=False,
                )
                if cv2.waitKey(20) & 0xFF in CANCEL_KEYS:
                    return samples, True
            continue

        frame = webcam_frame.frame
        detection = detector.detect(frame, now)
        gaze = gaze_model.estimate(frame, detection)
        mapped, _mapping_debug = mapper.map_with_debug(gaze)
        adjustment = apply_axis_adjustment(mapped, mapper.screen_width, mapper.screen_height, axis_config)
        filtered = smoother.update(adjustment.after.x, adjustment.after.y, now)
        observed_x = float(adjustment.after.x if filtered is None else filtered[0])
        observed_y = float(adjustment.after.y if filtered is None else filtered[1])
        usable = bool(detection.found and detection.eyes is not None)
        if now >= collect_after and usable:
            samples.append(
                DemoTargetSample(
                    name=name,
                    target_x=target_x,
                    target_y=target_y,
                    observed_x=observed_x,
                    observed_y=observed_y,
                )
            )
        if show_preview:
            if now < collect_after:
                phase = "stabilizing"
                progress = (now - started_at) / max(stabilization, 1e-6)
            else:
                phase = "collecting"
                progress = (now - collect_after) / collection_seconds
            draw_preview(
                name=name,
                target_x=target_x,
                target_y=target_y,
                screen_width=mapper.screen_width,
                screen_height=mapper.screen_height,
                phase=phase,
                count=len(samples),
                progress=progress,
                face_ready=usable,
            )
            if cv2.waitKey(1) & 0xFF in CANCEL_KEYS:
                return samples, True
    return samples, False


def draw_preview(
    *,
    name: str,
    target_x: int,
    target_y: int,
    screen_width: int,
    screen_height: int,
    phase: str,
    count: int,
    progress: float,
    face_ready: bool | None,
) -> None:
    width = max(320, int(screen_width))
    height = max(240, int(screen_height))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    point = (
        max(0, min(width - 1, int(target_x))),
        max(0, min(height - 1, int(target_y))),
    )
    color = (0, 208, 132) if phase == "collecting" else (0, 190, 255)
    cv2.circle(canvas, point, 20, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(canvas, point, 10, color, -1, cv2.LINE_AA)

    label = name.replace("_", " ").upper()
    if phase == "ready":
        instruction = f"Look at the {label} target - press SPACE or ENTER to begin"
        cv2.putText(canvas, instruction, (40, height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, "Esc or Q cancels", (40, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 190, 190), 1)
    else:
        # Keep the capture screen distraction-free so the user's gaze stays on the target.
        capture_color = color if face_ready else (0, 90, 255)
        cv2.circle(canvas, point, 28, capture_color, 2, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, canvas)

def print_target_summary(name: str, target_x: int, target_y: int, samples: list[DemoTargetSample]) -> None:
    print(f"{name}: samples={len(samples)}")
    if not samples:
        return
    observed_x = [sample.observed_x for sample in samples]
    observed_y = [sample.observed_y for sample in samples]
    median_x = median(observed_x)
    median_y = median(observed_y)
    print(f"  observed median=({median_x:.1f},{median_y:.1f})")
    print(f"  median error=({target_x - median_x:.1f},{target_y - median_y:.1f})")


def parse_targets(raw_targets: str, screen_width: int, screen_height: int) -> list[tuple[str, int, int]]:
    max_x = max(0, screen_width - 1)
    max_y = max(0, screen_height - 1)
    lookup = {
        "center": (screen_width // 2, screen_height // 2),
        "top_left": (0, 0),
        "top_right": (max_x, 0),
        "bottom_left": (0, max_y),
        "bottom_right": (max_x, max_y),
    }
    targets: list[tuple[str, int, int]] = []
    for item in raw_targets.split(","):
        name = item.strip().lower().replace("-", "_")
        if not name:
            continue
        if name not in lookup:
            raise SystemExit(f"Unknown target '{item}'. Choose from: {', '.join(lookup)}")
        x, y = lookup[name]
        targets.append((name, x, y))
    return targets or [("center", screen_width // 2, screen_height // 2)]


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_project_path(path_value: object) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()