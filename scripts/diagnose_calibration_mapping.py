from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visimove.calibration import diagnose_profile_mapping, load_calibration_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose a VisiMove calibration mapping profile.")
    parser.add_argument("--profile", default="data/calibration/user_profile_eyetrax.json")
    args = parser.parse_args()

    profile_path = resolve_project_path(args.profile)
    profile = load_calibration_profile(profile_path)
    diagnostics = diagnose_profile_mapping(profile)

    raw_x = [sample.raw_gaze[0] for sample in profile.raw_gaze_samples]
    raw_y = [sample.raw_gaze[1] for sample in profile.raw_gaze_samples]
    targets_x = [sample.target_screen[0] for sample in profile.raw_gaze_samples]
    targets_y = [sample.target_screen[1] for sample in profile.raw_gaze_samples]

    print("Calibration mapping diagnostics")
    print(f"  profile path: {profile_path}")
    print(f"  gaze_backend: {profile.gaze_backend}")
    print(f"  screen size: {profile.screen_width}x{profile.screen_height}")
    print(f"  mapping model type: {profile.mapping_model_type}")
    print(f"  raw_x min/max/range: {fmt_range(raw_x)}")
    print(f"  raw_y min/max/range: {fmt_range(raw_y)}")
    print(f"  calibrated raw domain: {format_profile_domain(profile)}")
    print(f"  target_x min/max/range: {fmt_range(targets_x)}")
    print(f"  target_y min/max/range: {fmt_range(targets_y)}")
    print()
    print("Per-point predictions from calibration sample means")
    for prediction in diagnostics.point_predictions:
        print(
            "  "
            f"point={prediction.index} "
            f"target=({prediction.target[0]},{prediction.target[1]}) "
            f"raw_mean=({prediction.raw_mean[0]:.3f},{prediction.raw_mean[1]:.3f}) "
            f"before_clamp=({prediction.predicted_before_clamp[0]:.1f},"
            f"{prediction.predicted_before_clamp[1]:.1f}) "
            f"after_clamp=({prediction.predicted_after_clamp[0]},"
            f"{prediction.predicted_after_clamp[1]}) "
            f"error=({prediction.error[0]:.1f},{prediction.error[1]:.1f}) "
            f"clipped=({'yes' if prediction.clipped_x or prediction.clipped_y else 'no'})"
        )

    print()
    print(f"  mean absolute error x: {diagnostics.mean_absolute_error_x:.1f}px")
    print(f"  mean absolute error y: {diagnostics.mean_absolute_error_y:.1f}px")
    print(f"  RMSE x: {diagnostics.rmse_x:.1f}px")
    print(f"  RMSE y: {diagnostics.rmse_y:.1f}px")
    print(f"  mapped X range: {diagnostics.predicted_x_range:.1f}px")
    print(f"  mapped Y range: {diagnostics.predicted_y_range:.1f}px")
    print(f"  clipped prediction ratio: {diagnostics.clipped_prediction_ratio:.0%}")
    print(f"  negative Y before clamp ratio: {diagnostics.negative_y_before_clamp_ratio:.0%}")
    print(f"  mapped Y stuck at 0: {'yes' if diagnostics.mapped_y_stuck_at_zero else 'no'}")
    grouped = defaultdict(list)
    for sample in profile.raw_gaze_samples:
        grouped[sample.target_screen].append(sample)
    print("  samples per calibration point: " + ", ".join(str(len(grouped[target])) for target in sorted(grouped)))
    if diagnostics.warnings:
        print("Warnings")
        for warning in diagnostics.warnings:
            print(f"  - {warning}")
    else:
        print("Warnings: none")
    print("Runtime note")
    print(
        "  If live raw_y falls below the calibrated raw_y minimum, affine mapping extrapolates above "
        "the screen and screen clamping can force mapped Y to 0."
    )
    print(
        "  VisiMove can clamp mapping input to the calibrated raw domain plus margin, but cursor "
        "control should remain blocked when live gaze is frequently outside that domain."
    )


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def fmt_range(values: list[float | int]) -> str:
    if not values:
        return "none/none/0.000"
    minimum = min(values)
    maximum = max(values)
    return f"{minimum:.3f}/{maximum:.3f}/{maximum - minimum:.3f}"


def format_profile_domain(profile: object) -> str:
    raw_x_min = getattr(profile, "raw_x_min", None)
    raw_x_max = getattr(profile, "raw_x_max", None)
    raw_y_min = getattr(profile, "raw_y_min", None)
    raw_y_max = getattr(profile, "raw_y_max", None)
    if None in (raw_x_min, raw_x_max, raw_y_min, raw_y_max):
        return "unknown"
    return f"x[{raw_x_min:.3f},{raw_x_max:.3f}], y[{raw_y_min:.3f},{raw_y_max:.3f}]"


if __name__ == "__main__":
    main()
