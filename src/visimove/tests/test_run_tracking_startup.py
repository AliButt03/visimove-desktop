from __future__ import annotations

import argparse
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import ModuleType

from visimove.calibration.calibration_store import CalibrationSample


def load_run_tracking_module() -> ModuleType:
    project_root = Path(__file__).resolve().parents[3]
    module_path = project_root / "scripts" / "run_tracking.py"
    spec = importlib.util.spec_from_file_location("run_tracking", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_run_calibration_module() -> ModuleType:
    project_root = Path(__file__).resolve().parents[3]
    module_path = project_root / "scripts" / "run_calibration.py"
    spec = importlib.util.spec_from_file_location("run_calibration", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "camera_index": None,
        "detector_backend": None,
        "gaze_backend": None,
        "blink_backend": None,
        "calibration_profile": None,
        "tuning_profile": None,
        "enable_cursor": False,
        "allow_dummy_cursor": False,
        "allow_low_quality_calibration": False,
        "allow_unstable_live_gaze": False,
        "show_debug": False,
        "no_preview": False,
        "horizontal_gain": None,
        "vertical_gain": None,
        "horizontal_offset": None,
        "vertical_offset": None,
        "cursor_backend": None,
        "cursor_mode": None,
        "smoothing_filter": None,
        "ema_alpha": None,
        "adaptive_fast_alpha": None,
        "adaptive_slow_alpha": None,
        "adaptive_fixation_radius": None,
        "adaptive_release_radius": None,
        "adaptive_hold_ms": None,
        "max_speed_px_per_sec": None,
        "edge_reach": None,
        "edge_margin_px": None,
        "edge_boost": None,
        "edge_boost_gamma": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def base_config() -> dict[str, object]:
    return {
        "camera": {"index": 0, "width": 1280, "height": 720},
        "detection": {"detector_backend": "opencv", "backend": "opencv"},
        "gaze": {"gaze_backend": "dummy", "backend": "dummy"},
        "blink": {"backend": "dummy"},
        "calibration": {"profile_path": "data/calibration/user_profile.json"},
        "cursor": {"enabled": False},
        "pipeline": {"dry_run": True, "show_preview": True},
        "smoothing": {"filter": "ema"},
    }


def test_dummy_gaze_blocks_enabled_cursor_by_default() -> None:
    run_tracking = load_run_tracking_module()
    config = base_config()

    run_tracking.apply_cli_overrides(config, make_args(enable_cursor=True))

    assert config["cursor"]["enabled"] is False  # type: ignore[index]
    assert config["pipeline"]["dry_run"] is True  # type: ignore[index]


def test_cli_unstable_live_gaze_override_updates_pipeline_config() -> None:
    run_tracking = load_run_tracking_module()
    config = base_config()

    run_tracking.apply_cli_overrides(
        config,
        make_args(allow_unstable_live_gaze=True),
    )

    assert config["pipeline"]["allow_unstable_live_gaze"] is True  # type: ignore[index]


def test_dummy_gaze_cursor_requires_explicit_allow_flag() -> None:
    run_tracking = load_run_tracking_module()
    config = base_config()

    run_tracking.apply_cli_overrides(
        config,
        make_args(enable_cursor=True, allow_dummy_cursor=True),
    )

    assert config["cursor"]["enabled"] is True  # type: ignore[index]
    assert config["pipeline"]["dry_run"] is False  # type: ignore[index]


def test_cli_backend_overrides_update_config() -> None:
    run_tracking = load_run_tracking_module()
    config = base_config()

    run_tracking.apply_cli_overrides(
        config,
        make_args(
            detector_backend="mediapipe",
            gaze_backend="eyetrax",
            blink_backend="onnx",
            calibration_profile="data/calibration/custom.json",
            camera_index=2,
            show_debug=True,
            no_preview=True,
            cursor_backend="win32",
            cursor_mode="velocity",
        ),
    )

    assert config["camera"]["index"] == 2  # type: ignore[index]
    assert config["detection"]["detector_backend"] == "mediapipe"  # type: ignore[index]
    assert config["gaze"]["gaze_backend"] == "eyetrax"  # type: ignore[index]
    assert config["blink"]["backend"] == "onnx"  # type: ignore[index]
    assert str(config["calibration"]["profile_path"]).endswith("data\\calibration\\custom.json")  # type: ignore[index]
    assert config["pipeline"]["show_debug"] is True  # type: ignore[index]
    assert config["pipeline"]["show_preview"] is False  # type: ignore[index]
    assert config["cursor"]["backend"] == "win32"  # type: ignore[index]
    assert config["cursor"]["control_mode"] == "velocity"  # type: ignore[index]


def test_cli_axis_tuning_overrides_update_calibration_config() -> None:
    run_tracking = load_run_tracking_module()
    config = base_config()

    run_tracking.apply_cli_overrides(
        config,
        make_args(
            horizontal_gain=1.4,
            vertical_gain=1.1,
            horizontal_offset=-120.0,
            vertical_offset=30.0,
        ),
    )

    assert config["calibration"]["horizontal_gain"] == 1.4  # type: ignore[index]
    assert config["calibration"]["vertical_gain"] == 1.1  # type: ignore[index]
    assert config["calibration"]["horizontal_offset"] == -120.0  # type: ignore[index]
    assert config["calibration"]["vertical_offset"] == 30.0  # type: ignore[index]


def test_cli_smoothing_and_cursor_speed_overrides_update_config() -> None:
    run_tracking = load_run_tracking_module()
    config = base_config()

    run_tracking.apply_cli_overrides(
        config,
        make_args(
            smoothing_filter="ema",
            ema_alpha=0.18,
            max_speed_px_per_sec=700.0,
        ),
    )

    assert config["smoothing"]["filter"] == "ema"  # type: ignore[index]
    assert config["smoothing"]["ema_alpha"] == 0.18  # type: ignore[index]
    assert config["smoothing"]["alpha"] == 0.18  # type: ignore[index]
    assert config["cursor"]["max_speed_px_per_sec"] == 700.0  # type: ignore[index]


def test_cli_adaptive_smoothing_overrides_update_config() -> None:
    run_tracking = load_run_tracking_module()
    config = base_config()

    run_tracking.apply_cli_overrides(
        config,
        make_args(
            smoothing_filter="adaptive",
            adaptive_fast_alpha=0.4,
            adaptive_slow_alpha=0.06,
            adaptive_fixation_radius=35,
            adaptive_release_radius=100,
            adaptive_hold_ms=80,
        ),
    )

    assert config["smoothing"]["filter"] == "adaptive"  # type: ignore[index]
    assert config["smoothing"]["adaptive_fast_alpha"] == 0.4  # type: ignore[index]
    assert config["smoothing"]["adaptive_slow_alpha"] == 0.06  # type: ignore[index]
    assert config["smoothing"]["adaptive_fixation_radius"] == 35  # type: ignore[index]
    assert config["smoothing"]["adaptive_release_radius"] == 100  # type: ignore[index]
    assert config["smoothing"]["adaptive_fixation_hold_ms"] == 80  # type: ignore[index]


def test_mobilegaze_calibration_uses_auto_point_mean_defaults() -> None:
    run_calibration = load_run_calibration_module()
    config = {
        "mode": "9",
        "mapping_model": "affine",
        "mapping_fit_strategy": "point_means",
    }

    resolved = run_calibration.resolve_mapping_settings(config, "mobilegaze")

    assert resolved == ("auto", "point_means")


def test_mobilegaze_calibration_uses_backend_specific_auto_config() -> None:
    run_calibration = load_run_calibration_module()
    config = {
        "mode": "9",
        "mapping_model": "affine",
        "mapping_fit_strategy": "point_means",
        "mobilegaze_mapping_model": "auto",
        "mobilegaze_mapping_fit_strategy": "point_means",
    }

    resolved = run_calibration.resolve_mapping_settings(config, "mobilegaze")

    assert resolved == ("auto", "point_means")


def test_release_defaults_keep_cursor_out_of_pyautogui_fail_safe_corners() -> None:
    from visimove.config import load_config

    config = load_config("config/default.yaml")

    assert config["calibration"]["edge_margin_px"] == 8
    assert config["cursor"]["control_mode"] == "absolute"
    assert config["cursor"]["velocity_max_speed_px_per_sec"] == 1400
    assert config["cursor"]["velocity_horizontal_deadzone"] == 0.10
    assert config["cursor"]["velocity_vertical_deadzone"] == 0.06
    assert config["cursor"]["velocity_response_exponent"] == 1.35
    assert config["cursor"]["dwell_time_ms"] == 1200
    assert config["smoothing"]["filter"] == "one_euro"
    assert config["smoothing"]["one_euro_min_cutoff"] == 0.35
    assert config["smoothing"]["one_euro_beta"] == 0.0001
    assert config["smoothing"]['one_euro_max_speed_px_per_sec'] == 1600
    assert config["smoothing"]['one_euro_max_acceleration_px_per_sec2'] == 5000
    assert config["smoothing"]["adaptive_edge_fast_alpha"] == 0.45


def test_cli_edge_reach_overrides_update_calibration_config() -> None:
    run_tracking = load_run_tracking_module()
    config = base_config()

    run_tracking.apply_cli_overrides(
        config,
        make_args(edge_reach=True, edge_margin_px=12, edge_boost=True, edge_boost_gamma=0.72),
    )

    assert config["calibration"]["edge_reach_enabled"] is True  # type: ignore[index]
    assert config["calibration"]["edge_margin_px"] == 12  # type: ignore[index]
    assert config["calibration"]["edge_boost_enabled"] is True  # type: ignore[index]
    assert config["calibration"]["edge_boost_gamma"] == 0.72  # type: ignore[index]


def test_missing_calibration_profile_blocks_enabled_cursor() -> None:
    run_tracking = load_run_tracking_module()
    config = base_config()
    config["gaze"] = {"gaze_backend": "eyetrax", "backend": "eyetrax"}
    config["calibration"] = {"profile_path": "data/calibration/does_not_exist.json"}
    args = make_args(enable_cursor=True, gaze_backend="eyetrax")

    run_tracking.apply_cli_overrides(config, args)
    startup = run_tracking.build_startup_summary(config, args)
    run_tracking.apply_calibration_cursor_gate(config, startup, args)

    assert config["cursor"]["enabled"] is False  # type: ignore[index]
    assert config["pipeline"]["dry_run"] is True  # type: ignore[index]


def test_calibration_mismatch_reasons_include_backend_and_screen() -> None:
    run_tracking = load_run_tracking_module()
    profile = run_tracking.CalibrationProfile(
        timestamp="2026-05-14T00:00:00+00:00",
        screen_width=1280,
        screen_height=720,
        camera_index=0,
        calibration_points=[],
        raw_gaze_samples=[],
        target_screen_coordinates=[],
        mapping_model_type="linear",
        mapping_parameters={"model_type": "linear", "coefficients": [[0, 0], [1, 0], [0, 1]]},
        gaze_backend="dummy",
        detector_backend="opencv",
        blink_backend="dummy",
        calibration_point_layout="9-point",
        backend_metadata={},
    )

    reasons = run_tracking.calibration_mismatch_reasons(
        profile,
        current_gaze_backend="eyetrax",
        current_detector_backend="mediapipe",
        current_screen=run_tracking.ScreenBounds(width=1920, height=1080),
    )

    assert "profile was created with dummy gaze" in reasons
    assert "profile detector_backend opencv != current mediapipe" in reasons
    assert any("profile screen size 1280x720 != current 1920x1080" in reason for reason in reasons)


def test_mobilegaze_calibration_rejects_incompatible_preprocessing() -> None:
    run_tracking = load_run_tracking_module()
    profile = run_tracking.CalibrationProfile(
        timestamp="2026-05-18T00:00:00+00:00",
        screen_width=2560,
        screen_height=1440,
        camera_index=0,
        calibration_points=[],
        raw_gaze_samples=[],
        target_screen_coordinates=[],
        mapping_model_type="grid",
        mapping_parameters={"model_type": "grid"},
        gaze_backend="mobilegaze",
        detector_backend="auto",
        blink_backend="dummy",
        calibration_point_layout="9-point",
        backend_metadata={},
    )

    old_reasons = run_tracking.calibration_mismatch_reasons(
        profile,
        current_gaze_backend="mobilegaze",
        current_detector_backend="auto",
        current_screen=run_tracking.ScreenBounds(width=2560, height=1440),
        current_mobilegaze_median_window=5,
    )
    assert any("preprocessing unknown" in reason for reason in old_reasons)
    assert any("median window unknown" in reason for reason in old_reasons)

    compatible = replace(
        profile,
        backend_metadata={
            "mobilegaze_preprocessing_version": run_tracking.MobileGazeAdapter.preprocessing_version,
            "mobilegaze_temporal_median_window": "5",
        },
    )
    compatible_reasons = run_tracking.calibration_mismatch_reasons(
        compatible,
        current_gaze_backend="mobilegaze",
        current_detector_backend="auto",
        current_screen=run_tracking.ScreenBounds(width=2560, height=1440),
        current_mobilegaze_median_window=5,
    )
    assert compatible_reasons == []


def test_effective_calibration_quality_downgrades_bad_mapping() -> None:
    run_tracking = load_run_tracking_module()
    profile = run_tracking.CalibrationProfile(
        timestamp="2026-05-18T00:00:00+00:00",
        screen_width=1000,
        screen_height=1000,
        camera_index=0,
        calibration_points=[],
        raw_gaze_samples=[
            CalibrationSample(raw_gaze=(0.1, 0.1), target_screen=(100, 100), timestamp=1.0, confidence=1.0),
            CalibrationSample(raw_gaze=(0.9, 0.9), target_screen=(900, 900), timestamp=2.0, confidence=1.0),
        ],
        target_screen_coordinates=[(100, 100), (900, 900)],
        mapping_model_type="linear",
        mapping_parameters={"model_type": "linear", "coefficients": [[100, 100], [0, 0], [0, 0]]},
        gaze_backend="mobilegaze",
        detector_backend="auto",
        blink_backend="dummy",
        calibration_point_layout="2-point",
        backend_metadata={},
        calibration_quality="good",
        calibration_warnings=[],
        raw_x_min=0.1,
        raw_x_max=0.9,
        raw_y_min=0.1,
        raw_y_max=0.9,
        raw_x_range=0.8,
        raw_y_range=0.8,
    )

    quality, warnings = run_tracking.effective_calibration_quality(profile)

    assert quality == "needs_review"
    assert any("mapped X range is too small" in warning for warning in warnings)


def test_poor_calibration_quality_blocks_enabled_cursor() -> None:
    run_tracking = load_run_tracking_module()
    config = base_config()
    config["gaze"] = {"gaze_backend": "eyetrax", "backend": "eyetrax"}
    config["cursor"] = {"enabled": True}
    config["pipeline"] = {"dry_run": False}
    startup = {
        "calibration_mismatch_reasons": [],
        "calibration_quality": "poor",
    }

    run_tracking.apply_calibration_cursor_gate(
        config,
        startup,
        make_args(enable_cursor=True, gaze_backend="eyetrax"),
    )

    assert config["cursor"]["enabled"] is False  # type: ignore[index]
    assert config["pipeline"]["dry_run"] is True  # type: ignore[index]


def test_poor_calibration_quality_allows_preview_when_cursor_disabled() -> None:
    run_tracking = load_run_tracking_module()
    config = base_config()
    startup = {
        "calibration_mismatch_reasons": [],
        "calibration_quality": "poor",
    }

    run_tracking.apply_calibration_cursor_gate(
        config,
        startup,
        make_args(enable_cursor=False, gaze_backend="eyetrax"),
    )

    assert config["cursor"]["enabled"] is False  # type: ignore[index]
    assert config["pipeline"]["dry_run"] is True  # type: ignore[index]


def test_good_and_acceptable_calibration_allow_enabled_cursor() -> None:
    run_tracking = load_run_tracking_module()
    for quality in ("good", "acceptable"):
        config = base_config()
        config["cursor"] = {"enabled": True}
        config["pipeline"] = {"dry_run": False}
        config["calibration"] = {"allow_acceptable_calibration_for_cursor": True}
        startup = {
            "calibration_mismatch_reasons": [],
            "calibration_quality": quality,
        }

        run_tracking.apply_calibration_cursor_gate(
            config,
            startup,
            make_args(enable_cursor=True, gaze_backend="eyetrax"),
        )

        assert config["cursor"]["enabled"] is True  # type: ignore[index]
        assert config["pipeline"]["dry_run"] is False  # type: ignore[index]


def test_acceptable_calibration_can_be_blocked_by_config() -> None:
    run_tracking = load_run_tracking_module()
    config = base_config()
    config["cursor"] = {"enabled": True}
    config["pipeline"] = {"dry_run": False}
    config["calibration"] = {"allow_acceptable_calibration_for_cursor": False}
    startup = {
        "calibration_mismatch_reasons": [],
        "calibration_quality": "acceptable",
    }

    run_tracking.apply_calibration_cursor_gate(
        config,
        startup,
        make_args(enable_cursor=True, gaze_backend="eyetrax"),
    )

    assert config["cursor"]["enabled"] is False  # type: ignore[index]
    assert config["pipeline"]["dry_run"] is True  # type: ignore[index]


def test_tuning_profile_applies_before_explicit_cli_overrides(tmp_path) -> None:
    run_tracking = load_run_tracking_module()
    profile_path = tmp_path / "demo_tuning.json"
    profile_path.write_text(
        """
{
  "version": 1,
  "gaze_backend": "mobilegaze",
  "calibration_profile_path": "data/calibration/user_profile_mobilegaze.json",
  "screen_width": 2560,
  "screen_height": 1440,
  "samples": [
    {"name": "center", "target_x": 1280, "target_y": 720, "observed_x": 1205.0, "observed_y": 760.0},
    {"name": "center", "target_x": 1280, "target_y": 720, "observed_x": 1204.0, "observed_y": 759.0},
    {"name": "center", "target_x": 1280, "target_y": 720, "observed_x": 1206.0, "observed_y": 761.0},
    {"name": "center", "target_x": 1280, "target_y": 720, "observed_x": 1205.0, "observed_y": 760.0},
    {"name": "center", "target_x": 1280, "target_y": 720, "observed_x": 1204.0, "observed_y": 759.0},
    {"name": "center", "target_x": 1280, "target_y": 720, "observed_x": 1206.0, "observed_y": 761.0},
    {"name": "center", "target_x": 1280, "target_y": 720, "observed_x": 1205.0, "observed_y": 760.0},
    {"name": "center", "target_x": 1280, "target_y": 720, "observed_x": 1204.0, "observed_y": 759.0}
  ],
  "calibration_overrides": {
    "horizontal_offset": 75.0,
    "vertical_offset": -40.0,
    "edge_reach_enabled": true,
    "edge_boost_enabled": false
  },
  "smoothing_overrides": {
    "filter": "adaptive",
    "adaptive_slow_alpha": 0.02
  },
  "cursor_overrides": {
    "movement_duration_sec": 0.0,
    "max_speed_px_per_sec": 3000
  },
  "notes": []
}
""".strip(),
        encoding="utf-8",
    )
    config = base_config()

    run_tracking.apply_cli_overrides(
        config,
        make_args(tuning_profile=str(profile_path), vertical_offset=0.0, adaptive_slow_alpha=0.03),
    )

    assert config["calibration"]["horizontal_offset"] == 75.0  # type: ignore[index]
    assert config["calibration"]["vertical_offset"] == 0.0  # type: ignore[index]
    assert config["calibration"]["edge_reach_enabled"] is True  # type: ignore[index]
    assert config["calibration"]["edge_boost_enabled"] is False  # type: ignore[index]
    assert config["smoothing"]["filter"] == "adaptive"  # type: ignore[index]
    assert config["smoothing"]["adaptive_slow_alpha"] == 0.03  # type: ignore[index]
    assert config["cursor"]["movement_duration_sec"] == 0.0  # type: ignore[index]
    assert config["demo_tuning"]["source_path"] == str(profile_path)  # type: ignore[index]