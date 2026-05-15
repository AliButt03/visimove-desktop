from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType


def load_run_tracking_module() -> ModuleType:
    project_root = Path(__file__).resolve().parents[3]
    module_path = project_root / "scripts" / "run_tracking.py"
    spec = importlib.util.spec_from_file_location("run_tracking", module_path)
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
        ),
    )

    assert config["camera"]["index"] == 2  # type: ignore[index]
    assert config["detection"]["detector_backend"] == "mediapipe"  # type: ignore[index]
    assert config["gaze"]["gaze_backend"] == "eyetrax"  # type: ignore[index]
    assert config["blink"]["backend"] == "onnx"  # type: ignore[index]
    assert str(config["calibration"]["profile_path"]).endswith("data\\calibration\\custom.json")  # type: ignore[index]
    assert config["pipeline"]["show_debug"] is True  # type: ignore[index]
    assert config["pipeline"]["show_preview"] is False  # type: ignore[index]


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
