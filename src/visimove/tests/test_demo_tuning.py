from __future__ import annotations

import pytest

from visimove.calibration.demo_tuning import (
    DemoTargetSample,
    apply_demo_tuning_profile,
    create_demo_tuning_profile,
    load_demo_tuning_profile,
    save_demo_tuning_profile,
)


def test_demo_tuning_profile_rejects_center_bias_at_offset_safety_limit() -> None:
    samples = [
        *[
            DemoTargetSample("center", 1280, 720, observed_x, observed_y)
            for observed_x, observed_y in ((1120.0, 790.0), (1110.0, 800.0), (1100.0, 810.0))
            for _ in range(4)
        ],
        DemoTargetSample("top_right", 2559, 0, 2500.0, 20.0),
    ]

    profile = create_demo_tuning_profile(
        gaze_backend="mobilegaze",
        calibration_profile_path="data/calibration/user_profile_mobilegaze.json",
        screen_width=2560,
        screen_height=1440,
        samples=samples,
        current_calibration={"horizontal_offset": 10.0, "vertical_offset": 0.0, "edge_reach_enabled": True},
        current_smoothing={"adaptive_fixation_radius": 90, "adaptive_slow_alpha": 0.025},
        current_cursor={"max_speed_px_per_sec": 3000},
        max_offset_px=150,
    )

    assert profile.quality == "needs_review"
    assert profile.calibration_overrides["horizontal_offset"] == 10.0
    assert profile.calibration_overrides["vertical_offset"] == 0.0
    assert profile.calibration_overrides["edge_boost_enabled"] is False
    assert profile.smoothing_overrides["filter"] == "one_euro"
    assert profile.cursor_overrides["max_speed_px_per_sec"] == 3000.0


def test_demo_tuning_profile_round_trips_and_applies_to_config(tmp_path) -> None:
    stable_center_samples = [
        DemoTargetSample("center", 1280, 720, 1290.0 + index % 3, 710.0 + index % 2)
        for index in range(12)
    ]
    profile = create_demo_tuning_profile(
        gaze_backend="mobilegaze",
        calibration_profile_path="data/calibration/user_profile_mobilegaze.json",
        screen_width=2560,
        screen_height=1440,
        samples=stable_center_samples,
        current_calibration={"edge_reach_enabled": True},
        current_smoothing={"adaptive_fixation_radius": 90},
        current_cursor={"max_speed_px_per_sec": 2500},
    )
    path = tmp_path / "demo_tuning.json"

    save_demo_tuning_profile(profile, path)
    loaded = load_demo_tuning_profile(path)
    config: dict[str, object] = {"calibration": {}, "smoothing": {}, "cursor": {}}
    apply_demo_tuning_profile(config, loaded)

    assert loaded.samples[0].name == "center"
    assert config["calibration"]["horizontal_offset"] == -11.0  # type: ignore[index]
    assert config["calibration"]["vertical_offset"] == 9.5  # type: ignore[index]
    assert config["smoothing"]["filter"] == "one_euro"  # type: ignore[index]
    assert config["cursor"]["max_speed_px_per_sec"] == 2500.0  # type: ignore[index]
    assert config["demo_tuning"]["loaded"] is True  # type: ignore[index]

def test_demo_tuning_profile_rejects_saturated_unstable_center_capture() -> None:
    samples = [
        DemoTargetSample("center", 1280, 720, float(200 + index * 90), 0.0)
        for index in range(12)
    ]
    profile = create_demo_tuning_profile(
        gaze_backend="mobilegaze",
        calibration_profile_path="data/calibration/user_profile_mobilegaze.json",
        screen_width=2560,
        screen_height=1440,
        samples=samples,
        current_calibration={"edge_reach_enabled": True},
    )

    assert profile.quality == "needs_review"
    assert profile.calibration_overrides["horizontal_offset"] == 0.0
    assert profile.calibration_overrides["vertical_offset"] == 0.0
    with pytest.raises(ValueError, match="unsafe demo tuning profile"):
        apply_demo_tuning_profile({"calibration": {}, "smoothing": {}, "cursor": {}}, profile)

def test_demo_tuning_profile_fits_measured_axis_gain_and_offset() -> None:
    width = 2560
    height = 1440
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2
    expected_x_gain = 1.10
    expected_x_offset = 50.0
    expected_y_gain = 0.90
    expected_y_offset = -30.0
    targets = {
        "center": (1280, 720),
        "top_left": (0, 0),
        "top_right": (2559, 0),
        "bottom_left": (0, 1439),
        "bottom_right": (2559, 1439),
    }
    samples: list[DemoTargetSample] = []
    for name, (target_x, target_y) in targets.items():
        observed_x = center_x + (target_x - center_x - expected_x_offset) / expected_x_gain
        observed_y = center_y + (target_y - center_y - expected_y_offset) / expected_y_gain
        samples.extend(
            DemoTargetSample(name, target_x, target_y, observed_x, observed_y)
            for _ in range(12)
        )

    profile = create_demo_tuning_profile(
        gaze_backend="mobilegaze",
        calibration_profile_path="data/calibration/user_profile_mobilegaze.json",
        screen_width=width,
        screen_height=height,
        samples=samples,
        current_calibration={
            "horizontal_gain": 1.0,
            "vertical_gain": 1.0,
            "horizontal_offset": 0.0,
            "vertical_offset": 0.0,
            "edge_reach_enabled": True,
        },
        current_smoothing={"filter": "one_euro", "one_euro_beta": 0.0001},
    )

    assert profile.quality == "good"
    assert profile.calibration_overrides["horizontal_gain"] == pytest.approx(expected_x_gain, abs=0.01)
    assert profile.calibration_overrides["vertical_gain"] == pytest.approx(expected_y_gain, abs=0.01)
    assert profile.calibration_overrides["horizontal_offset"] == pytest.approx(expected_x_offset, abs=1.0)
    assert profile.calibration_overrides["vertical_offset"] == pytest.approx(expected_y_offset, abs=1.0)
    assert profile.smoothing_overrides["filter"] == "one_euro"


def test_demo_tuning_profile_rejects_crossed_five_point_capture() -> None:
    target_observations = {
        "center": ((1280, 720), (1083.0, 1431.0)),
        "top_left": ((0, 0), (28.0, 1411.0)),
        "top_right": ((2559, 0), (2160.0, 1142.0)),
        "bottom_left": ((0, 1439), (994.0, 1409.0)),
        "bottom_right": ((2559, 1439), (871.0, 1431.0)),
    }
    samples: list[DemoTargetSample] = []
    for name, (target, observed) in target_observations.items():
        samples.extend(
            DemoTargetSample(name, target[0], target[1], observed[0], observed[1])
            for _ in range(12)
        )

    profile = create_demo_tuning_profile(
        gaze_backend="mobilegaze",
        calibration_profile_path="data/calibration/user_profile_mobilegaze.json",
        screen_width=2560,
        screen_height=1440,
        samples=samples,
        current_calibration={
            "horizontal_gain": 1.0,
            "vertical_gain": 1.0,
            "horizontal_offset": 0.0,
            "vertical_offset": 0.0,
            "edge_reach_enabled": True,
        },
    )

    assert profile.quality == "needs_review"
    assert profile.calibration_overrides["horizontal_offset"] == 0.0
    assert profile.calibration_overrides["vertical_offset"] == 0.0
    assert any("center target median" in warning for warning in profile.quality_warnings)
    assert any("separation is not usable" in warning for warning in profile.quality_warnings)
    with pytest.raises(ValueError, match="unsafe demo tuning profile"):
        apply_demo_tuning_profile({"calibration": {}, "smoothing": {}, "cursor": {}}, profile)