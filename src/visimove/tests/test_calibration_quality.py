from visimove.calibration import CalibrationSample, analyze_calibration_quality
from visimove.calibration.calibration_ui import CalibrationUiConfig


def make_sample(
    raw_x: float,
    raw_y: float,
    target: tuple[int, int] = (100, 100),
    confidence: float = 1.0,
) -> CalibrationSample:
    return CalibrationSample(
        raw_gaze=(raw_x, raw_y),
        target_screen=target,
        timestamp=1.0,
        confidence=confidence,
    )


def test_quality_detects_pinned_eyetrax_raw_gaze() -> None:
    samples = [
        make_sample(1.0, 0.0, (100, 100)),
        make_sample(1.0, 0.0, (200, 200)),
        make_sample(0.99, 0.01, (300, 300)),
        make_sample(1.0, 0.0, (400, 400)),
    ]

    report = analyze_calibration_quality(
        samples,
        expected_point_count=4,
        mapping_model_training_success=True,
        min_samples_per_point=1,
    )

    assert report.quality == "poor"
    assert any("raw_x samples are pinned near 1.0" in warning for warning in report.warnings)
    assert any("raw_y samples are pinned near 0.0" in warning for warning in report.warnings)


def test_quality_detects_low_sample_count() -> None:
    samples = [make_sample(0.2, 0.3, (100, 100))]

    report = analyze_calibration_quality(
        samples,
        expected_point_count=5,
        mapping_model_training_success=True,
        min_samples_per_point=3,
    )

    assert report.quality in {"poor", "needs_review"}
    assert any("fewer than 3 valid samples" in warning for warning in report.warnings)
    assert any("1/5 calibration points" in warning for warning in report.warnings)


def test_eyetrax_profile_quality_metadata_can_be_saved() -> None:
    samples = [make_sample(0.2, 0.3, (100, 100)), make_sample(0.6, 0.7, (500, 500))]
    report = analyze_calibration_quality(
        samples,
        expected_point_count=2,
        mapping_model_training_success=True,
        min_samples_per_point=1,
    )

    assert report.metrics["total_samples"] == 2
    assert report.confidence_statistics["mean"] == 1.0
    assert report.sample_count_per_point == [1, 1]


def test_quality_point_statistics_keep_collection_order() -> None:
    samples = [
        make_sample(0.1, 0.1, (307, 173)),
        make_sample(0.2, 0.2, (1280, 173)),
        make_sample(0.3, 0.3, (2252, 173)),
        make_sample(0.4, 0.4, (307, 720)),
    ]

    report = analyze_calibration_quality(
        samples,
        expected_point_count=4,
        mapping_model_training_success=True,
        min_samples_per_point=1,
    )

    assert [point["target_screen"] for point in report.metrics["point_statistics"]] == [
        [307, 173],
        [1280, 173],
        [2252, 173],
        [307, 720],
    ]


def test_mapping_point_means_keep_collection_order() -> None:
    from visimove.calibration.mapping_diagnostics import calibration_point_means

    samples = [
        make_sample(0.1, 0.1, (307, 173)),
        make_sample(0.2, 0.2, (1280, 173)),
        make_sample(0.3, 0.3, (2252, 173)),
        make_sample(0.4, 0.4, (307, 720)),
    ]

    _raw, targets = calibration_point_means(samples)

    assert targets == [(307, 173), (1280, 173), (2252, 173), (307, 720)]


def test_quality_detects_mapped_y_stuck_at_zero() -> None:
    samples = [
        make_sample(0.1, 0.1, (100, 100)),
        make_sample(0.5, 0.5, (500, 500)),
        make_sample(0.9, 0.9, (900, 900)),
    ]

    report = analyze_calibration_quality(
        samples,
        expected_point_count=3,
        mapping_model_training_success=True,
        min_samples_per_point=1,
        mapping_diagnostics={
            "mapped_y_stuck_at_zero": True,
            "mapped_y_range": 0.0,
            "mapped_x_range": 800.0,
            "mapping_warnings": ["mapped Y is stuck at 0 for many calibration points"],
        },
    )

    assert report.quality == "poor"
    assert "mapped Y is stuck at 0 after calibration" in report.warnings


def test_quality_flags_high_mapping_error_for_review() -> None:
    samples = [
        make_sample(0.1, 0.1, (100, 100)),
        make_sample(0.5, 0.5, (500, 500)),
        make_sample(0.9, 0.9, (900, 900)),
    ]

    report = analyze_calibration_quality(
        samples,
        expected_point_count=3,
        mapping_model_training_success=True,
        min_samples_per_point=1,
        mapping_diagnostics={
            "mapped_x_range": 800.0,
            "mapped_y_range": 800.0,
            "mapping_mean_absolute_error_x": 190.0,
            "mapping_mean_absolute_error_y": 175.0,
            "mapping_target_x_range": 800.0,
            "mapping_target_y_range": 800.0,
        },
    )

    assert report.quality == "needs_review"
    assert any("mean absolute X error is high" in warning for warning in report.warnings)
    assert any("mean absolute Y error is high" in warning for warning in report.warnings)


def test_quality_flags_raw_sample_mapping_instability_for_review() -> None:
    samples = [
        make_sample(0.1, 0.1, (100, 100)),
        make_sample(0.5, 0.5, (500, 500)),
        make_sample(0.9, 0.9, (900, 900)),
    ]

    report = analyze_calibration_quality(
        samples,
        expected_point_count=3,
        mapping_model_training_success=True,
        min_samples_per_point=1,
        mapping_diagnostics={
            "mapped_x_range": 800.0,
            "mapped_y_range": 800.0,
            "mapping_target_x_range": 800.0,
            "mapping_target_y_range": 800.0,
            "mapping_sample_mean_absolute_error_x": 180.0,
            "mapping_sample_mean_absolute_error_y": 190.0,
            "sample_clipped_prediction_ratio": 0.35,
        },
    )

    assert report.quality == "needs_review"
    assert any("raw calibration samples map outside" in warning for warning in report.warnings)
    assert any("raw-sample mean absolute X error is high" in warning for warning in report.warnings)


def test_one_weak_point_with_healthy_mapping_is_acceptable() -> None:
    samples = []
    targets = [(100, 100), (500, 500), (900, 900)]
    for target in targets:
        count = 10 if target == (100, 100) else 20
        samples.extend(make_sample(0.1 + index * 0.01, 0.2 + index * 0.01, target) for index in range(count))

    report = analyze_calibration_quality(
        samples,
        expected_point_count=3,
        mapping_model_training_success=True,
        min_samples_per_point=15,
        mapping_diagnostics={
            "mapped_x_range": 800.0,
            "mapped_y_range": 800.0,
            "clipped_prediction_ratio": 0.0,
            "negative_y_before_clamp_ratio": 0.0,
        },
    )

    assert report.quality == "acceptable"


def test_multiple_weak_points_need_review() -> None:
    samples = []
    for target in [(100, 100), (500, 500), (900, 900)]:
        count = 10 if target != (900, 900) else 20
        samples.extend(make_sample(0.1 + index * 0.02, 0.2 + index * 0.02, target) for index in range(count))

    report = analyze_calibration_quality(
        samples,
        expected_point_count=3,
        mapping_model_training_success=True,
        min_samples_per_point=15,
        mapping_diagnostics={"mapped_x_range": 800.0, "mapped_y_range": 800.0},
    )

    assert report.quality == "needs_review"


def test_mapping_failure_is_poor() -> None:
    report = analyze_calibration_quality(
        [make_sample(0.1, 0.2), make_sample(0.8, 0.9)],
        expected_point_count=1,
        mapping_model_training_success=False,
        min_samples_per_point=1,
    )

    assert report.quality == "poor"


def test_calibration_ui_all_samples_training_data_uses_every_sample() -> None:
    from visimove.calibration.calibration_ui import CalibrationUi

    ui = object.__new__(CalibrationUi)
    ui.config = CalibrationUiConfig(mapping_fit_strategy="all_samples")
    ui.samples = [
        make_sample(0.1, 0.2, (100, 200)),
        make_sample(0.2, 0.3, (100, 200)),
        make_sample(0.8, 0.7, (900, 700)),
    ]

    raw, targets = ui._mapping_training_data()

    assert raw == [(0.1, 0.2), (0.2, 0.3), (0.8, 0.7)]
    assert targets == [(100.0, 200.0), (100.0, 200.0), (900.0, 700.0)]


def test_calibration_ui_auto_mapping_prefers_stable_candidate() -> None:
    from visimove.calibration.calibration_ui import CalibrationUi

    ui = object.__new__(CalibrationUi)
    ui.config = CalibrationUiConfig(mapping_model_type="auto", mapping_fit_strategy="point_means")
    ui.screen_width = 1000
    ui.screen_height = 1000
    ui.samples = [
        make_sample(0.0, 0.0, (0, 0)),
        make_sample(0.5, 0.0, (250, 0)),
        make_sample(1.0, 0.0, (1000, 0)),
        make_sample(0.0, 1.0, (0, 1000)),
        make_sample(0.5, 1.0, (250, 1000)),
        make_sample(1.0, 1.0, (1000, 1000)),
    ]

    raw, targets = ui._mapping_training_data()
    fitted = ui._fit_mapping(raw, targets)

    assert fitted.model_type == "grid"
