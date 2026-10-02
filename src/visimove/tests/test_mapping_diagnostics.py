from visimove.calibration import CalibrationSample
from visimove.calibration.mapping_diagnostics import (
    PointPredictionDiagnostic,
    calibration_point_means,
    summarize_predictions,
)


def test_diagnostics_detects_clamping_to_top_edge() -> None:
    diagnostics = summarize_predictions(
        [
            PointPredictionDiagnostic(
                index=1,
                target=(100, 100),
                raw_mean=(0.2, 0.2),
                predicted_before_clamp=(100.0, -50.0),
                predicted_after_clamp=(100, 0),
                error=(0.0, -100.0),
                clipped_x=False,
                clipped_y=True,
            ),
            PointPredictionDiagnostic(
                index=2,
                target=(500, 500),
                raw_mean=(0.5, 0.5),
                predicted_before_clamp=(500.0, -20.0),
                predicted_after_clamp=(500, 0),
                error=(0.0, -500.0),
                clipped_x=False,
                clipped_y=True,
            ),
            PointPredictionDiagnostic(
                index=3,
                target=(900, 900),
                raw_mean=(0.9, 0.9),
                predicted_before_clamp=(900.0, 10.0),
                predicted_after_clamp=(900, 10),
                error=(0.0, -890.0),
                clipped_x=False,
                clipped_y=False,
            ),
        ],
        screen_width=1000,
        screen_height=1000,
    )

    assert diagnostics.negative_y_before_clamp_ratio > 0.5
    assert diagnostics.clipped_prediction_ratio > 0.5
    assert diagnostics.mapped_y_stuck_at_zero
    assert any("negative" in warning for warning in diagnostics.warnings)


def test_diagnostics_warns_when_raw_samples_are_unstable() -> None:
    diagnostics = summarize_predictions(
        [
            PointPredictionDiagnostic(
                index=1,
                target=(100, 100),
                raw_mean=(0.2, 0.2),
                predicted_before_clamp=(100.0, 100.0),
                predicted_after_clamp=(100, 100),
                error=(0.0, 0.0),
                clipped_x=False,
                clipped_y=False,
            ),
            PointPredictionDiagnostic(
                index=2,
                target=(900, 900),
                raw_mean=(0.8, 0.8),
                predicted_before_clamp=(900.0, 900.0),
                predicted_after_clamp=(900, 900),
                error=(0.0, 0.0),
                clipped_x=False,
                clipped_y=False,
            ),
        ],
        screen_width=1000,
        screen_height=1000,
        sample_metrics={
            "sample_mean_absolute_error_x": 350.0,
            "sample_mean_absolute_error_y": 260.0,
            "sample_clipped_prediction_ratio": 0.4,
            "sample_negative_y_before_clamp_ratio": 0.0,
        },
    )

    assert diagnostics.sample_clipped_prediction_ratio == 0.4
    assert any("raw calibration samples map outside" in warning for warning in diagnostics.warnings)
    assert any("raw-sample mean absolute X error" in warning for warning in diagnostics.warnings)


def test_diagnostics_warns_when_one_calibration_point_is_far_off_target() -> None:
    predictions = []
    for index in range(9):
        target_x = 100 + index * 100
        error_x = 200.0 if index == 8 else 0.0
        predictions.append(
            PointPredictionDiagnostic(
                index=index + 1,
                target=(target_x, 500),
                raw_mean=(0.5, 0.5),
                predicted_before_clamp=(target_x + error_x, 500.0),
                predicted_after_clamp=(round(target_x + error_x), 500),
                error=(error_x, 0.0),
                clipped_x=False,
                clipped_y=False,
            )
        )

    diagnostics = summarize_predictions(
        predictions,
        screen_width=1200,
        screen_height=1000,
    )

    assert diagnostics.mean_absolute_error_x < diagnostics.target_x_range * 0.15
    assert any("worst calibration point X error" in warning for warning in diagnostics.warnings)


def test_calibration_point_means_keep_samples_paired_with_targets() -> None:
    samples = [
        CalibrationSample(raw_gaze=(0.9, 0.1), target_screen=(900, 100), timestamp=0.0),
        CalibrationSample(raw_gaze=(0.1, 0.1), target_screen=(100, 100), timestamp=0.0),
        CalibrationSample(raw_gaze=(0.3, 0.1), target_screen=(100, 100), timestamp=0.0),
        CalibrationSample(raw_gaze=(0.7, 0.1), target_screen=(900, 100), timestamp=0.0),
    ]

    raw_means, targets = calibration_point_means(samples)

    assert targets == [(900, 100), (100, 100)]
    assert raw_means == [(0.8, 0.1), (0.2, 0.1)]
