from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from statistics import mean
from typing import Any

from visimove.calibration.calibration_store import CalibrationProfile, CalibrationSample
from visimove.calibration.mapper import CalibrationMapper
from visimove.types import GazeEstimate, Point


@dataclass(frozen=True)
class PointPredictionDiagnostic:
    index: int
    target: tuple[int, int]
    raw_mean: tuple[float, float]
    predicted_before_clamp: tuple[float, float]
    predicted_after_clamp: tuple[int, int]
    error: tuple[float, float]
    clipped_x: bool
    clipped_y: bool


@dataclass(frozen=True)
class MappingDiagnostics:
    point_predictions: list[PointPredictionDiagnostic]
    mean_absolute_error_x: float
    mean_absolute_error_y: float
    rmse_x: float
    rmse_y: float
    predicted_x_range: float
    predicted_y_range: float
    clipped_prediction_ratio: float
    negative_y_before_clamp_ratio: float
    mapped_y_stuck_at_zero: bool
    warnings: list[str]


def calibration_point_means(samples: list[CalibrationSample]) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    grouped: dict[tuple[int, int], list[CalibrationSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.target_screen].append(sample)

    raw_means: list[tuple[float, float]] = []
    targets: list[tuple[int, int]] = []
    for target in sorted(grouped):
        target_samples = grouped[target]
        raw_means.append(
            (
                mean(sample.raw_gaze[0] for sample in target_samples),
                mean(sample.raw_gaze[1] for sample in target_samples),
            )
        )
        targets.append(target)
    return raw_means, targets


def diagnose_profile_mapping(profile: CalibrationProfile) -> MappingDiagnostics:
    mapper = CalibrationMapper.from_profile(profile)
    raw_means, targets = calibration_point_means(profile.raw_gaze_samples)
    predictions: list[PointPredictionDiagnostic] = []
    for index, (raw, target) in enumerate(zip(raw_means, targets), start=1):
        mapped, debug = mapper.map_with_debug(GazeEstimate(point=Point(raw[0], raw[1]), confidence=1.0))
        predictions.append(
            PointPredictionDiagnostic(
                index=index,
                target=target,
                raw_mean=raw,
                predicted_before_clamp=(debug.before_clamp_x, debug.before_clamp_y),
                predicted_after_clamp=(mapped.x, mapped.y),
                error=(mapped.x - target[0], mapped.y - target[1]),
                clipped_x=debug.clipped_x,
                clipped_y=debug.clipped_y,
            )
        )
    return summarize_predictions(predictions, profile.screen_width, profile.screen_height)


def summarize_predictions(
    predictions: list[PointPredictionDiagnostic],
    screen_width: int,
    screen_height: int,
) -> MappingDiagnostics:
    if not predictions:
        return MappingDiagnostics([], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, ["no predictions"])

    errors_x = [prediction.error[0] for prediction in predictions]
    errors_y = [prediction.error[1] for prediction in predictions]
    predicted_x = [prediction.predicted_after_clamp[0] for prediction in predictions]
    predicted_y = [prediction.predicted_after_clamp[1] for prediction in predictions]
    clipped = [prediction for prediction in predictions if prediction.clipped_x or prediction.clipped_y]
    negative_y = [prediction for prediction in predictions if prediction.predicted_before_clamp[1] < 0]
    y_zero_count = sum(1 for value in predicted_y if value == 0)
    predicted_x_range = max(predicted_x) - min(predicted_x)
    predicted_y_range = max(predicted_y) - min(predicted_y)

    warnings: list[str] = []
    if predicted_x_range < screen_width * 0.25:
        warnings.append(f"mapped X range is too small ({predicted_x_range:.1f}px)")
    if predicted_y_range < screen_height * 0.25:
        warnings.append(f"mapped Y range is too small ({predicted_y_range:.1f}px)")
    if y_zero_count / len(predictions) > 0.4:
        warnings.append("mapped Y is stuck at 0 for many calibration points")
    if len(clipped) / len(predictions) > 0.25:
        warnings.append("many calibration predictions are clipped to a screen boundary")
    if len(negative_y) / len(predictions) > 0.25:
        warnings.append("many predicted Y values are negative before clamping")

    mean_abs_x = mean(abs(error) for error in errors_x)
    mean_abs_y = mean(abs(error) for error in errors_y)
    rmse_x = sqrt(mean(error * error for error in errors_x))
    rmse_y = sqrt(mean(error * error for error in errors_y))
    if mean_abs_x > screen_width * 0.25:
        warnings.append(f"mean absolute X error is high ({mean_abs_x:.1f}px)")
    if mean_abs_y > screen_height * 0.25:
        warnings.append(f"mean absolute Y error is high ({mean_abs_y:.1f}px)")

    return MappingDiagnostics(
        point_predictions=predictions,
        mean_absolute_error_x=mean_abs_x,
        mean_absolute_error_y=mean_abs_y,
        rmse_x=rmse_x,
        rmse_y=rmse_y,
        predicted_x_range=predicted_x_range,
        predicted_y_range=predicted_y_range,
        clipped_prediction_ratio=len(clipped) / len(predictions),
        negative_y_before_clamp_ratio=len(negative_y) / len(predictions),
        mapped_y_stuck_at_zero=y_zero_count / len(predictions) > 0.4,
        warnings=warnings,
    )


def diagnostics_to_metrics(diagnostics: MappingDiagnostics) -> dict[str, Any]:
    return {
        "mapping_mean_absolute_error_x": diagnostics.mean_absolute_error_x,
        "mapping_mean_absolute_error_y": diagnostics.mean_absolute_error_y,
        "mapping_rmse_x": diagnostics.rmse_x,
        "mapping_rmse_y": diagnostics.rmse_y,
        "mapped_x_range": diagnostics.predicted_x_range,
        "mapped_y_range": diagnostics.predicted_y_range,
        "clipped_prediction_ratio": diagnostics.clipped_prediction_ratio,
        "negative_y_before_clamp_ratio": diagnostics.negative_y_before_clamp_ratio,
        "mapped_y_stuck_at_zero": diagnostics.mapped_y_stuck_at_zero,
        "mapping_warnings": diagnostics.warnings,
    }
