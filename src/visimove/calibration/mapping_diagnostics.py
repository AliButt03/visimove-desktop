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
    sample_mean_absolute_error_x: float
    sample_mean_absolute_error_y: float
    sample_clipped_prediction_ratio: float
    sample_negative_y_before_clamp_ratio: float
    target_x_range: float
    target_y_range: float
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
    for target in _target_order(samples):
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
    mapper.clamp_raw_input_to_calibration_domain = True
    mapper.clamp_raw_input_to_mapping_domain = False
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
    sample_metrics = summarize_sample_predictions(profile.raw_gaze_samples, mapper)
    return summarize_predictions(predictions, profile.screen_width, profile.screen_height, sample_metrics)


def summarize_predictions(
    predictions: list[PointPredictionDiagnostic],
    screen_width: int,
    screen_height: int,
    sample_metrics: dict[str, float] | None = None,
) -> MappingDiagnostics:
    if not predictions:
        return MappingDiagnostics(
            point_predictions=[],
            mean_absolute_error_x=0.0,
            mean_absolute_error_y=0.0,
            rmse_x=0.0,
            rmse_y=0.0,
            sample_mean_absolute_error_x=0.0,
            sample_mean_absolute_error_y=0.0,
            sample_clipped_prediction_ratio=0.0,
            sample_negative_y_before_clamp_ratio=0.0,
            target_x_range=0.0,
            target_y_range=0.0,
            predicted_x_range=0.0,
            predicted_y_range=0.0,
            clipped_prediction_ratio=0.0,
            negative_y_before_clamp_ratio=0.0,
            mapped_y_stuck_at_zero=False,
            warnings=["no predictions"],
        )
    sample_metrics = sample_metrics or {
        "sample_mean_absolute_error_x": 0.0,
        "sample_mean_absolute_error_y": 0.0,
        "sample_clipped_prediction_ratio": 0.0,
        "sample_negative_y_before_clamp_ratio": 0.0,
    }

    errors_x = [prediction.error[0] for prediction in predictions]
    errors_y = [prediction.error[1] for prediction in predictions]
    predicted_x = [prediction.predicted_after_clamp[0] for prediction in predictions]
    predicted_y = [prediction.predicted_after_clamp[1] for prediction in predictions]
    target_x = [prediction.target[0] for prediction in predictions]
    target_y = [prediction.target[1] for prediction in predictions]
    clipped = [prediction for prediction in predictions if prediction.clipped_x or prediction.clipped_y]
    negative_y = [prediction for prediction in predictions if prediction.predicted_before_clamp[1] < 0]
    y_zero_count = sum(1 for value in predicted_y if value == 0)
    predicted_x_range = max(predicted_x) - min(predicted_x)
    predicted_y_range = max(predicted_y) - min(predicted_y)
    target_x_range = max(target_x) - min(target_x)
    target_y_range = max(target_y) - min(target_y)

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
    max_abs_x = max(abs(error) for error in errors_x)
    max_abs_y = max(abs(error) for error in errors_y)
    if target_x_range > 0 and max_abs_x > target_x_range * 0.15:
        warnings.append(f"worst calibration point X error is high ({max_abs_x:.1f}px)")
    if target_y_range > 0 and max_abs_y > target_y_range * 0.15:
        warnings.append(f"worst calibration point Y error is high ({max_abs_y:.1f}px)")
    if target_x_range > 0 and mean_abs_x > target_x_range * 0.15:
        warnings.append(f"mean absolute X error is high ({mean_abs_x:.1f}px)")
    elif mean_abs_x > screen_width * 0.25:
        warnings.append(f"mean absolute X error is high ({mean_abs_x:.1f}px)")
    if target_y_range > 0 and mean_abs_y > target_y_range * 0.15:
        warnings.append(f"mean absolute Y error is high ({mean_abs_y:.1f}px)")
    elif mean_abs_y > screen_height * 0.25:
        warnings.append(f"mean absolute Y error is high ({mean_abs_y:.1f}px)")

    sample_mean_abs_x = sample_metrics["sample_mean_absolute_error_x"]
    sample_mean_abs_y = sample_metrics["sample_mean_absolute_error_y"]
    sample_clipped_ratio = sample_metrics["sample_clipped_prediction_ratio"]
    sample_negative_y_ratio = sample_metrics["sample_negative_y_before_clamp_ratio"]
    if sample_clipped_ratio > 0.50:
        warnings.append(f"too many raw calibration samples map outside the screen ({sample_clipped_ratio:.0%})")
    elif sample_clipped_ratio > 0.25:
        warnings.append(f"many raw calibration samples map outside the screen ({sample_clipped_ratio:.0%})")
    if sample_negative_y_ratio > 0.25:
        warnings.append(f"many raw calibration samples predict negative Y before clamping ({sample_negative_y_ratio:.0%})")
    if target_x_range > 0 and sample_mean_abs_x > target_x_range * 0.30:
        warnings.append(f"raw-sample mean absolute X error is very high ({sample_mean_abs_x:.1f}px)")
    elif target_x_range > 0 and sample_mean_abs_x > target_x_range * 0.20:
        warnings.append(f"raw-sample mean absolute X error is high ({sample_mean_abs_x:.1f}px)")
    if target_y_range > 0 and sample_mean_abs_y > target_y_range * 0.30:
        warnings.append(f"raw-sample mean absolute Y error is very high ({sample_mean_abs_y:.1f}px)")
    elif target_y_range > 0 and sample_mean_abs_y > target_y_range * 0.20:
        warnings.append(f"raw-sample mean absolute Y error is high ({sample_mean_abs_y:.1f}px)")

    return MappingDiagnostics(
        point_predictions=predictions,
        mean_absolute_error_x=mean_abs_x,
        mean_absolute_error_y=mean_abs_y,
        rmse_x=rmse_x,
        rmse_y=rmse_y,
        sample_mean_absolute_error_x=sample_mean_abs_x,
        sample_mean_absolute_error_y=sample_mean_abs_y,
        sample_clipped_prediction_ratio=sample_clipped_ratio,
        sample_negative_y_before_clamp_ratio=sample_negative_y_ratio,
        target_x_range=target_x_range,
        target_y_range=target_y_range,
        predicted_x_range=predicted_x_range,
        predicted_y_range=predicted_y_range,
        clipped_prediction_ratio=len(clipped) / len(predictions),
        negative_y_before_clamp_ratio=len(negative_y) / len(predictions),
        mapped_y_stuck_at_zero=y_zero_count / len(predictions) > 0.4,
        warnings=warnings,
    )


def summarize_sample_predictions(
    samples: list[CalibrationSample],
    mapper: CalibrationMapper,
) -> dict[str, float]:
    if not samples:
        return {
            "sample_mean_absolute_error_x": 0.0,
            "sample_mean_absolute_error_y": 0.0,
            "sample_clipped_prediction_ratio": 0.0,
            "sample_negative_y_before_clamp_ratio": 0.0,
        }
    errors_x: list[float] = []
    errors_y: list[float] = []
    clipped_count = 0
    negative_y_count = 0
    for sample in samples:
        mapped, debug = mapper.map_with_debug(
            GazeEstimate(point=Point(sample.raw_gaze[0], sample.raw_gaze[1]), confidence=1.0)
        )
        errors_x.append(mapped.x - sample.target_screen[0])
        errors_y.append(mapped.y - sample.target_screen[1])
        if debug.clipped_x or debug.clipped_y:
            clipped_count += 1
        if debug.before_clamp_y < 0:
            negative_y_count += 1
    return {
        "sample_mean_absolute_error_x": mean(abs(error) for error in errors_x),
        "sample_mean_absolute_error_y": mean(abs(error) for error in errors_y),
        "sample_clipped_prediction_ratio": clipped_count / len(samples),
        "sample_negative_y_before_clamp_ratio": negative_y_count / len(samples),
    }


def diagnostics_to_metrics(diagnostics: MappingDiagnostics) -> dict[str, Any]:
    return {
        "mapping_mean_absolute_error_x": diagnostics.mean_absolute_error_x,
        "mapping_mean_absolute_error_y": diagnostics.mean_absolute_error_y,
        "mapping_rmse_x": diagnostics.rmse_x,
        "mapping_rmse_y": diagnostics.rmse_y,
        "mapping_sample_mean_absolute_error_x": diagnostics.sample_mean_absolute_error_x,
        "mapping_sample_mean_absolute_error_y": diagnostics.sample_mean_absolute_error_y,
        "sample_clipped_prediction_ratio": diagnostics.sample_clipped_prediction_ratio,
        "sample_negative_y_before_clamp_ratio": diagnostics.sample_negative_y_before_clamp_ratio,
        "mapping_target_x_range": diagnostics.target_x_range,
        "mapping_target_y_range": diagnostics.target_y_range,
        "mapped_x_range": diagnostics.predicted_x_range,
        "mapped_y_range": diagnostics.predicted_y_range,
        "clipped_prediction_ratio": diagnostics.clipped_prediction_ratio,
        "negative_y_before_clamp_ratio": diagnostics.negative_y_before_clamp_ratio,
        "mapped_y_stuck_at_zero": diagnostics.mapped_y_stuck_at_zero,
        "mapping_warnings": diagnostics.warnings,
    }


def _target_order(samples: list[CalibrationSample]) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for sample in samples:
        if sample.target_screen in seen:
            continue
        seen.add(sample.target_screen)
        order.append(sample.target_screen)
    return order
