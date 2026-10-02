from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from statistics import mean
from typing import Any

from visimove.calibration.calibration_store import CalibrationSample


@dataclass(frozen=True)
class CalibrationQualityReport:
    quality: str
    warnings: list[str]
    metrics: dict[str, Any]
    sample_count_per_point: list[int]
    confidence_statistics: dict[str, float | None]
    mapping_model_training_success: bool


def analyze_calibration_quality(
    samples: list[CalibrationSample],
    expected_point_count: int,
    mapping_model_training_success: bool,
    min_samples_per_point: int = 15,
    min_raw_range: float = 0.05,
    borderline_raw_range: float = 0.12,
    pin_epsilon: float = 0.01,
    pinned_ratio_threshold: float = 0.5,
    min_mean_confidence: float = 0.5,
    borderline_mean_confidence: float = 0.75,
    mapping_diagnostics: dict[str, Any] | None = None,
) -> CalibrationQualityReport:
    warnings: list[str] = []
    serious_issues: list[str] = []
    review_issues: list[str] = []
    valid_samples = [sample for sample in samples if _valid_raw(sample.raw_gaze)]
    counts_by_target = Counter(sample.target_screen for sample in valid_samples)
    point_order = _target_order(valid_samples)
    sample_count_per_point = [counts_by_target[target] for target in point_order]

    missing_point_count = max(0, expected_point_count - len(counts_by_target))
    if missing_point_count:
        sample_count_per_point.extend([0] * missing_point_count)

    if not mapping_model_training_success:
        serious_issues.append("mapping model training failed")
    if not valid_samples:
        serious_issues.append("no valid gaze samples collected")

    if expected_point_count and len(counts_by_target) < expected_point_count:
        serious_issues.append(
            f"only {len(counts_by_target)}/{expected_point_count} calibration points produced valid samples"
        )

    low_sample_points = sum(1 for count in sample_count_per_point if count < min_samples_per_point)
    if low_sample_points:
        issue = (
            f"{low_sample_points} calibration point(s) have fewer than {min_samples_per_point} valid samples"
        )
        if low_sample_points == 1 and sum(sample_count_per_point) >= min_samples_per_point * max(1, expected_point_count):
            review_issues.append(issue)
        elif low_sample_points >= max(3, (expected_point_count + 1) // 2):
            serious_issues.append(issue)
        else:
            review_issues.append(issue)

    raw_x_values = [sample.raw_gaze[0] for sample in valid_samples]
    raw_y_values = [sample.raw_gaze[1] for sample in valid_samples]
    confidence_values = [
        float(sample.confidence)
        for sample in valid_samples
        if sample.confidence is not None and isfinite(float(sample.confidence))
    ]

    raw_x_min, raw_x_max, raw_x_range = _range_stats(raw_x_values)
    raw_y_min, raw_y_max, raw_y_range = _range_stats(raw_y_values)
    confidence_stats = _confidence_stats(confidence_values)

    if valid_samples and raw_x_range < min_raw_range:
        serious_issues.append(f"raw_x range is too small ({raw_x_range:.4f})")
    elif valid_samples and raw_x_range < borderline_raw_range:
        review_issues.append(f"raw_x range is borderline ({raw_x_range:.4f})")
    if valid_samples and raw_y_range < min_raw_range:
        serious_issues.append(f"raw_y range is too small ({raw_y_range:.4f})")
    elif valid_samples and raw_y_range < borderline_raw_range:
        review_issues.append(f"raw_y range is borderline ({raw_y_range:.4f})")

    total = max(1, len(valid_samples))
    pinned_x_one_ratio = sum(1 for value in raw_x_values if abs(value - 1.0) <= pin_epsilon) / total
    pinned_y_zero_ratio = sum(1 for value in raw_y_values if abs(value - 0.0) <= pin_epsilon) / total
    pinned_x_zero_ratio = sum(1 for value in raw_x_values if abs(value - 0.0) <= pin_epsilon) / total
    pinned_y_one_ratio = sum(1 for value in raw_y_values if abs(value - 1.0) <= pin_epsilon) / total

    if pinned_x_one_ratio > pinned_ratio_threshold:
        serious_issues.append(f"more than 50% of raw_x samples are pinned near 1.0 ({pinned_x_one_ratio:.0%})")
    if pinned_y_zero_ratio > pinned_ratio_threshold:
        serious_issues.append(f"more than 50% of raw_y samples are pinned near 0.0 ({pinned_y_zero_ratio:.0%})")
    if pinned_x_zero_ratio > pinned_ratio_threshold:
        serious_issues.append(f"more than 50% of raw_x samples are pinned near 0.0 ({pinned_x_zero_ratio:.0%})")
    if pinned_y_one_ratio > pinned_ratio_threshold:
        serious_issues.append(f"more than 50% of raw_y samples are pinned near 1.0 ({pinned_y_one_ratio:.0%})")

    mean_confidence = confidence_stats["mean"]
    if mean_confidence is not None and mean_confidence < min_mean_confidence:
        serious_issues.append(f"mean gaze confidence is low ({mean_confidence:.2f})")
    elif mean_confidence is not None and mean_confidence < borderline_mean_confidence:
        review_issues.append(f"mean gaze confidence is borderline ({mean_confidence:.2f})")

    mapping_diagnostics = mapping_diagnostics or {}
    mapping_warnings = [str(warning) for warning in mapping_diagnostics.get("mapping_warnings", [])]
    mapped_y_stuck = bool(mapping_diagnostics.get("mapped_y_stuck_at_zero", False))
    clipped_ratio = float(mapping_diagnostics.get("clipped_prediction_ratio", 0.0))
    negative_y_ratio = float(mapping_diagnostics.get("negative_y_before_clamp_ratio", 0.0))
    sample_clipped_ratio = float(mapping_diagnostics.get("sample_clipped_prediction_ratio", 0.0))
    sample_negative_y_ratio = float(mapping_diagnostics.get("sample_negative_y_before_clamp_ratio", 0.0))
    mapped_x_range = float(mapping_diagnostics.get("mapped_x_range", 0.0))
    mapped_y_range = float(mapping_diagnostics.get("mapped_y_range", 0.0))
    mapping_mae_x = _optional_float(mapping_diagnostics.get("mapping_mean_absolute_error_x"))
    mapping_mae_y = _optional_float(mapping_diagnostics.get("mapping_mean_absolute_error_y"))
    sample_mapping_mae_x = _optional_float(mapping_diagnostics.get("mapping_sample_mean_absolute_error_x"))
    sample_mapping_mae_y = _optional_float(mapping_diagnostics.get("mapping_sample_mean_absolute_error_y"))
    target_x_range = _optional_float(mapping_diagnostics.get("mapping_target_x_range"))
    target_y_range = _optional_float(mapping_diagnostics.get("mapping_target_y_range"))
    if mapped_y_stuck:
        serious_issues.append("mapped Y is stuck at 0 after calibration")
    if clipped_ratio > 0.5:
        serious_issues.append(f"too many calibration predictions are clipped ({clipped_ratio:.0%})")
    elif clipped_ratio > 0.25:
        review_issues.append(f"many calibration predictions are clipped ({clipped_ratio:.0%})")
    if negative_y_ratio > 0.5:
        serious_issues.append(f"too many predicted Y values are negative before clamping ({negative_y_ratio:.0%})")
    elif negative_y_ratio > 0.25:
        review_issues.append(f"many predicted Y values are negative before clamping ({negative_y_ratio:.0%})")
    if sample_clipped_ratio > 0.5:
        serious_issues.append(f"too many raw calibration samples map outside the screen ({sample_clipped_ratio:.0%})")
    elif sample_clipped_ratio > 0.25:
        review_issues.append(f"many raw calibration samples map outside the screen ({sample_clipped_ratio:.0%})")
    if sample_negative_y_ratio > 0.5:
        serious_issues.append(
            f"too many raw calibration samples predict negative Y before clamping ({sample_negative_y_ratio:.0%})"
        )
    elif sample_negative_y_ratio > 0.25:
        review_issues.append(
            f"many raw calibration samples predict negative Y before clamping ({sample_negative_y_ratio:.0%})"
        )
    if mapping_diagnostics and mapped_x_range < 1.0:
        serious_issues.append("mapped X range is too small after calibration")
    if mapping_diagnostics and mapped_y_range < 1.0:
        serious_issues.append("mapped Y range is too small after calibration")
    if mapping_mae_x is not None and target_x_range and target_x_range > 0:
        x_error_ratio = mapping_mae_x / target_x_range
        if x_error_ratio > 0.30:
            serious_issues.append(f"mean absolute X error is very high ({mapping_mae_x:.1f}px)")
        elif x_error_ratio > 0.15:
            review_issues.append(f"mean absolute X error is high ({mapping_mae_x:.1f}px)")
    if mapping_mae_y is not None and target_y_range and target_y_range > 0:
        y_error_ratio = mapping_mae_y / target_y_range
        if y_error_ratio > 0.30:
            serious_issues.append(f"mean absolute Y error is very high ({mapping_mae_y:.1f}px)")
        elif y_error_ratio > 0.15:
            review_issues.append(f"mean absolute Y error is high ({mapping_mae_y:.1f}px)")
    if sample_mapping_mae_x is not None and target_x_range and target_x_range > 0:
        sample_x_error_ratio = sample_mapping_mae_x / target_x_range
        if sample_x_error_ratio > 0.35:
            serious_issues.append(f"raw-sample mean absolute X error is very high ({sample_mapping_mae_x:.1f}px)")
        elif sample_x_error_ratio > 0.20:
            review_issues.append(f"raw-sample mean absolute X error is high ({sample_mapping_mae_x:.1f}px)")
    if sample_mapping_mae_y is not None and target_y_range and target_y_range > 0:
        sample_y_error_ratio = sample_mapping_mae_y / target_y_range
        if sample_y_error_ratio > 0.35:
            serious_issues.append(f"raw-sample mean absolute Y error is very high ({sample_mapping_mae_y:.1f}px)")
        elif sample_y_error_ratio > 0.20:
            review_issues.append(f"raw-sample mean absolute Y error is high ({sample_mapping_mae_y:.1f}px)")
    for warning in mapping_warnings:
        if warning not in serious_issues and warning not in review_issues:
            review_issues.append(warning)

    valid_total = len(valid_samples)
    very_low_total = valid_total < min_samples_per_point * max(1, expected_point_count) * 0.6
    if very_low_total:
        serious_issues.append(
            f"total valid samples are too low ({valid_total}; expected about {min_samples_per_point * expected_point_count})"
        )

    if serious_issues:
        quality = "poor"
    elif review_issues:
        if low_sample_points == 1 and len(review_issues) == 1:
            quality = "acceptable"
        else:
            quality = "needs_review"
    else:
        quality = "good"

    warnings = [*serious_issues, *review_issues]

    metrics: dict[str, Any] = {
        "point_count": expected_point_count,
        "points_with_valid_samples": len(counts_by_target),
        "total_samples": len(samples),
        "valid_samples": len(valid_samples),
        "raw_x_min": raw_x_min,
        "raw_x_max": raw_x_max,
        "raw_x_range": raw_x_range,
        "raw_y_min": raw_y_min,
        "raw_y_max": raw_y_max,
        "raw_y_range": raw_y_range,
        "pinned_x_near_1_ratio": pinned_x_one_ratio if valid_samples else 0.0,
        "pinned_y_near_0_ratio": pinned_y_zero_ratio if valid_samples else 0.0,
        "pinned_x_near_0_ratio": pinned_x_zero_ratio if valid_samples else 0.0,
        "pinned_y_near_1_ratio": pinned_y_one_ratio if valid_samples else 0.0,
        "mapping_model_training_success": mapping_model_training_success,
        "low_sample_point_count": low_sample_points,
        "serious_issue_count": len(serious_issues),
        "review_issue_count": len(review_issues),
        "point_statistics": _point_statistics(valid_samples),
        **mapping_diagnostics,
    }

    return CalibrationQualityReport(
        quality=quality,
        warnings=warnings,
        metrics=metrics,
        sample_count_per_point=sample_count_per_point,
        confidence_statistics=confidence_stats,
        mapping_model_training_success=mapping_model_training_success,
    )


def _valid_raw(raw_gaze: tuple[float, float]) -> bool:
    return len(raw_gaze) == 2 and all(isfinite(float(value)) for value in raw_gaze)


def _range_stats(values: list[float]) -> tuple[float | None, float | None, float]:
    if not values:
        return None, None, 0.0
    minimum = min(values)
    maximum = max(values)
    return minimum, maximum, maximum - minimum


def _confidence_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": mean(values), "max": max(values)}


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _point_statistics(samples: list[CalibrationSample]) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    grouped: dict[tuple[int, int], list[CalibrationSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.target_screen, []).append(sample)
    for index, target in enumerate(_target_order(samples), start=1):
        point_samples = grouped[target]
        raw_x_values = [sample.raw_gaze[0] for sample in point_samples]
        raw_y_values = [sample.raw_gaze[1] for sample in point_samples]
        confidence_values = [
            float(sample.confidence)
            for sample in point_samples
            if sample.confidence is not None and isfinite(float(sample.confidence))
        ]
        _, _, raw_x_range = _range_stats(raw_x_values)
        _, _, raw_y_range = _range_stats(raw_y_values)
        confidence_stats = _confidence_stats(confidence_values)
        stats.append(
            {
                "index": index,
                "target_screen": list(target),
                "valid_sample_count": len(point_samples),
                "raw_x_mean": mean(raw_x_values) if raw_x_values else None,
                "raw_x_range": raw_x_range,
                "raw_y_mean": mean(raw_y_values) if raw_y_values else None,
                "raw_y_range": raw_y_range,
                "confidence_mean": confidence_stats["mean"],
            }
        )
    return stats


def _target_order(samples: list[CalibrationSample]) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for sample in samples:
        if sample.target_screen in seen:
            continue
        seen.add(sample.target_screen)
        order.append(sample.target_screen)
    return order
