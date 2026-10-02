from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from math import isfinite
from time import monotonic
from typing import Protocol

from visimove.calibration.calibration_points import CalibrationPoint, generate_calibration_points
from visimove.calibration.calibration_quality import analyze_calibration_quality
from visimove.calibration.calibration_store import CalibrationSample, new_profile, save_calibration_profile
from visimove.calibration.mapper import CalibrationMapper
from visimove.calibration.mapping_diagnostics import calibration_point_means
from visimove.calibration.mapping_diagnostics import diagnose_profile_mapping, diagnostics_to_metrics
from visimove.calibration.mapping_model import create_mapping_model
from visimove.gaze.gaze_output import GazeResult
from visimove.utils.screen import get_screen_bounds


@dataclass(frozen=True)
class CalibrationUiConfig:
    mode: str = "9"
    camera_index: int = 0
    output_path: str = "data/calibration/user_profile.json"
    mapping_model_type: str = "ridge"
    mapping_fit_strategy: str = "point_means"
    stabilization_seconds: float = 0.75
    sample_seconds: float = 1.7
    stabilization_ms: int = 800
    collection_ms: int = 2500
    min_valid_samples_per_point: int = 15
    max_collection_ms_per_point: int = 4500
    sample_interval_ms: int = 0
    gaze_backend: str = "dummy"
    detector_backend: str = "auto"
    blink_backend: str = "dummy"
    backend_metadata: dict[str, str] | None = None
    min_confidence: float = 0.5
    verbose_quality: bool = False


@dataclass(frozen=True)
class _FittedMapping:
    model_type: str
    parameters: dict[str, object]


class CalibrationGazeProvider(Protocol):
    def read_gaze(self) -> GazeResult | None: ...
    def close(self) -> None: ...


class MousePositionGazeProvider:
    """Temporary calibration input until the real gaze backend is connected."""

    def __init__(self, root: tk.Tk, screen_width: int, screen_height: int) -> None:
        self.root = root
        self.screen_width = screen_width
        self.screen_height = screen_height

    def read_gaze(self) -> GazeResult:
        x = self.root.winfo_pointerx() / max(1, self.screen_width - 1)
        y = self.root.winfo_pointery() / max(1, self.screen_height - 1)
        return GazeResult(
            raw_x=max(0.0, min(1.0, x)),
            raw_y=max(0.0, min(1.0, y)),
            confidence=1.0,
            metadata={"backend": "dummy"},
        )

    def close(self) -> None:
        return None


class CalibrationUi:
    def __init__(self, config: CalibrationUiConfig, provider: CalibrationGazeProvider | None = None) -> None:
        self.config = config
        bounds = get_screen_bounds()
        self.screen_width = bounds.width
        self.screen_height = bounds.height
        self.points = generate_calibration_points(
            mode=config.mode,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
        )
        self.samples: list[CalibrationSample] = []
        self.current_index = 0
        self.phase = "stabilize"
        self.phase_started_at = monotonic()
        self._last_sample_at = 0.0
        self._finished = False

        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.configure(background="black")
        self.root.bind("<Escape>", lambda _event: self._cancel())
        self.root.protocol("WM_DELETE_WINDOW", self._cancel)
        self.canvas = tk.Canvas(self.root, background="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.provider = provider or MousePositionGazeProvider(self.root, self.screen_width, self.screen_height)

    def run(self) -> None:
        self._draw_current_point()
        self.root.after(16, self._tick)
        self.root.mainloop()

    def _tick(self) -> None:
        if self.current_index >= len(self.points):
            self._finish()
            return

        now = monotonic()
        if self.phase == "stabilize":
            if now - self.phase_started_at >= self._stabilization_seconds:
                self.phase = "sample"
                self.phase_started_at = now
                self._last_sample_at = 0.0
        elif self.phase == "sample":
            point = self.points[self.current_index]
            if self._should_sample(now):
                self._last_sample_at = now
                gaze = self.provider.read_gaze()
            else:
                gaze = None
            if gaze is not None and self._is_usable_gaze(gaze):
                self.samples.append(
                    CalibrationSample(
                        raw_gaze=(gaze.raw_x, gaze.raw_y),
                        target_screen=(point.screen_x, point.screen_y),
                        timestamp=now,
                        confidence=gaze.confidence,
                    )
                )
            elapsed = now - self.phase_started_at
            valid_count = self._valid_sample_count_for_point(point)
            has_minimum_after_collection = (
                elapsed >= self._collection_seconds
                and valid_count >= self.config.min_valid_samples_per_point
            )
            hit_collection_cap = elapsed >= self._max_collection_seconds
            if has_minimum_after_collection or hit_collection_cap:
                self._print_point_stats(point, valid_count)
                self.current_index += 1
                self.phase = "stabilize"
                self.phase_started_at = now
                self._draw_current_point()

        self.root.after(16, self._tick)

    def _draw_current_point(self) -> None:
        self.canvas.delete("all")
        if self.current_index >= len(self.points):
            return
        point = self.points[self.current_index]
        radius = 14
        self.canvas.create_oval(
            point.screen_x - radius,
            point.screen_y - radius,
            point.screen_x + radius,
            point.screen_y + radius,
            fill="#00d084",
            outline="white",
            width=3,
        )
        self.canvas.create_text(
            self.screen_width // 2,
            self.screen_height - 60,
            text=f"Point {self.current_index + 1}/{len(self.points)} - look at the dot. Esc cancels.",
            fill="white",
            font=("Segoe UI", 18),
        )

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        raw, targets = self._mapping_training_data()
        mapping_parameters: dict[str, object] = {}
        mapping_success = False
        selected_mapping_model_type = self.config.mapping_model_type
        try:
            fitted_mapping = self._fit_mapping(raw, targets)
            selected_mapping_model_type = fitted_mapping.model_type
            mapping_parameters = fitted_mapping.parameters
            mapping_success = True
        except (ValueError, RuntimeError) as exc:
            print(f"Calibration warning: mapping model training failed. {exc}")
        if mapping_success and self.config.mapping_model_type == "auto":
            print(f"Calibration selected mapping model: {selected_mapping_model_type}")

        mapping_diagnostics = {}
        if mapping_success:
            temporary_profile = new_profile(
                screen_width=self.screen_width,
                screen_height=self.screen_height,
                camera_index=self.config.camera_index,
                calibration_points=self.points,
                samples=self.samples,
                mapping_model_type=selected_mapping_model_type,
                mapping_parameters=mapping_parameters,
                gaze_backend=self.config.gaze_backend,
                detector_backend=self.config.detector_backend,
                blink_backend=self.config.blink_backend,
                calibration_point_layout=f"{len(self.points)}-point",
                backend_metadata=self.config.backend_metadata,
            )
            mapping_diagnostics = diagnostics_to_metrics(diagnose_profile_mapping(temporary_profile))
        quality = analyze_calibration_quality(
            self.samples,
            expected_point_count=len(self.points),
            mapping_model_training_success=mapping_success,
            min_samples_per_point=self.config.min_valid_samples_per_point,
            mapping_diagnostics=mapping_diagnostics,
        )
        profile = new_profile(
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            camera_index=self.config.camera_index,
            calibration_points=self.points,
            samples=self.samples,
            mapping_model_type=selected_mapping_model_type,
            mapping_parameters=mapping_parameters,
            gaze_backend=self.config.gaze_backend,
            detector_backend=self.config.detector_backend,
            blink_backend=self.config.blink_backend,
            calibration_point_layout=f"{len(self.points)}-point",
            backend_metadata=self.config.backend_metadata,
            sample_count_per_point=quality.sample_count_per_point,
            confidence_statistics=quality.confidence_statistics,
            calibration_quality=quality.quality,
            calibration_warnings=quality.warnings,
            quality_metrics=quality.metrics,
        )
        save_calibration_profile(profile, Path(self.config.output_path))
        self._print_quality_report(quality.quality, quality.metrics, quality.confidence_statistics, quality.warnings)
        self.provider.close()
        self.canvas.delete("all")
        self.canvas.create_text(
            self.screen_width // 2,
            self.screen_height // 2,
            text=f"Calibration saved to {self.config.output_path}",
            fill="white",
            font=("Segoe UI", 24),
        )
        self.root.after(1200, self.root.destroy)

    def _cancel(self) -> None:
        self.provider.close()
        self.root.destroy()

    def _is_usable_gaze(self, gaze: GazeResult) -> bool:
        if not isfinite(gaze.raw_x) or not isfinite(gaze.raw_y):
            return False
        if gaze.confidence < self.config.min_confidence:
            return False
        return True

    @staticmethod
    def _print_quality_report(
        quality: str,
        metrics: dict[str, object],
        confidence_statistics: dict[str, float | None],
        warnings: list[str],
    ) -> None:
        print("Calibration quality report")
        print(f"  quality: {quality}")
        print(f"  points collected: {metrics.get('points_with_valid_samples')}/{metrics.get('point_count')}")
        print(f"  total samples: {metrics.get('total_samples')}")
        print(f"  valid samples: {metrics.get('valid_samples')}")
        print(
            "  raw_x min/max/range: "
            f"{metrics.get('raw_x_min')}/{metrics.get('raw_x_max')}/{metrics.get('raw_x_range')}"
        )
        print(
            "  raw_y min/max/range: "
            f"{metrics.get('raw_y_min')}/{metrics.get('raw_y_max')}/{metrics.get('raw_y_range')}"
        )
        print(
            "  confidence min/mean/max: "
            f"{confidence_statistics.get('min')}/{confidence_statistics.get('mean')}/"
            f"{confidence_statistics.get('max')}"
        )
        print(f"  mapping model training success: {metrics.get('mapping_model_training_success')}")
        for warning in warnings:
            print(f"  warning: {warning}")
        if metrics.get("point_statistics"):
            print("  per-point stats:")
            for point in metrics.get("point_statistics", []):
                if not isinstance(point, dict):
                    continue
                print(
                    "    "
                    f"point={point.get('index')} "
                    f"target={point.get('target_screen')} "
                    f"samples={point.get('valid_sample_count')} "
                    f"raw_x_mean={_fmt(point.get('raw_x_mean'))} "
                    f"raw_x_range={_fmt(point.get('raw_x_range'))} "
                    f"raw_y_mean={_fmt(point.get('raw_y_mean'))} "
                    f"raw_y_range={_fmt(point.get('raw_y_range'))} "
                    f"confidence_mean={_fmt(point.get('confidence_mean'))}"
                )

    @property
    def _stabilization_seconds(self) -> float:
        return self.config.stabilization_ms / 1000 if self.config.stabilization_ms else self.config.stabilization_seconds

    @property
    def _collection_seconds(self) -> float:
        return self.config.collection_ms / 1000 if self.config.collection_ms else self.config.sample_seconds

    @property
    def _max_collection_seconds(self) -> float:
        if self.config.max_collection_ms_per_point:
            return self.config.max_collection_ms_per_point / 1000
        return self._collection_seconds

    def _should_sample(self, now: float) -> bool:
        if self.config.sample_interval_ms <= 0:
            return True
        return now - self._last_sample_at >= self.config.sample_interval_ms / 1000

    def _valid_sample_count_for_point(self, point: CalibrationPoint) -> int:
        target = (point.screen_x, point.screen_y)
        return sum(1 for sample in self.samples if sample.target_screen == target)

    def _print_point_stats(self, point: CalibrationPoint, valid_count: int) -> None:
        target = (point.screen_x, point.screen_y)
        point_samples = [sample for sample in self.samples if sample.target_screen == target]
        raw_x_values = [sample.raw_gaze[0] for sample in point_samples]
        raw_y_values = [sample.raw_gaze[1] for sample in point_samples]
        confidence_values = [
            sample.confidence
            for sample in point_samples
            if sample.confidence is not None
        ]
        status = "ok" if valid_count >= self.config.min_valid_samples_per_point else "weak"
        print(
            "Calibration point "
            f"{self.current_index + 1}/{len(self.points)} "
            f"target=({point.screen_x},{point.screen_y}) "
            f"samples={valid_count} "
            f"raw_x_mean={_fmt(_mean_or_none(raw_x_values))} "
            f"raw_x_range={_fmt(_range_or_zero(raw_x_values))} "
            f"raw_y_mean={_fmt(_mean_or_none(raw_y_values))} "
            f"raw_y_range={_fmt(_range_or_zero(raw_y_values))} "
            f"confidence_mean={_fmt(_mean_or_none(confidence_values))} "
            f"status={status}"
        )

    def _mapping_training_data(self) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        if self.config.mapping_fit_strategy == "all_samples":
            return (
                [sample.raw_gaze for sample in self.samples],
                [(float(sample.target_screen[0]), float(sample.target_screen[1])) for sample in self.samples],
            )
        return calibration_point_means(self.samples)

    def _fit_mapping(
        self,
        raw: list[tuple[float, float]],
        targets: list[tuple[float, float]],
    ) -> _FittedMapping:
        requested_type = str(self.config.mapping_model_type).lower()
        if requested_type != "auto":
            model = create_mapping_model(requested_type)
            model.fit(raw, targets)
            return _FittedMapping(model_type=model.model_type.value, parameters=model.to_parameters())

        candidates = ("grid", "idw", "polynomial", "linear", "affine")
        best_score: float | None = None
        best_model_type: str | None = None
        best_parameters: dict[str, object] | None = None
        for candidate in candidates:
            model = create_mapping_model(candidate)
            try:
                model.fit(raw, targets)
            except (ValueError, RuntimeError):
                continue
            score = self._mapping_score(model)
            if best_score is None or score < best_score:
                best_score = score
                best_model_type = model.model_type.value
                best_parameters = model.to_parameters()

        if best_model_type is None or best_parameters is None:
            raise RuntimeError("auto mapping could not fit any candidate model")
        return _FittedMapping(model_type=best_model_type, parameters=best_parameters)

    def _mapping_score(self, model: object) -> float:
        raw_means, targets = calibration_point_means(self.samples)
        if not raw_means:
            return float("inf")
        point_score = self._mean_prediction_error_score(model, raw_means, targets)
        sample_raw = [sample.raw_gaze for sample in self.samples]
        sample_targets = [(float(sample.target_screen[0]), float(sample.target_screen[1])) for sample in self.samples]
        sample_score = self._mean_prediction_error_score(model, sample_raw, sample_targets)
        sample_clipped_ratio = self._prediction_clipped_ratio(model, sample_raw)
        return max(point_score, sample_score) + (sample_clipped_ratio * 4.0)

    def _mean_prediction_error_score(
        self,
        model: object,
        raw_values: list[tuple[float, float]],
        targets: list[tuple[float, float]],
    ) -> float:
        if not raw_values:
            return float("inf")
        total = 0.0
        for raw, target in zip(raw_values, targets):
            prediction = model.predict(raw)  # type: ignore[attr-defined]
            total += abs(prediction.x - target[0]) / max(1, self.screen_width)
            total += abs(prediction.y - target[1]) / max(1, self.screen_height)
        return total / len(raw_values)

    def _prediction_clipped_ratio(
        self,
        model: object,
        raw_values: list[tuple[float, float]],
    ) -> float:
        if not raw_values:
            return 1.0
        clipped_count = 0
        for raw in raw_values:
            prediction = model.predict(raw)  # type: ignore[attr-defined]
            if (
                prediction.x < 0
                or prediction.x > self.screen_width - 1
                or prediction.y < 0
                or prediction.y > self.screen_height - 1
            ):
                clipped_count += 1
        return clipped_count / len(raw_values)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _range_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return max(values) - min(values)


def _fmt(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, (float, int)):
        return f"{float(value):.3f}"
    return str(value)
