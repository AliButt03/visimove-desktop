from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any

import cv2

from visimove.blink import BlinkModel, BlinkResult, BlinkStateMachine, BlinkStateMachineConfig, ClickEvent
from visimove.blink import KeyboardDummyBlinkModel, OcecBlinkAdapter, OnnxBlinkModel, OnnxBlinkModelConfig
from visimove.calibration import AxisAdjustmentConfig, AxisAdjustmentResult, apply_axis_adjustment
from visimove.calibration import CalibrationMapper, LiveTrackingQualityMonitor, LiveTrackingQualityState
from visimove.calibration import MappingDebugInfo, load_calibration_profile
from visimove.camera.base import CameraConfig
from visimove.camera.webcam import WebcamCamera
from visimove.cursor import CursorController, CursorSafety, CursorSafetyConfig, DwellSelector, VelocityCursorFilter
from visimove.cursor import DryRunCursorController, PyAutoGuiCursorController, Win32CursorController
from visimove.detection import BaseDetector, DummyDetector, MediaPipeFaceMeshDetector, OpenCvHaarDetector
from visimove.detection import YoloDetector
from visimove.gaze import BaseGazeModel, EyeTraxAdapter, GazeBackendUnavailable
from visimove.gaze import GazeFollowerAdapter, GazeResult, MobileGazeAdapter, MovingDummyGazeModel
from visimove.pipeline.performance_monitor import PipelinePerformanceMonitor
from visimove.smoothing import AdaptiveSmoothingFilter, DeadzoneFilter, EmaFilter, FixationFilter, KalmanFilter2D, OneEuroFilter2D
from visimove.types import DetectionResult, ScreenPoint
from visimove.utils.screen import get_screen_bounds


@dataclass(frozen=True)
class RealtimePipelineConfig:
    show_preview: bool = True
    dry_run: bool = True
    report_interval_seconds: float = 1.0
    show_debug: bool = False
    debug_interval_seconds: float = 0.5
    cursor_enabled: bool = False
    calibration_quality: str = "unknown"
    calibration_warnings: tuple[str, ...] = ()
    allow_unstable_live_gaze: bool = False
    block_cursor_when_outside_calibration_domain: bool = True
    live_domain_violation_ratio_threshold: float = 0.30
    live_domain_window_size: int = 30
    axis_adjustment: AxisAdjustmentConfig = field(default_factory=AxisAdjustmentConfig)
    cursor_control_mode: str = "absolute"


class RealtimeVisiMovePipeline:
    def __init__(
        self,
        camera: WebcamCamera,
        detector: BaseDetector,
        gaze_model: BaseGazeModel,
        blink_model: BlinkModel,
        blink_state_machine: BlinkStateMachine,
        mapper: CalibrationMapper,
        smoother: Any,
        cursor: CursorController,
        performance: PipelinePerformanceMonitor,
        config: RealtimePipelineConfig,
        dwell_selector: DwellSelector | None = None,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.gaze_model = gaze_model
        self.blink_model = blink_model
        self.blink_state_machine = blink_state_machine
        self.mapper = mapper
        self.smoother = smoother
        self.cursor = cursor
        self.performance = performance
        self.config = config
        self.dwell_selector = dwell_selector
        self._last_debug_at = 0.0
        self.live_quality_monitor = LiveTrackingQualityMonitor(
            window_size=config.live_domain_window_size,
            unsafe_ratio_threshold=config.live_domain_violation_ratio_threshold,
        )

    def run(self) -> None:
        self.camera.open()
        try:
            while True:
                now = monotonic()
                with self.performance.stage("camera"):
                    webcam_frame = self.camera.read()
                if webcam_frame is None:
                    raise RuntimeError("Webcam returned no frame. Try another camera index.")

                frame = webcam_frame.frame
                with self.performance.stage("detector"):
                    detection = self.detector.detect(frame, now)
                with self.performance.stage("gaze"):
                    gaze = self.gaze_model.estimate(frame, detection)
                with self.performance.stage("blink"):
                    blink = self.blink_model.infer(frame, detection)
                    click_event = self.blink_state_machine.update(
                        blink.combined_closed_probability,
                        now,
                    )
                with self.performance.stage("calibration"):
                    screen_point, mapping_debug = self.mapper.map_with_debug(gaze)
                    live_quality = self.live_quality_monitor.update(mapping_debug)
                    axis_adjustment = apply_axis_adjustment(
                        screen_point,
                        self.mapper.screen_width,
                        self.mapper.screen_height,
                        self.config.axis_adjustment,
                    )
                with self.performance.stage("smoothing"):
                    adjusted_point = axis_adjustment.after
                    control_point = (
                        _velocity_control_point(gaze, self.mapper.screen_width, self.mapper.screen_height)
                        if self.config.cursor_control_mode == "velocity"
                        else adjusted_point
                    )
                    target_visible = detection.found and detection.eyes is not None
                    smoothing_target_visible = self._smoothing_target_visible(
                        gaze.confidence,
                        target_visible,
                        live_quality,
                    )
                    if smoothing_target_visible:
                        smoothed_xy = self.smoother.update(control_point.x, control_point.y, now)
                    else:
                        smoothed_xy = self.smoother.update(None, None, now)
                    if smoothed_xy is None:
                        smoothed = control_point
                    else:
                        smoothed = ScreenPoint(round(smoothed_xy[0]), round(smoothed_xy[1]))
                with self.performance.stage("cursor"):
                    skip_reason = self._cursor_skip_reason(gaze.confidence, target_visible, live_quality)
                    cursor_target_visible = target_visible and skip_reason == "none"
                    self.cursor.move_to(
                        smoothed.x,
                        smoothed.y,
                        gaze_confidence=gaze.confidence,
                        target_visible=cursor_target_visible,
                    )
                    click_name = self._handle_click_event(click_event) if cursor_target_visible else "none"
                    dwell_allowed = cursor_target_visible and self._dwell_allowed()
                    if dwell_allowed and click_name == "none" and self.dwell_selector is not None:
                        if self.dwell_selector.update(smoothed, timestamp=now):
                            self.cursor.left_click()
                            click_name = "dwell_left_click"
                    elif self.dwell_selector is not None:
                        self.dwell_selector.reset()

                self.performance.mark_frame()
                if self.config.show_debug and now - self._last_debug_at >= self.config.debug_interval_seconds:
                    self._last_debug_at = now
                    self._print_debug(
                        detection=detection,
                        gaze=gaze,
                        blink=blink,
                        mapped=screen_point,
                        axis_adjustment=axis_adjustment,
                        mapping_debug=mapping_debug,
                        live_quality=live_quality,
                        smoothed=smoothed,
                        click_name=click_name,
                        cursor_skip_reason=skip_reason,
                    )
                if self.config.show_preview:
                    self._draw_preview(frame, detection, gaze, blink, smoothed)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("b") and hasattr(self.blink_model, "trigger"):
                        self.blink_model.trigger()
                    if key == ord("p"):
                        self._toggle_cursor_pause()
                    if key == ord("c"):
                        self._recenter_velocity_control()
                    if key == ord("q"):
                        break

                if self.performance.should_report(self.config.report_interval_seconds):
                    print(self.performance.format_report())
        finally:
            self.camera.close()
            if self.config.show_preview:
                cv2.destroyAllWindows()

    @staticmethod
    def _draw_preview(
        frame: Any,
        detection: DetectionResult,
        gaze: GazeResult,
        blink: BlinkResult,
        point: ScreenPoint,
    ) -> None:
        if detection.face_box:
            x, y, w, h = detection.face_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 180, 70), 2)
        if detection.eyes:
            for eye in (detection.eyes.left, detection.eyes.right):
                if eye:
                    x, y, w, h = eye
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (230, 180, 40), 2)

        status = (
            f"gaze_conf={gaze.confidence:.2f} gaze=({gaze.point.x:.2f},{gaze.point.y:.2f}) "
            f"screen=({point.x},{point.y}) blink={blink.combined_closed_probability:.2f}"
        )
        cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, "q: quit | b: fake blink | p: pause/resume", (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("VisiMove Tracking", frame)

    def _toggle_cursor_pause(self) -> None:
        safety = getattr(self.cursor, "safety", None)
        if safety is not None and getattr(safety, "paused", True):
            self.cursor.resume()
            set_position = getattr(self.smoother, "set_position", None)
            if callable(set_position):
                set_position(getattr(safety, "current_position", None))
            begin_neutral_calibration = getattr(self.smoother, "begin_neutral_calibration", None)
            if callable(begin_neutral_calibration):
                begin_neutral_calibration()
                print("Look at the screen center for one second to set neutral gaze.")
            if self.dwell_selector is not None:
                self.dwell_selector.reset()
            print("Cursor resumed.")
        else:
            self.cursor.pause()
            print("Cursor paused.")

    def _recenter_velocity_control(self) -> None:
        begin_neutral_calibration = getattr(self.smoother, "begin_neutral_calibration", None)
        if not callable(begin_neutral_calibration):
            return
        begin_neutral_calibration()
        if self.dwell_selector is not None:
            self.dwell_selector.reset()
        print("Velocity control recentering: look at the screen center for one second.")

    def _dwell_allowed(self) -> bool:
        if self.config.cursor_control_mode != "velocity":
            return True
        return _format_smoothing_state(self.smoother) == "velocity_hold"
    def _cursor_skip_reason(
        self,
        gaze_confidence: float,
        target_visible: bool,
        live_quality: LiveTrackingQualityState,
    ) -> str:
        if self.config.dry_run or not self.config.cursor_enabled:
            return "cursor disabled"
        safety = getattr(self.cursor, "safety", None)
        if safety is not None and getattr(safety, "paused", False):
            return "cursor paused"
        if not target_visible:
            return "no face/eyes"
        min_confidence = getattr(getattr(safety, "config", None), "min_gaze_confidence", 0.0)
        if gaze_confidence < min_confidence:
            return "low gaze confidence"
        if (
            self.config.block_cursor_when_outside_calibration_domain
            and not self.config.allow_unstable_live_gaze
            and live_quality.quality == "unsafe"
        ):
            return "live gaze outside calibrated domain"
        return "none"

    def _smoothing_target_visible(
        self,
        gaze_confidence: float,
        target_visible: bool,
        live_quality: LiveTrackingQualityState,
    ) -> bool:
        if not target_visible:
            return False
        safety = getattr(self.cursor, "safety", None)
        if (
            self.config.cursor_control_mode == "velocity"
            and self.config.cursor_enabled
            and safety is not None
            and getattr(safety, "paused", False)
        ):
            return False
        min_confidence = getattr(getattr(safety, "config", None), "min_gaze_confidence", 0.0)
        if gaze_confidence < min_confidence:
            return False
        if (
            self.config.block_cursor_when_outside_calibration_domain
            and not self.config.allow_unstable_live_gaze
            and live_quality.quality == "unsafe"
        ):
            return False
        return True

    def _print_debug(
        self,
        detection: DetectionResult,
        gaze: GazeResult,
        blink: BlinkResult,
        mapped: ScreenPoint,
        axis_adjustment: AxisAdjustmentResult,
        mapping_debug: MappingDebugInfo,
        live_quality: LiveTrackingQualityState,
        smoothed: ScreenPoint,
        click_name: str,
        cursor_skip_reason: str,
    ) -> None:
        gaze_backend = str(gaze.metadata.get("backend", "unknown"))
        gaze_reason = str(gaze.metadata.get("unavailable_reason", "none"))
        fallback = "yes" if gaze.metadata.get("fallback", False) else "no"
        print(
            "DEBUG "
            f"gaze_backend={gaze_backend} "
            f"fallback={fallback} "
            f"raw_gaze=({gaze.point.x:.3f},{gaze.point.y:.3f}) "
            f"{_format_gaze_filter_debug(gaze.metadata)}"
            f"{_format_gaze_native_debug(gaze.metadata)}"
            f"calibration_raw_domain={_format_domain(mapping_debug)} "
            f"raw_domain_status={mapping_debug.raw_domain_status} "
            f"raw_domain_violation={_format_violations(mapping_debug.raw_domain_violations)} "
            f"mapped_input_after_domain_clamp=({mapping_debug.mapped_input_x:.3f},{mapping_debug.mapped_input_y:.3f}) "
            f"mapped_raw_before_screen_clamp=({mapping_debug.before_clamp_x:.1f},{mapping_debug.before_clamp_y:.1f}) "
            f"mapped_after_clamp=({mapped.x},{mapped.y}) "
            f"edge_reach_after=({axis_adjustment.after_edge_reach.x},{axis_adjustment.after_edge_reach.y}) "
            f"edge_reach_enabled={'yes' if axis_adjustment.edge_reach_enabled else 'no'} "
            f"edge_margin_px={axis_adjustment.edge_margin_px} "
            f"edge_boost={'yes' if axis_adjustment.edge_boost_enabled else 'no'} "
            f"edge_boost_gamma={axis_adjustment.edge_boost_gamma:.2f} "
            f"adjusted_after_gain=({axis_adjustment.after.x},{axis_adjustment.after.y}) "
            f"smoothed=({smoothed.x},{smoothed.y}) "
            f"smoothing_state={_format_smoothing_state(self.smoother)} "
            f"cursor_mode={self.config.cursor_control_mode} "
            f"horizontal_gain={axis_adjustment.horizontal_gain:.2f} "
            f"vertical_gain={axis_adjustment.vertical_gain:.2f} "
            f"horizontal_offset={axis_adjustment.horizontal_offset:.1f} "
            f"vertical_offset={axis_adjustment.vertical_offset:.1f} "
            f"live_tracking_quality={live_quality.quality} "
            f"live_domain_violation_ratio={live_quality.domain_violation_ratio:.2f} "
            f"gaze_conf={gaze.confidence:.2f} "
            f"gaze_reason={gaze_reason} "
            f"blink={blink.combined_closed_probability:.2f} "
            f"click={click_name} "
            f"cursor_enabled={'yes' if self.config.cursor_enabled else 'no'} "
            f"cursor_skip={cursor_skip_reason} "
            f"face_found={'yes' if detection.found else 'no'} "
            f"calibration_quality={self.config.calibration_quality}"
        )
        if self.config.calibration_warnings:
            print("DEBUG calibration_warnings=" + " | ".join(self.config.calibration_warnings[:3]))

    def _handle_click_event(self, event: ClickEvent | None) -> str:
        if event is None or event is ClickEvent.NONE:
            return "none"
        if event is ClickEvent.LEFT_CLICK:
            self.cursor.left_click()
        elif event is ClickEvent.RIGHT_CLICK:
            self.cursor.right_click()
        elif event is ClickEvent.DOUBLE_CLICK:
            self.cursor.double_click()
        elif event is ClickEvent.PAUSE_TOGGLE:
            self._toggle_cursor_pause()
        return event.value


def _velocity_control_point(gaze: GazeResult, screen_width: int, screen_height: int) -> ScreenPoint:
    """Scale continuous normalized gaze into direction space for relative control."""
    raw_x = min(max(float(gaze.raw_x), 0.0), 1.0)
    raw_y = min(max(float(gaze.raw_y), 0.0), 1.0)
    return ScreenPoint(
        round(raw_x * max(screen_width - 1, 0)),
        round(raw_y * max(screen_height - 1, 0)),
    )

def build_realtime_pipeline(config: dict[str, Any]) -> RealtimeVisiMovePipeline:
    camera_config = CameraConfig(**config.get("camera", {}))
    performance_config = config.get("performance", {})
    pipeline_config = config.get("pipeline", {})
    cursor_config = config.get("cursor", {})
    blink_config = config.get("blink", {})
    detection_config = config.get("detection", {})
    gaze_config = config.get("gaze", {})
    smoothing_config = dict(config.get("smoothing", {}))
    calibration_config = config.get("calibration", {})

    dry_run = bool(pipeline_config.get("dry_run", True)) or not bool(cursor_config.get("enabled", False))
    screen_bounds = get_screen_bounds()
    smoothing_config.setdefault("screen_width", screen_bounds.width)
    smoothing_config.setdefault("screen_height", screen_bounds.height)

    safety = CursorSafety(
        bounds=screen_bounds,
        config=CursorSafetyConfig(
            min_gaze_confidence=float(cursor_config.get("min_gaze_confidence", 0.65)),
            max_speed_px_per_sec=float(cursor_config.get("max_speed_px_per_sec", 1400)),
            click_cooldown_ms=int(cursor_config.get("click_cooldown_ms", 500)),
            movement_duration_sec=float(cursor_config.get("movement_duration_sec", 0.03)),
            emergency_pause_key=str(cursor_config.get("emergency_pause_key", "p")),
        ),
        paused=bool(cursor_config.get("start_paused", True)),
    )

    backend = str(cursor_config.get("backend", "pyautogui")).lower()
    cursor: CursorController
    if dry_run:
        cursor = DryRunCursorController(safety=safety)
    elif backend == "win32":
        cursor = Win32CursorController(safety=safety)
    else:
        cursor = PyAutoGuiCursorController(
            safety=safety,
            movement_duration_sec=float(cursor_config.get("movement_duration_sec", 0.03)),
        )

    cursor_control_mode = str(cursor_config.get("control_mode", "absolute")).lower()
    if cursor_control_mode == "velocity":
        smoother = VelocityCursorFilter(
            screen_width=screen_bounds.width,
            screen_height=screen_bounds.height,
            initial_position=safety.current_position,
            deadzone=float(cursor_config.get("velocity_deadzone", 0.20)),
            horizontal_deadzone=float(cursor_config.get("velocity_horizontal_deadzone", 0.10)),
            vertical_deadzone=float(cursor_config.get("velocity_vertical_deadzone", 0.06)),
            max_speed_px_per_sec=float(cursor_config.get("velocity_max_speed_px_per_sec", 1400)),
            response_exponent=float(cursor_config.get("velocity_response_exponent", 1.35)),
            max_dt_seconds=float(cursor_config.get("velocity_max_dt_seconds", 0.10)),
            edge_margin_px=int(cursor_config.get("velocity_edge_margin_px", 8)),
            neutral_acquisition_seconds=float(cursor_config.get("velocity_neutral_acquisition_seconds", 1.0)),
            neutral_min_samples=int(cursor_config.get("velocity_neutral_min_samples", 8)),
            median_window=int(cursor_config.get("velocity_median_window", 5)),
        )
    elif cursor_control_mode == "absolute":
        smoother = build_smoothing_filter(smoothing_config)
    else:
        raise ValueError(f"Unsupported cursor control mode: {cursor_control_mode}")

    dwell_selector = None
    if bool(cursor_config.get("dwell_enabled", False)):
        dwell_selector = DwellSelector(
            dwell_time_ms=int(cursor_config.get("dwell_time_ms", 900)),
            radius_px=int(cursor_config.get("dwell_radius_px", 35)),
        )

    return RealtimeVisiMovePipeline(
        camera=WebcamCamera(
            config=camera_config,
            resize_width=performance_config.get("resize_width"),
        ),
        detector=build_detector(detection_config),
        gaze_model=build_gaze_model(build_gaze_backend_config(config)),
        blink_model=build_blink_model(blink_config),
        blink_state_machine=BlinkStateMachine(
            BlinkStateMachineConfig(
                closed_threshold=float(blink_config.get("closed_threshold", 0.65)),
                min_blink_ms=int(blink_config.get("min_blink_ms", 150)),
                max_blink_ms=int(blink_config.get("max_blink_ms", 600)),
                long_blink_ms=int(blink_config.get("long_blink_ms", 800)),
                double_blink_gap_ms=int(blink_config.get("double_blink_gap_ms", 450)),
                click_cooldown_ms=int(blink_config.get("click_cooldown_ms", 1000)),
                long_blink_event=ClickEvent(
                    str(blink_config.get("long_blink_event", "right_click"))
                ),
            )
        ),
        mapper=build_calibration_mapper(calibration_config),
        smoother=smoother,
        cursor=cursor,
        performance=PipelinePerformanceMonitor(),
        config=RealtimePipelineConfig(
            show_preview=bool(pipeline_config.get("show_preview", True)),
            dry_run=dry_run,
            report_interval_seconds=float(performance_config.get("log_interval_seconds", 1)),
            show_debug=bool(pipeline_config.get("show_debug", False)),
            debug_interval_seconds=float(pipeline_config.get("debug_interval_seconds", 0.5)),
            cursor_enabled=bool(cursor_config.get("enabled", False)) and not dry_run,
            calibration_quality=str(calibration_config.get("quality", "unknown")),
            calibration_warnings=tuple(str(warning) for warning in calibration_config.get("warnings", [])),
            allow_unstable_live_gaze=bool(pipeline_config.get("allow_unstable_live_gaze", False)),
            block_cursor_when_outside_calibration_domain=bool(
                calibration_config.get("block_cursor_when_outside_calibration_domain", True)
            ),
            live_domain_violation_ratio_threshold=float(
                calibration_config.get("live_domain_violation_ratio_threshold", 0.30)
            ),
            live_domain_window_size=int(calibration_config.get("live_domain_window_size", 30)),
            axis_adjustment=build_axis_adjustment_config(calibration_config),
            cursor_control_mode=cursor_control_mode,
        ),
        dwell_selector=dwell_selector,
    )


def build_axis_adjustment_config(config: dict[str, Any]) -> AxisAdjustmentConfig:
    return AxisAdjustmentConfig(
        horizontal_gain=float(config.get("horizontal_gain", 1.0)),
        vertical_gain=float(config.get("vertical_gain", 1.0)),
        horizontal_offset=float(config.get("horizontal_offset", 0.0)),
        vertical_offset=float(config.get("vertical_offset", 0.0)),
        enabled=bool(config.get("center_bias_correction", True)),
        edge_reach_enabled=bool(config.get("edge_reach_enabled", False)),
        edge_margin_px=int(config.get("edge_margin_px", 0)),
        edge_boost_enabled=bool(config.get("edge_boost_enabled", False)),
        edge_boost_gamma=float(config.get("edge_boost_gamma", 0.80)),
        source_min_x=_optional_float(config.get("edge_source_min_x")),
        source_max_x=_optional_float(config.get("edge_source_max_x")),
        source_min_y=_optional_float(config.get("edge_source_min_y")),
        source_max_y=_optional_float(config.get("edge_source_max_y")),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_calibration_mapper(config: dict[str, Any]) -> CalibrationMapper:
    profile_path = config.get("profile_path")
    if not profile_path:
        return CalibrationMapper()
    try:
        profile = load_calibration_profile(str(profile_path))
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        print(f"Calibration profile could not be loaded; using normalized fallback mapper. {exc}")
        return CalibrationMapper()
    try:
        mapper = CalibrationMapper.from_profile(profile)
        mapper.clamp_raw_input_to_calibration_domain = bool(
            config.get("clamp_raw_input_to_calibration_domain", True)
        )
        mapper.clamp_raw_input_to_mapping_domain = bool(
            config.get("clamp_raw_input_to_mapping_domain", True)
        )
        mapper.raw_domain_margin = float(config.get("raw_domain_margin", 0.05))
        return mapper
    except (KeyError, ValueError, RuntimeError) as exc:
        print(f"Calibration mapping parameters are invalid; using normalized fallback mapper. {exc}")
        return CalibrationMapper(screen_width=profile.screen_width, screen_height=profile.screen_height)


def _format_domain(mapping_debug: MappingDebugInfo) -> str:
    domain = mapping_debug.raw_domain
    if domain is None:
        return "unavailable"
    return (
        f"x[{domain.raw_x_min:.3f},{domain.raw_x_max:.3f}]"
        f",y[{domain.raw_y_min:.3f},{domain.raw_y_max:.3f}]"
    )


def _format_violations(violations: tuple[str, ...]) -> str:
    return "none" if not violations else ",".join(violations)


def _format_gaze_filter_debug(metadata: dict[str, Any]) -> str:
    unfiltered = metadata.get("unfiltered_raw_prediction")
    if unfiltered is None:
        return ""
    try:
        raw_x = float(unfiltered[0])
        raw_y = float(unfiltered[1])
        window = int(metadata.get("temporal_median_window", 1))
    except (TypeError, ValueError, IndexError):
        return ""
    return f"unfiltered_raw_gaze=({raw_x:.3f},{raw_y:.3f}) raw_median_window={window} "


def _format_gaze_native_debug(metadata: dict[str, Any]) -> str:
    native = metadata.get("native_prediction")
    if native is None:
        return ""
    try:
        native_x = float(native[0])
        native_y = float(native[1])
    except (TypeError, ValueError, IndexError):
        return ""
    units = str(metadata.get("native_prediction_units", "unknown"))
    return f"native_gaze=({native_x:.3f},{native_y:.3f}) native_units={units} "


def build_gaze_backend_config(config: dict[str, Any]) -> dict[str, Any]:
    gaze_config = dict(config.get("gaze", {}))
    backend = str(gaze_config.get("gaze_backend", gaze_config.get("backend", "dummy"))).lower()
    backend_config = dict(config.get(backend, {}))
    merged = {**backend_config, **gaze_config}
    if backend in {"eyetrax", "gazefollower", "mobilegaze"}:
        model_path = gaze_config.get("model_path")
        if model_path in {None, "", "null"} and backend_config.get("model_path"):
            merged["model_path"] = backend_config["model_path"]
    return merged


def build_smoothing_filter(config: dict[str, Any]) -> Any:
    filter_name = str(config.get("filter", "ema")).lower()
    max_jump_pixels = float(config.get("max_jump_pixels", 180))

    if filter_name in {"ema", "exponential"}:
        return EmaFilter(
            alpha=float(config.get("ema_alpha", config.get("alpha", 0.35))),
            max_jump_pixels=max_jump_pixels,
        )
    if filter_name == "deadzone":
        return DeadzoneFilter(
            radius=float(config.get("deadzone_radius", 8)),
            max_jump_pixels=max_jump_pixels,
        )
    if filter_name == "fixation":
        return FixationFilter(
            fixation_duration_ms=int(config.get("fixation_duration_ms", 180)),
            fixation_radius=float(config.get("fixation_radius", 18)),
            max_jump_pixels=max_jump_pixels,
        )
    if filter_name == "adaptive":
        return AdaptiveSmoothingFilter(
            fast_alpha=float(config.get("adaptive_fast_alpha", 0.38)),
            slow_alpha=float(config.get("adaptive_slow_alpha", 0.025)),
            fixation_radius=float(config.get("adaptive_fixation_radius", 90)),
            release_radius=float(config.get("adaptive_release_radius", 220)),
            fixation_hold_ms=int(config.get("adaptive_fixation_hold_ms", 180)),
            max_jump_pixels=max_jump_pixels,
            jump_confirm_radius=float(config.get("adaptive_jump_confirm_radius", 180)),
            jump_confirm_samples=int(config.get("adaptive_jump_confirm_samples", 2)),
            screen_width=_optional_float(config.get("screen_width")),
            screen_height=_optional_float(config.get("screen_height")),
            edge_snap_margin=float(config.get("adaptive_edge_snap_margin", 32)),
            edge_fast_alpha=float(config.get("adaptive_edge_fast_alpha", 0.65)),
        )
    if filter_name in {"one_euro", "1euro"}:
        return OneEuroFilter2D(
            min_cutoff=float(config.get("one_euro_min_cutoff", 0.35)),
            beta=float(config.get("one_euro_beta", 0.0001)),
            derivative_cutoff=float(config.get("one_euro_derivative_cutoff", 1.0)),
            max_dt_seconds=float(config.get("one_euro_max_dt_seconds", 0.20)),
            max_speed_px_per_sec=float(config.get("one_euro_max_speed_px_per_sec", 1600)),
            max_acceleration_px_per_sec2=float(
                config.get("one_euro_max_acceleration_px_per_sec2", 5000)
            ),
        )
    if filter_name == "kalman":
        return KalmanFilter2D(
            process_noise=float(config.get("kalman_process_noise", 0.01)),
            measurement_noise=float(config.get("kalman_measurement_noise", 4.0)),
            max_jump_pixels=max_jump_pixels,
        )
    raise ValueError(f"Unsupported smoothing filter: {filter_name}")


def _format_smoothing_state(smoother: Any) -> str:
    debug_state = getattr(smoother, "debug_state", None)
    if callable(debug_state):
        return str(debug_state())
    return "n/a"


def build_detector(config: dict[str, Any]) -> BaseDetector:
    backend = str(config.get("detector_backend", config.get("backend", "auto"))).lower()
    reuse_frames = int(config.get("reuse_frames", 2))

    if backend == "dummy":
        return DummyDetector()
    if backend == "opencv":
        return OpenCvHaarDetector(reuse_frames=reuse_frames)
    if backend == "auto":
        detector = _try_build_mediapipe_detector(config, reuse_frames, quiet=True)
        if detector is not None:
            return detector
        return OpenCvHaarDetector(reuse_frames=reuse_frames)
    if backend == "yolo":
        try:
            return YoloDetector(
                model_path=str(config.get("yolo_model_path", "")),
                confidence_threshold=float(config.get("yolo_confidence_threshold", 0.4)),
                image_size=int(config.get("yolo_image_size", 640)),
                device=str(config.get("yolo_device", "cpu")),
                detect_every_n_frames=int(config.get("yolo_detect_every_n_frames", 3)),
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"YOLO detector unavailable; falling back to MediaPipe/OpenCV. {exc}")
            detector = _try_build_mediapipe_detector(config, reuse_frames, quiet=False)
            return detector if detector is not None else OpenCvHaarDetector(reuse_frames=reuse_frames)
    if backend == "mediapipe":
        detector = _try_build_mediapipe_detector(config, reuse_frames, quiet=False)
        return detector if detector is not None else OpenCvHaarDetector(reuse_frames=reuse_frames)
    raise ValueError(f"Unsupported detector backend: {backend}")


def _try_build_mediapipe_detector(
    config: dict[str, Any],
    reuse_frames: int,
    quiet: bool,
) -> BaseDetector | None:
    try:
        return MediaPipeFaceMeshDetector(
            min_detection_confidence=float(config.get("min_face_confidence", 0.6)),
            min_tracking_confidence=float(config.get("min_tracking_confidence", 0.6)),
            reuse_frames=reuse_frames,
            face_landmarker_model_path=config.get("face_landmarker_model_path"),
        )
    except RuntimeError as exc:
        if not quiet:
            print(f"MediaPipe detector unavailable; falling back to OpenCV Haar detector. {exc}")
        return None


def build_blink_model(config: dict[str, Any]) -> BlinkModel:
    backend = str(config.get("backend", "dummy")).lower()
    if backend == "dummy":
        return KeyboardDummyBlinkModel()

    if backend == "onnx":
        model_path = config.get("model_path")
        if not model_path:
            print("Blink ONNX model path is not configured; falling back to dummy blink model.")
            return KeyboardDummyBlinkModel()
        try:
            return OnnxBlinkModel(
                OnnxBlinkModelConfig(
                    model_path=str(model_path),
                    input_width=int(config.get("input_width", 64)),
                    input_height=int(config.get("input_height", 64)),
                    normalize_mean=float(config.get("normalize_mean", 0.5)),
                    normalize_std=float(config.get("normalize_std", 0.5)),
                )
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Blink ONNX model unavailable; falling back to dummy blink model. {exc}")
            return KeyboardDummyBlinkModel()

    if backend == "ocec":
        try:
            return OcecBlinkAdapter()
        except RuntimeError as exc:
            print(f"OCEC blink backend unavailable; falling back to dummy blink model. {exc}")
            return KeyboardDummyBlinkModel()

    raise ValueError(f"Unsupported blink backend: {backend}")


def build_gaze_model(config: dict[str, Any]) -> BaseGazeModel:
    backend = str(config.get("gaze_backend", config.get("backend", "dummy"))).lower()
    model_path = config.get("model_path")
    fallback_to_dummy = bool(config.get("fallback_to_dummy", True))
    if backend == "dummy":
        return MovingDummyGazeModel()

    if backend == "eyetrax":
        try:
            model = EyeTraxAdapter.from_config(config)
            print("EyeTrax backend selected and preflight checks passed.")
            return model
        except (GazeBackendUnavailable, RuntimeError) as exc:
            print(f"EyeTrax backend unavailable. {exc}")
            if fallback_to_dummy:
                print("Falling back to dummy gaze model.")
                return MovingDummyGazeModel(
                    metadata={
                        "fallback": True,
                        "requested_backend": backend,
                        "unavailable_reason": str(exc),
                    }
                )
            raise

    if backend == "gazefollower":
        try:
            model = GazeFollowerAdapter.from_config(config)
            print("GazeFollower backend selected and preflight checks passed.")
            return model
        except (GazeBackendUnavailable, RuntimeError) as exc:
            print(f"GazeFollower backend unavailable. {exc}")
            if fallback_to_dummy:
                print("Falling back to dummy gaze model.")
                return MovingDummyGazeModel(
                    metadata={
                        "fallback": True,
                        "requested_backend": backend,
                        "unavailable_reason": str(exc),
                    }
            )
            raise

    if backend == "mobilegaze":
        try:
            model = MobileGazeAdapter.from_config(config)
            print("MobileGaze backend selected and preflight checks passed.")
            return model
        except (GazeBackendUnavailable, RuntimeError) as exc:
            print(f"MobileGaze backend unavailable. {exc}")
            if fallback_to_dummy:
                print("Falling back to dummy gaze model.")
                return MovingDummyGazeModel(
                    metadata={
                        "fallback": True,
                        "requested_backend": backend,
                        "unavailable_reason": str(exc),
                    }
                )
            raise

    adapter_map: dict[str, type[BaseGazeModel]] = {
    }
    adapter_class = adapter_map.get(backend)
    if adapter_class is None:
        raise ValueError(f"Unsupported gaze backend: {backend}")

    try:
        return adapter_class(model_path=str(model_path) if model_path else None)  # type: ignore[call-arg]
    except (GazeBackendUnavailable, RuntimeError) as exc:
        print(f"{backend} gaze backend unavailable. {exc}")
        if fallback_to_dummy:
            print("Falling back to dummy gaze model.")
            return MovingDummyGazeModel(
                metadata={
                    "fallback": True,
                    "requested_backend": backend,
                    "unavailable_reason": str(exc),
                }
            )
        raise
