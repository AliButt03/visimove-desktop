from types import SimpleNamespace

from visimove.calibration import LiveTrackingQualityState
from visimove.pipeline.realtime_pipeline import RealtimePipelineConfig, RealtimeVisiMovePipeline


def make_pipeline(allow_unstable: bool = False) -> RealtimeVisiMovePipeline:
    pipeline = object.__new__(RealtimeVisiMovePipeline)
    pipeline.config = RealtimePipelineConfig(
        dry_run=False,
        cursor_enabled=True,
        allow_unstable_live_gaze=allow_unstable,
        block_cursor_when_outside_calibration_domain=True,
    )
    pipeline.cursor = SimpleNamespace(
        safety=SimpleNamespace(
            paused=False,
            config=SimpleNamespace(min_gaze_confidence=0.5),
        )
    )
    return pipeline


def test_cursor_blocked_when_live_tracking_quality_is_unsafe() -> None:
    pipeline = make_pipeline()
    state = LiveTrackingQualityState(
        quality="unsafe",
        domain_violation_ratio=0.5,
        clipped_output_ratio=0.0,
        sample_count=30,
    )

    reason = pipeline._cursor_skip_reason(1.0, True, state)

    assert reason == "live gaze outside calibrated domain"


def test_cursor_allowed_when_live_tracking_quality_is_stable() -> None:
    pipeline = make_pipeline()
    state = LiveTrackingQualityState(
        quality="stable",
        domain_violation_ratio=0.0,
        clipped_output_ratio=0.0,
        sample_count=30,
    )

    reason = pipeline._cursor_skip_reason(1.0, True, state)

    assert reason == "none"


def test_cursor_allowed_when_live_tracking_quality_is_only_unstable() -> None:
    pipeline = make_pipeline()
    state = LiveTrackingQualityState(
        quality="unstable",
        domain_violation_ratio=0.03,
        clipped_output_ratio=0.0,
        sample_count=30,
    )

    reason = pipeline._cursor_skip_reason(1.0, True, state)

    assert reason == "none"


def test_unstable_live_gaze_override_allows_cursor() -> None:
    pipeline = make_pipeline(allow_unstable=True)
    state = LiveTrackingQualityState(
        quality="unsafe",
        domain_violation_ratio=0.8,
        clipped_output_ratio=0.0,
        sample_count=30,
    )

    reason = pipeline._cursor_skip_reason(1.0, True, state)

    assert reason == "none"


def test_velocity_dwell_only_runs_when_cursor_is_holding() -> None:
    pipeline = make_pipeline()
    pipeline.config = RealtimePipelineConfig(
        dry_run=False,
        cursor_enabled=True,
        cursor_control_mode="velocity",
    )
    pipeline.smoother = SimpleNamespace(debug_state=lambda: "velocity_moving")
    assert pipeline._dwell_allowed() is False

    pipeline.smoother = SimpleNamespace(debug_state=lambda: "velocity_hold")
    assert pipeline._dwell_allowed() is True


def test_absolute_dwell_uses_existing_radius_behavior() -> None:
    pipeline = make_pipeline()
    pipeline.config = RealtimePipelineConfig(
        dry_run=False,
        cursor_enabled=True,
        cursor_control_mode="absolute",
    )
    pipeline.smoother = SimpleNamespace(debug_state=lambda: "moving")

    assert pipeline._dwell_allowed() is True

def test_pipeline_stores_dwell_selector() -> None:
    dwell = SimpleNamespace()
    pipeline = RealtimeVisiMovePipeline(
        camera=None,  # type: ignore[arg-type]
        detector=None,  # type: ignore[arg-type]
        gaze_model=None,  # type: ignore[arg-type]
        blink_model=None,  # type: ignore[arg-type]
        blink_state_machine=None,  # type: ignore[arg-type]
        mapper=None,  # type: ignore[arg-type]
        smoother=SimpleNamespace(),
        cursor=SimpleNamespace(),
        performance=None,  # type: ignore[arg-type]
        config=RealtimePipelineConfig(),
        dwell_selector=dwell,  # type: ignore[arg-type]
    )

    assert pipeline.dwell_selector is dwell

def test_debug_print_uses_supplied_click_name(capsys) -> None:
    from visimove.blink import BlinkResult
    from visimove.calibration import AxisAdjustmentConfig, MappingDebugInfo, apply_axis_adjustment
    from visimove.gaze import GazeResult
    from visimove.types import DetectionResult, ScreenPoint
    pipeline = make_pipeline()
    pipeline.config = RealtimePipelineConfig(
        dry_run=False,
        cursor_enabled=True,
        cursor_control_mode="velocity",
    )
    pipeline.smoother = SimpleNamespace(debug_state=lambda: "velocity_hold")
    point = ScreenPoint(500, 400)
    mapping_debug = MappingDebugInfo(
        raw_x=0.5,
        raw_y=0.5,
        mapped_input_x=0.5,
        mapped_input_y=0.5,
        before_clamp_x=500.0,
        before_clamp_y=400.0,
        after_clamp_x=500,
        after_clamp_y=400,
        clipped_x=False,
        clipped_y=False,
    )

    pipeline._print_debug(
        detection=DetectionResult(found=True),
        gaze=GazeResult(raw_x=0.5, raw_y=0.5, confidence=0.75),
        blink=BlinkResult(0.05, 0.05, 0.05, 0.95),
        mapped=point,
        axis_adjustment=apply_axis_adjustment(
            point,
            screen_width=1000,
            screen_height=800,
            config=AxisAdjustmentConfig(),
        ),
        mapping_debug=mapping_debug,
        live_quality=LiveTrackingQualityState("stable", 0.0, 0.0, 1),
        smoothed=point,
        click_name="dwell_left_click",
        cursor_skip_reason="none",
    )

    output = capsys.readouterr().out
    assert "click=dwell_left_click" in output
    assert "cursor_mode=velocity" in output
