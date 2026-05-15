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
