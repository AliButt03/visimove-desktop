from visimove.calibration import LiveTrackingQualityMonitor, MappingDebugInfo


def make_debug(outside: bool = False, clipped: bool = False) -> MappingDebugInfo:
    return MappingDebugInfo(
        raw_x=0.5,
        raw_y=0.5,
        mapped_input_x=0.5,
        mapped_input_y=0.5,
        before_clamp_x=500.0,
        before_clamp_y=-10.0 if clipped else 500.0,
        after_clamp_x=500,
        after_clamp_y=0 if clipped else 500,
        clipped_x=False,
        clipped_y=clipped,
        raw_domain_status="outside" if outside else "inside",
        raw_domain_violations=("raw_y_below_min",) if outside else (),
    )


def test_live_domain_violation_ratio_is_calculated() -> None:
    monitor = LiveTrackingQualityMonitor(window_size=10, unsafe_ratio_threshold=0.30)
    for _ in range(7):
        monitor.update(make_debug())
    state = monitor.update(make_debug(outside=True))

    assert state.quality == "unstable"
    assert state.domain_violation_ratio == 0.125


def test_live_quality_becomes_unsafe_when_violations_cross_threshold() -> None:
    monitor = LiveTrackingQualityMonitor(window_size=10, unsafe_ratio_threshold=0.30)
    for _ in range(7):
        monitor.update(make_debug())
    for _ in range(3):
        state = monitor.update(make_debug(outside=True))

    assert state.quality == "unsafe"
    assert state.domain_violation_ratio == 0.3
