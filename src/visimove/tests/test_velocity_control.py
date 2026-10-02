from visimove.cursor.velocity_control import VelocityCursorFilter


def make_filter() -> VelocityCursorFilter:
    return VelocityCursorFilter(
        screen_width=1000,
        screen_height=800,
        initial_position=(500, 400),
        deadzone=0.20,
        max_speed_px_per_sec=1000,
        response_exponent=1.0,
        max_dt_seconds=0.10,
        edge_margin_px=8,
        neutral_acquisition_seconds=0.2,
        neutral_min_samples=3,
        median_window=3,
    )


def calibrate(controller: VelocityCursorFilter, x: float = 500, y: float = 400) -> None:
    controller.update(x, y, 1.0)
    controller.update(x, y, 1.1)
    controller.update(x, y, 1.2)


def test_off_center_neutral_gaze_holds_cursor_still() -> None:
    controller = make_filter()

    controller.update(300, 600, 1.0)
    controller.update(305, 595, 1.1)
    controller.update(295, 605, 1.2)

    assert controller.neutral_point() == (300.0, 600.0)
    assert controller.update(310, 590, 1.3) == (500.0, 400.0)
    assert controller.debug_state() == "velocity_hold"


def test_right_gaze_moves_cursor_incrementally() -> None:
    controller = make_filter()
    calibrate(controller, 300, 600)

    controller.update(900, 600, 1.3)
    controller.update(900, 600, 1.4)
    moved = controller.update(900, 600, 1.5)

    assert moved is not None
    assert 500.0 < moved[0] <= 700.0
    assert moved[1] == 400.0
    assert controller.debug_state() == "velocity_moving"


def test_vertical_deadzone_can_be_more_sensitive_than_general_deadzone() -> None:
    controller = VelocityCursorFilter(
        screen_width=1000,
        screen_height=800,
        initial_position=(500, 400),
        deadzone=0.20,
        horizontal_deadzone=0.10,
        vertical_deadzone=0.05,
        max_speed_px_per_sec=1000,
        response_exponent=1.0,
        max_dt_seconds=0.10,
        edge_margin_px=8,
        neutral_acquisition_seconds=0.2,
        neutral_min_samples=3,
        median_window=3,
    )
    calibrate(controller)

    controller.update(500, 360, 1.3)
    controller.update(500, 360, 1.4)
    moved = controller.update(500, 360, 1.5)

    assert moved is not None
    assert moved[0] == 500.0
    assert moved[1] < 400.0
    assert controller.debug_state() == "velocity_moving"

def test_long_frame_gap_cannot_create_a_teleport() -> None:
    controller = make_filter()
    calibrate(controller)
    first = controller.update(999, 799, 9.8)
    second = controller.update(999, 799, 9.9)
    moved = controller.update(999, 799, 10.0)

    assert first is not None and second is not None and moved is not None
    assert second[0] - first[0] <= 100.0
    assert moved[0] - second[0] <= 100.0
    assert second[1] - first[1] <= 100.0
    assert moved[1] - second[1] <= 100.0


def test_velocity_output_stays_inside_safe_screen_margin() -> None:
    controller = VelocityCursorFilter(
        screen_width=1000,
        screen_height=800,
        initial_position=(990, 790),
        deadzone=0.0,
        max_speed_px_per_sec=1000,
        response_exponent=1.0,
        max_dt_seconds=0.10,
        edge_margin_px=8,
        neutral_acquisition_seconds=0.2,
        neutral_min_samples=3,
        median_window=3,
    )
    calibrate(controller)
    controller.update(999, 799, 1.3)
    controller.update(999, 799, 1.4)

    assert controller.update(999, 799, 1.5) == (991.0, 791.0)


def test_missing_gaze_holds_last_cursor_position() -> None:
    controller = make_filter()

    controller.update(500, 400, 1.0)
    assert controller.update(None, None, 1.1) == (500.0, 400.0)
    assert controller.debug_state() == "velocity_missing"


def test_set_position_resynchronizes_after_pause() -> None:
    controller = make_filter()
    calibrate(controller)

    controller.set_position((250, 300))

    assert controller.update(500, 400, 2.0) == (250.0, 300.0)
    assert controller.debug_state() == "velocity_hold"


def test_isolated_gaze_spike_is_rejected_by_median_window() -> None:
    controller = make_filter()
    calibrate(controller)

    before = controller.update(500, 400, 1.3)
    after = controller.update(999, 799, 1.4)

    assert after == before
    assert controller.debug_state() == "velocity_hold"


def test_neutral_calibration_can_be_restarted() -> None:
    controller = make_filter()
    calibrate(controller, 300, 600)
    assert controller.neutral_point() == (300.0, 600.0)

    controller.begin_neutral_calibration()
    controller.update(700, 200, 2.0)
    controller.update(700, 200, 2.1)
    controller.update(700, 200, 2.2)

    assert controller.neutral_point() == (700.0, 200.0)
    assert controller.debug_state() == "velocity_hold"


def test_pipeline_builder_selects_velocity_mode_and_dwell() -> None:
    from visimove.pipeline.realtime_pipeline import build_realtime_pipeline

    pipeline = build_realtime_pipeline(
        {
            "camera": {},
            "detection": {"detector_backend": "dummy"},
            "gaze": {"gaze_backend": "dummy"},
            "blink": {"backend": "dummy"},
            "calibration": {},
            "cursor": {
                "enabled": False,
                "control_mode": "velocity",
                "dwell_enabled": True,
            },
            "pipeline": {"dry_run": True},
        }
    )

    assert isinstance(pipeline.smoother, VelocityCursorFilter)
    assert pipeline.config.cursor_control_mode == "velocity"
    assert pipeline.dwell_selector is not None
def test_velocity_direction_uses_continuous_raw_gaze_not_grid_mapping() -> None:
    from visimove.gaze import GazeResult
    from visimove.pipeline.realtime_pipeline import _velocity_control_point

    upper = _velocity_control_point(GazeResult(raw_x=0.5, raw_y=0.2), 1000, 800)
    lower = _velocity_control_point(GazeResult(raw_x=0.5, raw_y=0.8), 1000, 800)
    clamped = _velocity_control_point(GazeResult(raw_x=-0.5, raw_y=1.5), 1000, 800)

    assert upper.y < lower.y
    assert (upper.x, upper.y) == (500, 160)
    assert (lower.x, lower.y) == (500, 639)
    assert (clamped.x, clamped.y) == (0, 799)
