from visimove.smoothing import AdaptiveSmoothingFilter, DeadzoneFilter, EmaFilter, ExponentialSmoothingFilter, OneEuroFilter2D
from visimove.types import ScreenPoint


def test_exponential_smoothing_starts_at_first_point() -> None:
    smoother = ExponentialSmoothingFilter(alpha=0.5)
    assert smoother.update(ScreenPoint(10, 20)) == ScreenPoint(10, 20)


def test_exponential_smoothing_blends_points() -> None:
    smoother = ExponentialSmoothingFilter(alpha=0.5)
    smoother.update(ScreenPoint(0, 0))
    assert smoother.update(ScreenPoint(10, 20)) == ScreenPoint(5, 10)


def test_ema_smoothing_blends_points() -> None:
    smoother = EmaFilter(alpha=0.5)

    assert smoother.update(0, 0, 1.0) == (0.0, 0.0)
    assert smoother.update(10, 20, 1.1) == (5.0, 10.0)


def test_deadzone_holds_small_movements() -> None:
    smoother = DeadzoneFilter(radius=10)

    assert smoother.update(100, 100, 1.0) == (100.0, 100.0)
    assert smoother.update(105, 104, 1.1) == (100.0, 100.0)
    assert smoother.update(120, 100, 1.2) == (120.0, 100.0)


def test_large_jump_is_clamped() -> None:
    smoother = EmaFilter(alpha=1.0, max_jump_pixels=10)

    assert smoother.update(0, 0, 1.0) == (0.0, 0.0)
    assert smoother.update(100, 0, 1.1) == (10.0, 0.0)


def test_missing_gaze_reuses_last_value() -> None:
    smoother = EmaFilter(alpha=0.5)

    assert smoother.update(4, 8, 1.0) == (4.0, 8.0)
    assert smoother.update(None, None, 1.1) == (4.0, 8.0)


def test_reset_clears_filter_state() -> None:
    smoother = EmaFilter(alpha=0.5)

    smoother.update(0, 0, 1.0)
    smoother.update(10, 10, 1.1)
    smoother.reset()

    assert smoother.update(100, 100, 2.0) == (100.0, 100.0)


def test_adaptive_filter_moves_fast_for_large_intentional_motion() -> None:
    smoother = AdaptiveSmoothingFilter(
        fast_alpha=0.5,
        slow_alpha=0.1,
        release_radius=50,
        jump_confirm_samples=1,
    )

    assert smoother.update(0, 0, 1.0) == (0.0, 0.0)
    assert smoother.update(100, 0, 1.1) == (50.0, 0.0)
    assert smoother.debug_state() == "moving"


def test_adaptive_filter_releases_when_smoothed_cursor_is_far_from_target() -> None:
    smoother = AdaptiveSmoothingFilter(
        fast_alpha=0.5,
        slow_alpha=0.05,
        fixation_radius=40,
        release_radius=100,
        jump_confirm_samples=1,
    )

    assert smoother.update(1600, 800, 1.0) == (1600.0, 800.0)
    smoother._anchor = (0.0, 800.0)

    assert smoother.update(0, 800, 1.1) == (800.0, 800.0)
    assert smoother.debug_state() == "moving"


def test_adaptive_filter_holds_small_fixation_jitter() -> None:
    smoother = AdaptiveSmoothingFilter(
        fast_alpha=0.5,
        slow_alpha=0.1,
        fixation_radius=20,
        fixation_hold_ms=100,
    )

    assert smoother.update(100, 100, 1.0) == (100.0, 100.0)
    assert smoother.update(108, 104, 1.2) == (100.0, 100.0)
    assert smoother.debug_state() == "hold"


def test_adaptive_filter_holds_jitter_around_fixation_anchor() -> None:
    smoother = AdaptiveSmoothingFilter(
        fast_alpha=0.5,
        slow_alpha=0.1,
        fixation_radius=40,
        release_radius=120,
        fixation_hold_ms=100,
    )

    assert smoother.update(100, 100, 1.0) == (100.0, 100.0)
    assert smoother.update(130, 100, 1.2) == (100.0, 100.0)
    assert smoother.debug_state() == "hold"


def test_adaptive_filter_holds_last_position_when_gaze_missing() -> None:
    smoother = AdaptiveSmoothingFilter()

    assert smoother.update(100, 100, 1.0) == (100.0, 100.0)
    assert smoother.update(None, None, 1.1) == (100.0, 100.0)
    assert smoother.debug_state() == "missing"


def test_adaptive_filter_confirms_large_jump_before_moving() -> None:
    smoother = AdaptiveSmoothingFilter(
        fast_alpha=0.5,
        slow_alpha=0.1,
        release_radius=50,
        jump_confirm_radius=30,
        jump_confirm_samples=2,
    )

    assert smoother.update(0, 0, 1.0) == (0.0, 0.0)
    assert smoother.update(100, 0, 1.1) == (0.0, 0.0)
    assert smoother.debug_state() == "confirming"
    assert smoother.update(108, 5, 1.2) == (54.0, 2.5)
    assert smoother.debug_state() == "moving"


def test_adaptive_filter_rejects_isolated_large_jump() -> None:
    smoother = AdaptiveSmoothingFilter(
        fast_alpha=0.5,
        slow_alpha=0.1,
        release_radius=50,
        jump_confirm_radius=30,
        jump_confirm_samples=2,
    )

    assert smoother.update(200, 200, 1.0) == (200.0, 200.0)
    assert smoother.update(500, 500, 1.1) == (200.0, 200.0)
    assert smoother.debug_state() == "confirming"
    assert smoother.update(205, 203, 1.2) == (200.0, 200.0)
    assert smoother.debug_state() == "hold"


def test_adaptive_filter_catches_up_to_confirmed_edge_target() -> None:
    smoother = AdaptiveSmoothingFilter(
        fast_alpha=0.38,
        slow_alpha=0.025,
        release_radius=220,
        jump_confirm_radius=180,
        jump_confirm_samples=2,
        screen_width=2560,
        screen_height=1440,
        edge_snap_margin=32,
        edge_fast_alpha=0.65,
    )

    assert smoother.update(500, 300, 1.0) == (500.0, 300.0)
    assert smoother.update(0, 0, 1.1) == (500.0, 300.0)
    assert smoother.debug_state() == "confirming"
    edge_step = smoother.update(0, 0, 1.2)

    assert edge_step is not None
    assert edge_step[0] < 200
    assert edge_step[1] < 120
    assert smoother.debug_state() == "edge"


def test_adaptive_filter_snaps_when_already_close_to_edge_target() -> None:
    smoother = AdaptiveSmoothingFilter(
        screen_width=2560,
        screen_height=1440,
        edge_snap_margin=32,
    )

    assert smoother.update(20, 18, 1.0) == (20.0, 18.0)
    assert smoother.update(0, 0, 1.1) == (0.0, 0.0)
    assert smoother.debug_state() == "edge_hold"

def test_one_euro_filter_smooths_absolute_gaze_without_freezing() -> None:
    smoother = OneEuroFilter2D(min_cutoff=0.35, beta=0.0004)

    assert smoother.update(100, 100, 1.0) == (100.0, 100.0)
    moved = smoother.update(500, 300, 1.1)

    assert moved is not None
    assert 100.0 < moved[0] < 500.0
    assert 100.0 < moved[1] < 300.0
    assert smoother.debug_state() in {"one_euro_tracking", "one_euro_bounded"}


def test_one_euro_filter_holds_last_position_when_gaze_is_missing() -> None:
    smoother = OneEuroFilter2D()

    assert smoother.update(120, 240, 1.0) == (120.0, 240.0)
    assert smoother.update(None, None, 1.1) == (120.0, 240.0)
    assert smoother.debug_state() == "one_euro_missing"


def test_one_euro_filter_responds_faster_to_high_speed_input_when_beta_increases() -> None:
    fixed = OneEuroFilter2D(
        min_cutoff=0.35,
        beta=0.0,
        max_speed_px_per_sec=100_000.0,
        max_acceleration_px_per_sec2=1_000_000.0,
    )
    adaptive = OneEuroFilter2D(
        min_cutoff=0.35,
        beta=0.002,
        max_speed_px_per_sec=100_000.0,
        max_acceleration_px_per_sec2=1_000_000.0,
    )
    fixed.update(0, 0, 1.0)
    adaptive.update(0, 0, 1.0)

    fixed_step = fixed.update(1000, 0, 1.1)
    adaptive_step = adaptive.update(1000, 0, 1.1)

    assert fixed_step is not None and adaptive_step is not None
    assert adaptive_step[0] > fixed_step[0]


def test_one_euro_filter_reset_clears_previous_target() -> None:
    smoother = OneEuroFilter2D()
    smoother.update(100, 100, 1.0)
    smoother.update(500, 500, 1.1)

    smoother.reset()

    assert smoother.update(900, 700, 2.0) == (900.0, 700.0)

def test_one_euro_filter_limits_large_output_step_by_speed() -> None:
    smoother = OneEuroFilter2D(
        min_cutoff=0.35,
        beta=0.0001,
        max_speed_px_per_sec=100.0,
        max_acceleration_px_per_sec2=10_000.0,
    )

    assert smoother.update(0, 0, 1.0) == (0.0, 0.0)
    moved = smoother.update(1000, 0, 1.1)

    assert moved is not None
    assert 0.0 < moved[0] <= 10.1
    assert moved[1] == 0.0


def test_one_euro_filter_limits_direction_reversal_by_acceleration() -> None:
    smoother = OneEuroFilter2D(
        min_cutoff=100.0,
        beta=0.0,
        max_speed_px_per_sec=1000.0,
        max_acceleration_px_per_sec2=100.0,
    )

    smoother.update(0, 0, 1.0)
    first = smoother.update(1000, 0, 1.1)
    reversed_step = smoother.update(-1000, 0, 1.2)

    assert first is not None and reversed_step is not None
    assert first[0] > 0.0
    assert reversed_step[0] >= first[0]


def test_one_euro_filter_eventually_converges_to_stationary_target() -> None:
    smoother = OneEuroFilter2D(
        min_cutoff=1.0,
        beta=0.0001,
        max_speed_px_per_sec=500.0,
        max_acceleration_px_per_sec2=2000.0,
    )
    smoother.update(0, 0, 0.0)

    position = None
    for step in range(1, 101):
        position = smoother.update(500, 300, step * 0.05)

    assert position is not None
    assert abs(position[0] - 500.0) < 1.0
    assert abs(position[1] - 300.0) < 1.0

def test_one_euro_filter_can_resynchronize_to_cursor_position() -> None:
    smoother = OneEuroFilter2D()
    smoother.update(100, 100, 1.0)
    smoother.update(900, 700, 1.1)

    smoother.set_position((400, 300))

    synchronized = smoother.update(400, 300, 1.2)
    assert synchronized is not None
    assert abs(synchronized[0] - 400.0) < 0.001
    assert abs(synchronized[1] - 300.0) < 0.001
    assert smoother.debug_state() == "one_euro_tracking"
