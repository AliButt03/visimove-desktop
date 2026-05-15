from visimove.smoothing import DeadzoneFilter, EmaFilter, ExponentialSmoothingFilter
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
