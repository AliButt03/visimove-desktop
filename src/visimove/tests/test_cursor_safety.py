from visimove.cursor import CursorSafety, CursorSafetyConfig, DryRunCursorController
from visimove.utils.screen import ScreenBounds


def test_screen_bounds_clamps_coordinates() -> None:
    bounds = ScreenBounds(width=100, height=50)

    assert bounds.clamp(-25, 200) == (0, 49)
    assert bounds.clamp(120, -3) == (99, 0)


def test_paused_controller_ignores_movement_and_clicks() -> None:
    safety = CursorSafety(bounds=ScreenBounds(width=100, height=100), paused=True)
    controller = DryRunCursorController(safety=safety)

    controller.move_to(50, 50)
    controller.left_click()

    assert controller.last_position is None
    assert controller.left_click_count == 0


def test_resumed_controller_clamps_movement() -> None:
    safety = CursorSafety(bounds=ScreenBounds(width=100, height=50), paused=False)
    controller = DryRunCursorController(safety=safety)

    controller.move_to(250, -10)

    assert controller.last_position == (99, 0)


def test_low_gaze_confidence_freezes_cursor() -> None:
    safety = CursorSafety(
        bounds=ScreenBounds(width=100, height=100),
        config=CursorSafetyConfig(min_gaze_confidence=0.7),
        paused=False,
    )
    controller = DryRunCursorController(safety=safety)

    controller.move_to(50, 50, gaze_confidence=0.2, target_visible=True)

    assert controller.last_position is None


def test_click_cooldown_blocks_immediate_second_click() -> None:
    safety = CursorSafety(
        bounds=ScreenBounds(width=100, height=100),
        config=CursorSafetyConfig(click_cooldown_ms=10_000),
        paused=False,
    )
    controller = DryRunCursorController(safety=safety)

    controller.left_click()
    controller.left_click()

    assert controller.left_click_count == 1

