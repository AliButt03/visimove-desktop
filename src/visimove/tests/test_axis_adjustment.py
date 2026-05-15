from visimove.calibration import AxisAdjustmentConfig, apply_axis_adjustment
from visimove.types import ScreenPoint


def test_horizontal_gain_expands_x_around_screen_center() -> None:
    result = apply_axis_adjustment(
        ScreenPoint(x=700, y=500),
        screen_width=1000,
        screen_height=800,
        config=AxisAdjustmentConfig(horizontal_gain=1.5),
    )

    assert result.after.x > 700
    assert result.after.y == 500
    assert result.horizontal_gain == 1.5


def test_offset_shifts_x_and_y() -> None:
    result = apply_axis_adjustment(
        ScreenPoint(x=500, y=400),
        screen_width=1000,
        screen_height=800,
        config=AxisAdjustmentConfig(horizontal_offset=-50, vertical_offset=25),
    )

    assert result.after.x == 450
    assert result.after.y == 425


def test_gain_output_is_clamped_to_screen_bounds() -> None:
    result = apply_axis_adjustment(
        ScreenPoint(x=990, y=790),
        screen_width=1000,
        screen_height=800,
        config=AxisAdjustmentConfig(horizontal_gain=2.0, vertical_gain=2.0),
    )

    assert result.after.x == 999
    assert result.after.y == 799


def test_adjustment_result_populates_debug_fields() -> None:
    result = apply_axis_adjustment(
        ScreenPoint(x=250, y=200),
        screen_width=1000,
        screen_height=800,
        config=AxisAdjustmentConfig(horizontal_gain=1.2, vertical_gain=1.1),
    )

    assert result.before == ScreenPoint(x=250, y=200)
    assert result.after != result.before
    assert isinstance(result.adjusted_x_float, float)
    assert isinstance(result.adjusted_y_float, float)
