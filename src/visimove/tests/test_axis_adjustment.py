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
    assert result.after_edge_reach == result.before
    assert isinstance(result.adjusted_x_float, float)
    assert isinstance(result.adjusted_y_float, float)


def test_edge_reach_expands_calibrated_area_to_screen_edges() -> None:
    result = apply_axis_adjustment(
        ScreenPoint(x=307, y=173),
        screen_width=2560,
        screen_height=1440,
        config=AxisAdjustmentConfig(
            edge_reach_enabled=True,
            source_min_x=307,
            source_max_x=2252,
            source_min_y=173,
            source_max_y=1266,
        ),
    )

    assert result.after_edge_reach == ScreenPoint(x=0, y=0)
    assert result.after == ScreenPoint(x=0, y=0)


def test_edge_reach_respects_margin() -> None:
    result = apply_axis_adjustment(
        ScreenPoint(x=2252, y=1266),
        screen_width=2560,
        screen_height=1440,
        config=AxisAdjustmentConfig(
            edge_reach_enabled=True,
            edge_margin_px=20,
            source_min_x=307,
            source_max_x=2252,
            source_min_y=173,
            source_max_y=1266,
        ),
    )

    assert result.after_edge_reach == ScreenPoint(x=2539, y=1419)


def test_center_bias_offsets_do_not_pull_exact_edge_targets_inward() -> None:
    result = apply_axis_adjustment(
        ScreenPoint(x=2252, y=173),
        screen_width=2560,
        screen_height=1440,
        config=AxisAdjustmentConfig(
            horizontal_offset=-44,
            vertical_offset=180,
            edge_reach_enabled=True,
            source_min_x=307,
            source_max_x=2252,
            source_min_y=173,
            source_max_y=1266,
        ),
    )

    assert result.after_edge_reach == ScreenPoint(x=2559, y=0)
    assert result.after == ScreenPoint(x=2559, y=0)


def test_edge_boost_pushes_points_toward_edges() -> None:
    result = apply_axis_adjustment(
        ScreenPoint(x=800, y=200),
        screen_width=1000,
        screen_height=800,
        config=AxisAdjustmentConfig(edge_boost_enabled=True, edge_boost_gamma=0.75),
    )

    assert result.after_edge_reach.x > 800
    assert result.after_edge_reach.y < 200
