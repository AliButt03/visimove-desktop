from __future__ import annotations

from dataclasses import dataclass

from visimove.types import ScreenPoint


@dataclass(frozen=True)
class AxisAdjustmentConfig:
    horizontal_gain: float = 1.0
    vertical_gain: float = 1.0
    horizontal_offset: float = 0.0
    vertical_offset: float = 0.0
    enabled: bool = True
    edge_reach_enabled: bool = False
    edge_margin_px: int = 0
    edge_boost_enabled: bool = False
    edge_boost_gamma: float = 0.80
    source_min_x: float | None = None
    source_max_x: float | None = None
    source_min_y: float | None = None
    source_max_y: float | None = None


@dataclass(frozen=True)
class AxisAdjustmentResult:
    before: ScreenPoint
    after_edge_reach: ScreenPoint
    after: ScreenPoint
    edge_reach_x_float: float
    edge_reach_y_float: float
    adjusted_x_float: float
    adjusted_y_float: float
    horizontal_gain: float
    vertical_gain: float
    horizontal_offset: float
    vertical_offset: float
    edge_reach_enabled: bool
    edge_margin_px: int
    edge_boost_enabled: bool
    edge_boost_gamma: float


def apply_axis_adjustment(
    point: ScreenPoint,
    screen_width: int,
    screen_height: int,
    config: AxisAdjustmentConfig,
) -> AxisAdjustmentResult:
    edge_x, edge_y = _apply_edge_reach(point, screen_width, screen_height, config)
    edge_x, edge_y = _apply_edge_boost(edge_x, edge_y, screen_width, screen_height, config)
    edge_point = ScreenPoint(x=_clamp(round(edge_x), 0, screen_width - 1), y=_clamp(round(edge_y), 0, screen_height - 1))

    if not config.enabled:
        return AxisAdjustmentResult(
            before=point,
            after_edge_reach=edge_point,
            after=edge_point,
            edge_reach_x_float=edge_x,
            edge_reach_y_float=edge_y,
            adjusted_x_float=edge_x,
            adjusted_y_float=edge_y,
            horizontal_gain=config.horizontal_gain,
            vertical_gain=config.vertical_gain,
            horizontal_offset=config.horizontal_offset,
            vertical_offset=config.vertical_offset,
            edge_reach_enabled=config.edge_reach_enabled,
            edge_margin_px=config.edge_margin_px,
            edge_boost_enabled=config.edge_boost_enabled,
            edge_boost_gamma=config.edge_boost_gamma,
        )

    max_x = float(screen_width - 1)
    max_y = float(screen_height - 1)
    center_x = max_x / 2
    center_y = max_y / 2
    horizontal_offset = 0.0 if config.edge_reach_enabled and _is_screen_edge(edge_x, max_x) else config.horizontal_offset
    vertical_offset = 0.0 if config.edge_reach_enabled and _is_screen_edge(edge_y, max_y) else config.vertical_offset
    adjusted_x = center_x + (edge_x - center_x) * config.horizontal_gain + horizontal_offset
    adjusted_y = center_y + (edge_y - center_y) * config.vertical_gain + vertical_offset
    clamped_x = _clamp(round(adjusted_x), 0, screen_width - 1)
    clamped_y = _clamp(round(adjusted_y), 0, screen_height - 1)
    return AxisAdjustmentResult(
        before=point,
        after_edge_reach=edge_point,
        after=ScreenPoint(x=clamped_x, y=clamped_y),
        edge_reach_x_float=edge_x,
        edge_reach_y_float=edge_y,
        adjusted_x_float=adjusted_x,
        adjusted_y_float=adjusted_y,
        horizontal_gain=config.horizontal_gain,
        vertical_gain=config.vertical_gain,
        horizontal_offset=config.horizontal_offset,
        vertical_offset=config.vertical_offset,
        edge_reach_enabled=config.edge_reach_enabled,
        edge_margin_px=config.edge_margin_px,
        edge_boost_enabled=config.edge_boost_enabled,
        edge_boost_gamma=config.edge_boost_gamma,
    )


def _is_screen_edge(value: float, maximum: float, tolerance: float = 0.5) -> bool:
    return value <= tolerance or value >= maximum - tolerance


def _apply_edge_reach(
    point: ScreenPoint,
    screen_width: int,
    screen_height: int,
    config: AxisAdjustmentConfig,
) -> tuple[float, float]:
    if not config.edge_reach_enabled:
        return float(point.x), float(point.y)
    if (
        config.source_min_x is None
        or config.source_max_x is None
        or config.source_min_y is None
        or config.source_max_y is None
    ):
        return float(point.x), float(point.y)

    margin = max(0, int(config.edge_margin_px))
    target_min_x = min(margin, screen_width - 1)
    target_max_x = max(target_min_x, screen_width - 1 - margin)
    target_min_y = min(margin, screen_height - 1)
    target_max_y = max(target_min_y, screen_height - 1 - margin)

    expanded_x = _scale_between(
        float(point.x),
        config.source_min_x,
        config.source_max_x,
        target_min_x,
        target_max_x,
    )
    expanded_y = _scale_between(
        float(point.y),
        config.source_min_y,
        config.source_max_y,
        target_min_y,
        target_max_y,
    )
    return expanded_x, expanded_y


def _scale_between(value: float, source_min: float, source_max: float, target_min: float, target_max: float) -> float:
    source_range = source_max - source_min
    if abs(source_range) < 1e-6:
        return value
    ratio = (value - source_min) / source_range
    return target_min + ratio * (target_max - target_min)


def _apply_edge_boost(
    x: float,
    y: float,
    screen_width: int,
    screen_height: int,
    config: AxisAdjustmentConfig,
) -> tuple[float, float]:
    if not config.edge_boost_enabled:
        return x, y
    gamma = max(0.35, min(1.0, float(config.edge_boost_gamma)))
    exponent = 1.0 / gamma
    return (
        _boost_axis(x, 0, screen_width - 1, exponent),
        _boost_axis(y, 0, screen_height - 1, exponent),
    )


def _boost_axis(value: float, minimum: float, maximum: float, exponent: float) -> float:
    span = maximum - minimum
    if span <= 0:
        return value
    normalized = max(0.0, min(1.0, (value - minimum) / span))
    if normalized < 0.5:
        boosted = 0.5 * ((2.0 * normalized) ** exponent)
    else:
        boosted = 1.0 - 0.5 * ((2.0 * (1.0 - normalized)) ** exponent)
    return minimum + boosted * span


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))
