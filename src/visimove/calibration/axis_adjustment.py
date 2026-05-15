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


@dataclass(frozen=True)
class AxisAdjustmentResult:
    before: ScreenPoint
    after: ScreenPoint
    adjusted_x_float: float
    adjusted_y_float: float
    horizontal_gain: float
    vertical_gain: float
    horizontal_offset: float
    vertical_offset: float


def apply_axis_adjustment(
    point: ScreenPoint,
    screen_width: int,
    screen_height: int,
    config: AxisAdjustmentConfig,
) -> AxisAdjustmentResult:
    if not config.enabled:
        return AxisAdjustmentResult(
            before=point,
            after=point,
            adjusted_x_float=float(point.x),
            adjusted_y_float=float(point.y),
            horizontal_gain=config.horizontal_gain,
            vertical_gain=config.vertical_gain,
            horizontal_offset=config.horizontal_offset,
            vertical_offset=config.vertical_offset,
        )

    center_x = (screen_width - 1) / 2
    center_y = (screen_height - 1) / 2
    adjusted_x = center_x + (point.x - center_x) * config.horizontal_gain + config.horizontal_offset
    adjusted_y = center_y + (point.y - center_y) * config.vertical_gain + config.vertical_offset
    clamped_x = _clamp(round(adjusted_x), 0, screen_width - 1)
    clamped_y = _clamp(round(adjusted_y), 0, screen_height - 1)
    return AxisAdjustmentResult(
        before=point,
        after=ScreenPoint(x=clamped_x, y=clamped_y),
        adjusted_x_float=adjusted_x,
        adjusted_y_float=adjusted_y,
        horizontal_gain=config.horizontal_gain,
        vertical_gain=config.vertical_gain,
        horizontal_offset=config.horizontal_offset,
        vertical_offset=config.vertical_offset,
    )


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))
