from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CalibrationPointMode(str, Enum):
    FIVE = "5"
    NINE = "9"
    SIXTEEN = "16"


@dataclass(frozen=True)
class CalibrationPoint:
    index: int
    normalized_x: float
    normalized_y: float
    screen_x: int
    screen_y: int


def generate_calibration_points(
    mode: CalibrationPointMode | str,
    screen_width: int,
    screen_height: int,
    margin: float = 0.12,
) -> list[CalibrationPoint]:
    normalized = generate_normalized_points(mode=mode, margin=margin)
    return [
        CalibrationPoint(
            index=index,
            normalized_x=x,
            normalized_y=y,
            screen_x=round(x * (screen_width - 1)),
            screen_y=round(y * (screen_height - 1)),
        )
        for index, (x, y) in enumerate(normalized)
    ]


def generate_normalized_points(
    mode: CalibrationPointMode | str,
    margin: float = 0.12,
) -> list[tuple[float, float]]:
    point_mode = CalibrationPointMode(str(mode))
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must be in the range [0.0, 0.5).")

    if point_mode is CalibrationPointMode.FIVE:
        low = margin
        high = 1.0 - margin
        return [(0.5, 0.5), (low, low), (high, low), (low, high), (high, high)]

    grid_size = 3 if point_mode is CalibrationPointMode.NINE else 4
    values = _linspace(margin, 1.0 - margin, grid_size)
    return [(x, y) for y in values for x in values]


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count < 2:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]

