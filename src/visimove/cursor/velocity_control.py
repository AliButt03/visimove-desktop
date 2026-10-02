from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import copysign
from statistics import median


PointValue = tuple[float, float] | None


@dataclass
class VelocityCursorFilter:
    """Convert gaze direction into bounded relative cursor movement."""

    screen_width: int
    screen_height: int
    initial_position: tuple[int, int] | None = None
    deadzone: float = 0.20
    horizontal_deadzone: float | None = None
    vertical_deadzone: float | None = None
    max_speed_px_per_sec: float = 1400.0
    response_exponent: float = 1.35
    max_dt_seconds: float = 0.10
    edge_margin_px: int = 8
    neutral_acquisition_seconds: float = 1.0
    neutral_min_samples: int = 8
    median_window: int = 5
    _position: tuple[float, float] | None = None
    _last_timestamp: float | None = None
    _state: str = "init"
    _neutral: tuple[float, float] | None = None
    _neutral_started_at: float | None = None
    _neutral_x_samples: list[float] = field(default_factory=list)
    _neutral_y_samples: list[float] = field(default_factory=list)
    _recent_x: deque[float] = field(default_factory=deque)
    _recent_y: deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.screen_width <= 0 or self.screen_height <= 0:
            raise ValueError("screen dimensions must be positive")
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError("deadzone must be in [0, 1)")
        for axis_name, axis_deadzone in (
            ("horizontal", self.horizontal_deadzone),
            ("vertical", self.vertical_deadzone),
        ):
            if axis_deadzone is not None and not 0.0 <= axis_deadzone < 1.0:
                raise ValueError(f"{axis_name} deadzone must be in [0, 1)")
        if self.max_speed_px_per_sec <= 0:
            raise ValueError("max speed must be positive")
        if self.response_exponent <= 0:
            raise ValueError("response exponent must be positive")
        if self.max_dt_seconds <= 0:
            raise ValueError("maximum frame interval must be positive")
        if self.neutral_acquisition_seconds < 0:
            raise ValueError("neutral acquisition time cannot be negative")
        if self.neutral_min_samples < 1:
            raise ValueError("neutral calibration requires at least one sample")
        if self.median_window < 1 or self.median_window % 2 == 0:
            raise ValueError("median window must be a positive odd number")
        self._recent_x = deque(maxlen=self.median_window)
        self._recent_y = deque(maxlen=self.median_window)
        if self.initial_position is not None:
            self._position = self._clamp_position(*self.initial_position)

    def update(self, x: float | None, y: float | None, timestamp: float) -> PointValue:
        if self._position is None:
            self._position = self._clamp_position(
                (self.screen_width - 1) / 2.0,
                (self.screen_height - 1) / 2.0,
            )

        if x is None or y is None:
            self._last_timestamp = timestamp
            self._state = "missing"
            return self._position

        gaze_x = float(x)
        gaze_y = float(y)
        if self._neutral is None:
            self._collect_neutral(gaze_x, gaze_y, timestamp)
            return self._position

        self._recent_x.append(gaze_x)
        self._recent_y.append(gaze_y)
        filtered_x = median(self._recent_x)
        filtered_y = median(self._recent_y)

        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            self._state = "hold"
            return self._position

        elapsed = min(max(timestamp - self._last_timestamp, 0.0), self.max_dt_seconds)
        self._last_timestamp = timestamp
        direction_x = self._axis_response(
            self._normalize_from_neutral(filtered_x, self._neutral[0], self.screen_width),
            self.deadzone if self.horizontal_deadzone is None else self.horizontal_deadzone,
        )
        direction_y = self._axis_response(
            self._normalize_from_neutral(filtered_y, self._neutral[1], self.screen_height),
            self.deadzone if self.vertical_deadzone is None else self.vertical_deadzone,
        )

        if direction_x == 0.0 and direction_y == 0.0:
            self._state = "hold"
            return self._position

        next_x = self._position[0] + direction_x * self.max_speed_px_per_sec * elapsed
        next_y = self._position[1] + direction_y * self.max_speed_px_per_sec * elapsed
        self._position = self._clamp_position(next_x, next_y)
        self._state = "moving"
        return self._position

    def begin_neutral_calibration(self) -> None:
        self._neutral = None
        self._neutral_started_at = None
        self._neutral_x_samples.clear()
        self._neutral_y_samples.clear()
        self._recent_x.clear()
        self._recent_y.clear()
        self._last_timestamp = None
        self._state = "calibrating"

    def neutral_point(self) -> tuple[float, float] | None:
        return self._neutral

    def set_position(self, position: tuple[int, int] | None) -> None:
        if position is not None:
            self._position = self._clamp_position(*position)
        self._last_timestamp = None
        self._state = "init" if self._neutral is None else "hold"

    def reset(self) -> None:
        self._position = None if self.initial_position is None else self._clamp_position(*self.initial_position)
        self.begin_neutral_calibration()

    def debug_state(self) -> str:
        return f"velocity_{self._state}"

    def _collect_neutral(self, x: float, y: float, timestamp: float) -> None:
        if self._neutral_started_at is None:
            self._neutral_started_at = timestamp
        self._neutral_x_samples.append(x)
        self._neutral_y_samples.append(y)
        elapsed = max(timestamp - self._neutral_started_at, 0.0)
        if elapsed + 1e-9 < self.neutral_acquisition_seconds or len(self._neutral_x_samples) < self.neutral_min_samples:
            self._last_timestamp = timestamp
            self._state = "calibrating"
            return

        self._neutral = (median(self._neutral_x_samples), median(self._neutral_y_samples))
        self._recent_x.extend([self._neutral[0]] * self.median_window)
        self._recent_y.extend([self._neutral[1]] * self.median_window)
        self._last_timestamp = timestamp
        self._state = "hold"

    @staticmethod
    def _normalize_from_neutral(value: float, neutral: float, size: int) -> float:
        maximum = float(size - 1)
        if value >= neutral:
            span = maximum - neutral
        else:
            span = neutral
        if span <= 0:
            return 0.0
        return min(max((value - neutral) / span, -1.0), 1.0)

    def _axis_response(self, value: float, deadzone: float) -> float:
        magnitude = abs(value)
        if magnitude <= deadzone:
            return 0.0
        scaled = (magnitude - deadzone) / (1.0 - deadzone)
        return copysign(scaled**self.response_exponent, value)

    def _clamp_position(self, x: float, y: float) -> tuple[float, float]:
        margin = max(0, int(self.edge_margin_px))
        min_x = min(float(margin), float(self.screen_width - 1))
        min_y = min(float(margin), float(self.screen_height - 1))
        max_x = max(min_x, float(self.screen_width - 1 - margin))
        max_y = max(min_y, float(self.screen_height - 1 - margin))
        return min(max(float(x), min_x), max_x), min(max(float(y), min_y), max_y)

