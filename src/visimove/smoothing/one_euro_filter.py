from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, pi, sqrt


PointValue = tuple[float, float] | None


@dataclass
class _OneEuroAxis:
    min_cutoff: float
    beta: float
    derivative_cutoff: float
    _raw: float | None = None
    _filtered: float | None = None
    _derivative: float = 0.0

    def update(self, value: float, elapsed: float) -> float:
        if self._raw is None or self._filtered is None:
            self._raw = value
            self._filtered = value
            return value

        derivative = (value - self._raw) / elapsed
        derivative_alpha = self._alpha(self.derivative_cutoff, elapsed)
        self._derivative = self._low_pass(derivative, self._derivative, derivative_alpha)
        cutoff = self.min_cutoff + self.beta * abs(self._derivative)
        value_alpha = self._alpha(cutoff, elapsed)
        self._filtered = self._low_pass(value, self._filtered, value_alpha)
        self._raw = value
        return self._filtered

    def reset(self) -> None:
        self._raw = None
        self._filtered = None
        self._derivative = 0.0

    @staticmethod
    def _alpha(cutoff: float, elapsed: float) -> float:
        time_constant = 1.0 / (2.0 * pi * cutoff)
        return 1.0 / (1.0 + time_constant / elapsed)

    @staticmethod
    def _low_pass(value: float, previous: float, alpha: float) -> float:
        return alpha * value + (1.0 - alpha) * previous


@dataclass
class OneEuroFilter2D:
    """Speed-adaptive gaze filtering with bounded cursor velocity and acceleration."""

    min_cutoff: float = 0.35
    beta: float = 0.0001
    derivative_cutoff: float = 1.0
    max_dt_seconds: float = 0.20
    max_speed_px_per_sec: float = 1600.0
    max_acceleration_px_per_sec2: float = 5000.0
    _x_filter: _OneEuroAxis = field(init=False)
    _y_filter: _OneEuroAxis = field(init=False)
    _last_timestamp: float | None = None
    _last: tuple[float, float] | None = None
    _velocity: tuple[float, float] = (0.0, 0.0)
    _state: str = "init"

    def __post_init__(self) -> None:
        if self.min_cutoff <= 0:
            raise ValueError("minimum cutoff must be positive")
        if self.beta < 0:
            raise ValueError("beta cannot be negative")
        if self.derivative_cutoff <= 0:
            raise ValueError("derivative cutoff must be positive")
        if self.max_dt_seconds <= 0:
            raise ValueError("maximum frame interval must be positive")
        if self.max_speed_px_per_sec <= 0:
            raise ValueError("maximum speed must be positive")
        if self.max_acceleration_px_per_sec2 <= 0:
            raise ValueError("maximum acceleration must be positive")
        self._x_filter = _OneEuroAxis(self.min_cutoff, self.beta, self.derivative_cutoff)
        self._y_filter = _OneEuroAxis(self.min_cutoff, self.beta, self.derivative_cutoff)

    def update(self, x: float | None, y: float | None, timestamp: float) -> PointValue:
        if x is None or y is None:
            self._last_timestamp = timestamp
            self._velocity = (0.0, 0.0)
            self._state = "missing"
            return self._last

        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            filtered_x = self._x_filter.update(float(x), 1.0 / 60.0)
            filtered_y = self._y_filter.update(float(y), 1.0 / 60.0)
            self._last = (filtered_x, filtered_y)
            self._velocity = (0.0, 0.0)
            self._state = "init"
            return self._last

        elapsed = min(max(timestamp - self._last_timestamp, 1.0 / 240.0), self.max_dt_seconds)
        self._last_timestamp = timestamp
        target = (
            self._x_filter.update(float(x), elapsed),
            self._y_filter.update(float(y), elapsed),
        )
        self._last, limited = self._apply_motion_limits(target, elapsed)
        self._state = "bounded" if limited else "tracking"
        return self._last

    def _apply_motion_limits(
        self,
        target: tuple[float, float],
        elapsed: float,
    ) -> tuple[tuple[float, float], bool]:
        if self._last is None:
            return target, False

        dx = target[0] - self._last[0]
        dy = target[1] - self._last[1]
        distance = hypot(dx, dy)
        if distance <= 0.5:
            self._velocity = (0.0, 0.0)
            return target, False

        stopping_speed = sqrt(2.0 * self.max_acceleration_px_per_sec2 * distance)
        desired_speed = min(self.max_speed_px_per_sec, distance / elapsed, stopping_speed)
        desired_velocity = (dx / distance * desired_speed, dy / distance * desired_speed)

        velocity_dx = desired_velocity[0] - self._velocity[0]
        velocity_dy = desired_velocity[1] - self._velocity[1]
        velocity_change = hypot(velocity_dx, velocity_dy)
        max_velocity_change = self.max_acceleration_px_per_sec2 * elapsed
        acceleration_limited = velocity_change > max_velocity_change
        if acceleration_limited:
            scale = max_velocity_change / velocity_change
            velocity_dx *= scale
            velocity_dy *= scale

        next_velocity = (
            self._velocity[0] + velocity_dx,
            self._velocity[1] + velocity_dy,
        )
        step = (next_velocity[0] * elapsed, next_velocity[1] * elapsed)
        if step[0] * dx + step[1] * dy >= distance * distance:
            self._velocity = (0.0, 0.0)
            return target, acceleration_limited or desired_speed >= self.max_speed_px_per_sec

        self._velocity = next_velocity
        next_position = (self._last[0] + step[0], self._last[1] + step[1])
        speed_limited = desired_speed >= self.max_speed_px_per_sec and distance > desired_speed * elapsed
        return next_position, acceleration_limited or speed_limited

    def set_position(self, position: tuple[int, int] | None) -> None:
        if position is None:
            return
        x, y = float(position[0]), float(position[1])
        self._x_filter.reset()
        self._y_filter.reset()
        self._x_filter.update(x, 1.0 / 60.0)
        self._y_filter.update(y, 1.0 / 60.0)
        self._last = (x, y)
        self._velocity = (0.0, 0.0)
        self._state = "init"

    def reset(self) -> None:
        self._x_filter.reset()
        self._y_filter.reset()
        self._last_timestamp = None
        self._last = None
        self._velocity = (0.0, 0.0)
        self._state = "init"

    def debug_state(self) -> str:
        return f"one_euro_{self._state}"