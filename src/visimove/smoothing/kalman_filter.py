from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanFilter2D:
    process_noise: float = 1e-2
    measurement_noise: float = 4.0
    max_jump_pixels: float | None = None
    _state: np.ndarray | None = None
    _covariance: np.ndarray | None = None
    _last_timestamp: float | None = None

    def update(
        self,
        x: float | None,
        y: float | None,
        timestamp: float,
    ) -> tuple[float, float] | None:
        if self._state is None:
            if x is None or y is None:
                return None
            self._state = np.array([float(x), float(y), 0.0, 0.0], dtype=float)
            self._covariance = np.eye(4, dtype=float)
            self._last_timestamp = timestamp
            return float(x), float(y)

        dt = self._delta_time(timestamp)
        self._predict(dt)

        if x is not None and y is not None:
            measurement = np.array([float(x), float(y)], dtype=float)
            measurement = self._clamp_jump(self._state[:2], measurement)
            self._correct(measurement)

        self._last_timestamp = timestamp
        return float(self._state[0]), float(self._state[1])

    def reset(self) -> None:
        self._state = None
        self._covariance = None
        self._last_timestamp = None

    def _delta_time(self, timestamp: float) -> float:
        if self._last_timestamp is None:
            return 1 / 60
        return max(1 / 120, min(timestamp - self._last_timestamp, 0.25))

    def _predict(self, dt: float) -> None:
        assert self._state is not None
        assert self._covariance is not None
        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        process = self.process_noise * np.eye(4, dtype=float)
        self._state = transition @ self._state
        self._covariance = transition @ self._covariance @ transition.T + process

    def _correct(self, measurement: np.ndarray) -> None:
        assert self._state is not None
        assert self._covariance is not None
        observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        noise = self.measurement_noise * np.eye(2, dtype=float)
        residual = measurement - observation @ self._state
        residual_covariance = observation @ self._covariance @ observation.T + noise
        gain = self._covariance @ observation.T @ np.linalg.inv(residual_covariance)
        self._state = self._state + gain @ residual
        self._covariance = (np.eye(4, dtype=float) - gain @ observation) @ self._covariance

    def _clamp_jump(self, origin: np.ndarray, target: np.ndarray) -> np.ndarray:
        if self.max_jump_pixels is None or self.max_jump_pixels <= 0:
            return target
        delta = target - origin
        distance = float(np.linalg.norm(delta))
        if distance <= self.max_jump_pixels:
            return target
        return origin + delta * (self.max_jump_pixels / distance)

