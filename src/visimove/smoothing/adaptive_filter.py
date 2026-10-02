from __future__ import annotations

from dataclasses import dataclass


PointValue = tuple[float, float] | None


@dataclass
class AdaptiveSmoothingFilter:
    """Fast movement for intentional gaze changes, hold/slow smoothing for fixation jitter."""

    fast_alpha: float = 0.38
    slow_alpha: float = 0.025
    fixation_radius: float = 90.0
    release_radius: float = 220.0
    fixation_hold_ms: int = 180
    max_jump_pixels: float | None = None
    jump_confirm_radius: float = 180.0
    jump_confirm_samples: int = 2
    screen_width: float | None = None
    screen_height: float | None = None
    edge_snap_margin: float = 32.0
    edge_fast_alpha: float = 0.65
    _last: tuple[float, float] | None = None
    _anchor: tuple[float, float] | None = None
    _anchor_started_at: float | None = None
    _pending_target: tuple[float, float] | None = None
    _pending_count: int = 0
    _state: str = "init"

    def update(self, x: float | None, y: float | None, timestamp: float) -> PointValue:
        if x is None or y is None:
            self._state = "missing"
            self._clear_pending_target()
            return self._last

        target = (float(x), float(y))
        if self._last is None:
            self._last = target
            self._anchor = target
            self._anchor_started_at = timestamp
            self._state = "init"
            return self._last

        target = self._clamp_jump(self._last, target)
        distance_from_last = self._distance(self._last, target)
        distance_from_anchor = self._distance(self._anchor or self._last, target)

        if self._is_edge_target(target):
            if distance_from_last > self.release_radius and not self._large_jump_confirmed(target):
                self._state = "confirming"
                return self._last
            self._clear_pending_target()
            self._anchor = target
            self._anchor_started_at = timestamp
            if distance_from_last <= self.edge_snap_margin:
                self._state = "edge_hold"
                self._last = target
                return self._last
            self._state = "edge"
            self._last = self._blend(self._last, target, max(self.fast_alpha, self.edge_fast_alpha))
            return self._last

        if distance_from_last > self.release_radius:
            if not self._large_jump_confirmed(target):
                self._state = "confirming"
                return self._last
            self._anchor = target
            self._anchor_started_at = timestamp
            self._state = "moving"
            self._last = self._blend(self._last, target, self.fast_alpha)
            return self._last

        if distance_from_anchor <= self.fixation_radius:
            self._clear_pending_target()
            if self._anchor is None:
                self._anchor = self._last
                self._anchor_started_at = timestamp
            elapsed_ms = 0.0 if self._anchor_started_at is None else (timestamp - self._anchor_started_at) * 1000
            if elapsed_ms >= self.fixation_hold_ms:
                self._state = "hold"
                return self._last
            self._state = "settling"
            self._last = self._blend(self._last, target, self.slow_alpha)
            return self._last

        if distance_from_last <= self.fixation_radius:
            self._clear_pending_target()
            if self._anchor is None:
                self._anchor = self._last
                self._anchor_started_at = timestamp
            elapsed_ms = 0.0 if self._anchor_started_at is None else (timestamp - self._anchor_started_at) * 1000
            if elapsed_ms >= self.fixation_hold_ms:
                self._state = "hold"
                return self._last
            self._state = "settling"
            self._last = self._blend(self._last, target, self.slow_alpha)
            return self._last

        if distance_from_anchor > self.release_radius:
            if not self._large_jump_confirmed(target):
                self._state = "confirming"
                return self._last
            self._anchor = target
            self._anchor_started_at = timestamp
            self._state = "moving"
            self._last = self._blend(self._last, target, self.fast_alpha)
            return self._last

        self._clear_pending_target()
        self._state = "slow"
        self._last = self._blend(self._last, target, self.slow_alpha)
        return self._last

    def reset(self) -> None:
        self._last = None
        self._anchor = None
        self._anchor_started_at = None
        self._pending_target = None
        self._pending_count = 0
        self._state = "init"

    def debug_state(self) -> str:
        return self._state

    @staticmethod
    def _blend(origin: tuple[float, float], target: tuple[float, float], alpha: float) -> tuple[float, float]:
        return (
            alpha * target[0] + (1.0 - alpha) * origin[0],
            alpha * target[1] + (1.0 - alpha) * origin[1],
        )

    def _clamp_jump(
        self,
        origin: tuple[float, float],
        target: tuple[float, float],
    ) -> tuple[float, float]:
        if self.max_jump_pixels is None or self.max_jump_pixels <= 0:
            return target
        distance = self._distance(origin, target)
        if distance <= self.max_jump_pixels:
            return target
        scale = self.max_jump_pixels / distance
        return origin[0] + (target[0] - origin[0]) * scale, origin[1] + (target[1] - origin[1]) * scale

    def _large_jump_confirmed(self, target: tuple[float, float]) -> bool:
        required = max(1, int(self.jump_confirm_samples))
        if required <= 1:
            self._clear_pending_target()
            return True
        if self._pending_target is None or self._distance(self._pending_target, target) > self.jump_confirm_radius:
            self._pending_target = target
            self._pending_count = 1
            return False
        self._pending_count += 1
        if self._pending_count < required:
            return False
        self._clear_pending_target()
        return True

    def _clear_pending_target(self) -> None:
        self._pending_target = None
        self._pending_count = 0

    def _is_edge_target(self, target: tuple[float, float]) -> bool:
        if self.edge_snap_margin <= 0 or self.screen_width is None or self.screen_height is None:
            return False
        max_x = None if self.screen_width is None else max(0.0, float(self.screen_width) - 1.0)
        max_y = None if self.screen_height is None else max(0.0, float(self.screen_height) - 1.0)
        near_min_x = target[0] <= self.edge_snap_margin
        near_min_y = target[1] <= self.edge_snap_margin
        near_max_x = max_x is not None and target[0] >= max_x - self.edge_snap_margin
        near_max_y = max_y is not None and target[1] >= max_y - self.edge_snap_margin
        return near_min_x or near_min_y or near_max_x or near_max_y

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return (dx * dx + dy * dy) ** 0.5
