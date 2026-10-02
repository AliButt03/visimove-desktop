from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from visimove.utils.screen import ScreenBounds


@dataclass(frozen=True)
class CursorSafetyConfig:
    min_gaze_confidence: float = 0.65
    max_speed_px_per_sec: float = 1400.0
    click_cooldown_ms: int = 500
    movement_duration_sec: float = 0.03
    emergency_pause_key: str = "p"


@dataclass
class CursorSafety:
    bounds: ScreenBounds
    config: CursorSafetyConfig = CursorSafetyConfig()
    paused: bool = True
    _last_position: tuple[int, int] | None = None
    _last_move_at: float | None = None
    _last_click_at: float = 0.0

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def reset_motion(self, current_position: tuple[int, int] | None = None) -> None:
        self._last_position = current_position
        self._last_move_at = monotonic()

    @property
    def current_position(self) -> tuple[int, int] | None:
        return self._last_position

    def can_move(self, gaze_confidence: float, target_visible: bool) -> bool:
        return (
            not self.paused
            and target_visible
            and gaze_confidence >= self.config.min_gaze_confidence
        )

    def can_click(self) -> bool:
        if self.paused:
            return False
        elapsed_ms = (monotonic() - self._last_click_at) * 1000
        return elapsed_ms >= self.config.click_cooldown_ms

    def mark_click(self) -> None:
        self._last_click_at = monotonic()

    def safe_target(
        self,
        x: float,
        y: float,
        gaze_confidence: float = 1.0,
        target_visible: bool = True,
    ) -> tuple[int, int] | None:
        if not self.can_move(gaze_confidence=gaze_confidence, target_visible=target_visible):
            return None

        target = self.bounds.clamp(x, y)
        now = monotonic()
        if self._last_position is None or self._last_move_at is None:
            self._last_position = target
            self._last_move_at = now
            return target

        elapsed = max(now - self._last_move_at, 1 / 120)
        max_distance = max(1.0, self.config.max_speed_px_per_sec * elapsed)
        limited = self._limit_distance(self._last_position, target, max_distance)
        self._last_position = limited
        self._last_move_at = now
        return limited

    @staticmethod
    def _limit_distance(
        origin: tuple[int, int],
        target: tuple[int, int],
        max_distance: float,
    ) -> tuple[int, int]:
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        distance = (dx * dx + dy * dy) ** 0.5
        if distance <= max_distance:
            return target
        scale = max_distance / distance
        return round(origin[0] + dx * scale), round(origin[1] + dy * scale)
