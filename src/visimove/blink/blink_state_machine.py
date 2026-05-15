from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ClickEvent(str, Enum):
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"
    PAUSE_TOGGLE = "pause_toggle"
    NONE = "none"


@dataclass(frozen=True)
class BlinkStateMachineConfig:
    closed_threshold: float = 0.65
    min_blink_ms: int = 150
    max_blink_ms: int = 600
    long_blink_ms: int = 800
    double_blink_gap_ms: int = 450
    click_cooldown_ms: int = 1000
    long_blink_event: ClickEvent = ClickEvent.RIGHT_CLICK


class BlinkStateMachine:
    """Turns closed-eye probabilities into debounced click events."""

    def __init__(self, config: BlinkStateMachineConfig | None = None) -> None:
        self.config = config or BlinkStateMachineConfig()
        self._is_closed = False
        self._closed_started_at: float | None = None
        self._last_click_at: float | None = None
        self._pending_single_click_at: float | None = None
        self._pending_single_consumed_by_double = False

    def update(self, closed_probability: float, timestamp: float) -> ClickEvent | None:
        is_closed_now = closed_probability >= self.config.closed_threshold

        if is_closed_now and not self._is_closed:
            self._is_closed = True
            self._closed_started_at = timestamp
            return self._flush_pending_if_expired(timestamp)

        if is_closed_now:
            return self._flush_pending_if_expired(timestamp)

        if self._is_closed:
            self._is_closed = False
            if self._closed_started_at is None:
                return self._flush_pending_if_expired(timestamp)

            duration_ms = self._elapsed_ms(self._closed_started_at, timestamp)
            self._closed_started_at = None
            return self._event_for_blink(duration_ms, timestamp)

        return self._flush_pending_if_expired(timestamp)

    def flush(self, timestamp: float) -> ClickEvent | None:
        return self._flush_pending_if_expired(timestamp, force=True)

    def _event_for_blink(self, duration_ms: float, timestamp: float) -> ClickEvent | None:
        if duration_ms < self.config.min_blink_ms:
            return self._flush_pending_if_expired(timestamp)

        if duration_ms >= self.config.long_blink_ms:
            self._clear_pending()
            return self._emit(self.config.long_blink_event, timestamp)

        if duration_ms > self.config.max_blink_ms:
            return self._flush_pending_if_expired(timestamp)

        if (
            self._pending_single_click_at is not None
            and self._elapsed_ms(self._pending_single_click_at, timestamp)
            <= self.config.double_blink_gap_ms
            and not self._pending_single_consumed_by_double
        ):
            self._pending_single_consumed_by_double = True
            self._pending_single_click_at = None
            return self._emit(ClickEvent.DOUBLE_CLICK, timestamp)

        if self._cooldown_active(timestamp):
            self._clear_pending()
            return None

        self._pending_single_click_at = timestamp
        self._pending_single_consumed_by_double = False
        return None

    def _flush_pending_if_expired(self, timestamp: float, force: bool = False) -> ClickEvent | None:
        if self._pending_single_click_at is None:
            return None

        elapsed_ms = self._elapsed_ms(self._pending_single_click_at, timestamp)
        if force or elapsed_ms > self.config.double_blink_gap_ms:
            self._pending_single_click_at = None
            if self._pending_single_consumed_by_double:
                self._pending_single_consumed_by_double = False
                return None
            return self._emit(ClickEvent.LEFT_CLICK, timestamp)
        return None

    def _emit(self, event: ClickEvent, timestamp: float) -> ClickEvent | None:
        if event is ClickEvent.NONE:
            return None
        if self._cooldown_active(timestamp):
            return None
        self._last_click_at = timestamp
        return event

    def _cooldown_active(self, timestamp: float) -> bool:
        if self._last_click_at is None:
            return False
        return self._elapsed_ms(self._last_click_at, timestamp) < self.config.click_cooldown_ms

    def _clear_pending(self) -> None:
        self._pending_single_click_at = None
        self._pending_single_consumed_by_double = False

    @staticmethod
    def _elapsed_ms(start: float, end: float) -> float:
        return max(0.0, (end - start) * 1000)

