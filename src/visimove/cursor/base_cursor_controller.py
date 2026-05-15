from __future__ import annotations

from typing import Protocol

from visimove.cursor.cursor_safety import CursorSafety
from visimove.utils.screen import ScreenBounds


class BaseCursorController(Protocol):
    def move_to(self, x: int, y: int, gaze_confidence: float = 1.0, target_visible: bool = True) -> None: ...

    def left_click(self) -> None: ...

    def right_click(self) -> None: ...

    def double_click(self) -> None: ...

    def scroll(self, amount: int) -> None: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...


class DryRunCursorController:
    def __init__(self, safety: CursorSafety | None = None) -> None:
        self.safety = safety or CursorSafety(bounds=ScreenBounds(width=1920, height=1080))
        self.last_position: tuple[int, int] | None = None
        self.left_click_count = 0
        self.right_click_count = 0
        self.double_click_count = 0
        self.scroll_total = 0

    def move_to(
        self,
        x: int,
        y: int,
        gaze_confidence: float = 1.0,
        target_visible: bool = True,
    ) -> None:
        target = self.safety.safe_target(x, y, gaze_confidence, target_visible)
        if target is not None:
            self.last_position = target

    def left_click(self) -> None:
        if self.safety.can_click():
            self.left_click_count += 1
            self.safety.mark_click()

    def right_click(self) -> None:
        if self.safety.can_click():
            self.right_click_count += 1
            self.safety.mark_click()

    def double_click(self) -> None:
        if self.safety.can_click():
            self.double_click_count += 1
            self.safety.mark_click()

    def scroll(self, amount: int) -> None:
        if not self.safety.paused:
            self.scroll_total += amount

    def pause(self) -> None:
        self.safety.pause()

    def resume(self) -> None:
        self.safety.resume()

    def click(self) -> None:
        self.left_click()

