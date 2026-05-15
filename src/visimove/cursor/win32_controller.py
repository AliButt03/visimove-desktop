from __future__ import annotations

import ctypes

from visimove.cursor.cursor_safety import CursorSafety
from visimove.utils.screen import get_screen_bounds


class Win32CursorController:
    """Minimal Win32 cursor backend for later use; PyAutoGUI remains the default."""

    def __init__(self, safety: CursorSafety | None = None) -> None:
        self.safety = safety or CursorSafety(bounds=get_screen_bounds())
        self.user32 = ctypes.windll.user32

    def move_to(
        self,
        x: int,
        y: int,
        gaze_confidence: float = 1.0,
        target_visible: bool = True,
    ) -> None:
        target = self.safety.safe_target(x, y, gaze_confidence, target_visible)
        if target is not None:
            self.user32.SetCursorPos(target[0], target[1])

    def left_click(self) -> None:
        if self.safety.can_click():
            self.user32.mouse_event(0x0002, 0, 0, 0, 0)
            self.user32.mouse_event(0x0004, 0, 0, 0, 0)
            self.safety.mark_click()

    def right_click(self) -> None:
        if self.safety.can_click():
            self.user32.mouse_event(0x0008, 0, 0, 0, 0)
            self.user32.mouse_event(0x0010, 0, 0, 0, 0)
            self.safety.mark_click()

    def double_click(self) -> None:
        if self.safety.can_click():
            self.user32.mouse_event(0x0002, 0, 0, 0, 0)
            self.user32.mouse_event(0x0004, 0, 0, 0, 0)
            self.user32.mouse_event(0x0002, 0, 0, 0, 0)
            self.user32.mouse_event(0x0004, 0, 0, 0, 0)
            self.safety.mark_click()

    def scroll(self, amount: int) -> None:
        if not self.safety.paused:
            self.user32.mouse_event(0x0800, 0, 0, amount, 0)

    def pause(self) -> None:
        self.safety.pause()

    def resume(self) -> None:
        self.safety.resume()

    def click(self) -> None:
        self.left_click()
