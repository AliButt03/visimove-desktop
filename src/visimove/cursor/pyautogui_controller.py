from __future__ import annotations

from visimove.cursor.cursor_safety import CursorSafety
from visimove.utils.screen import get_screen_bounds


class PyAutoGuiCursorController:
    def __init__(
        self,
        safety: CursorSafety | None = None,
        movement_duration_sec: float = 0.03,
    ) -> None:
        import pyautogui

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0
        self._pyautogui = pyautogui
        self.safety = safety or CursorSafety(bounds=get_screen_bounds())
        self.movement_duration_sec = movement_duration_sec
        self.safety.reset_motion(tuple(map(int, self._pyautogui.position())))

    def move_to(
        self,
        x: int,
        y: int,
        gaze_confidence: float = 1.0,
        target_visible: bool = True,
    ) -> None:
        target = self.safety.safe_target(x, y, gaze_confidence, target_visible)
        if target is not None:
            try:
                self._pyautogui.moveTo(*target, duration=self.movement_duration_sec)
            except self._pyautogui.FailSafeException:
                self.safety.pause()
                print("Cursor paused: PyAutoGUI fail-safe detected a screen corner. Move the mouse away, then press 'p' to resume.")

    def left_click(self) -> None:
        if self.safety.can_click():
            self._pyautogui.click(button="left")
            self.safety.mark_click()

    def right_click(self) -> None:
        if self.safety.can_click():
            self._pyautogui.click(button="right")
            self.safety.mark_click()

    def double_click(self) -> None:
        if self.safety.can_click():
            self._pyautogui.doubleClick()
            self.safety.mark_click()

    def scroll(self, amount: int) -> None:
        if not self.safety.paused:
            self._pyautogui.scroll(amount)

    def pause(self) -> None:
        self.safety.pause()

    def resume(self) -> None:
        self.safety.resume()
        self.safety.reset_motion(tuple(map(int, self._pyautogui.position())))

    def click(self) -> None:
        self.left_click()

