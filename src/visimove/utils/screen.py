from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenBounds:
    width: int
    height: int

    @property
    def max_x(self) -> int:
        return max(0, self.width - 1)

    @property
    def max_y(self) -> int:
        return max(0, self.height - 1)

    def clamp(self, x: float, y: float) -> tuple[int, int]:
        clamped_x = min(max(round(x), 0), self.max_x)
        clamped_y = min(max(round(y), 0), self.max_y)
        return clamped_x, clamped_y


def get_screen_bounds() -> ScreenBounds:
    try:
        import pyautogui

        width, height = pyautogui.size()
        return ScreenBounds(width=int(width), height=int(height))
    except Exception:
        return ScreenBounds(width=1920, height=1080)

