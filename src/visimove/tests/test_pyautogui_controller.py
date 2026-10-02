from __future__ import annotations

import sys
from types import SimpleNamespace

from visimove.cursor import CursorSafety
from visimove.cursor.pyautogui_controller import PyAutoGuiCursorController
from visimove.utils.screen import ScreenBounds


class FakeFailSafeException(Exception):
    pass


class FakePyAutoGui:
    FAILSAFE = False
    PAUSE = 0.1
    FailSafeException = FakeFailSafeException

    @staticmethod
    def position() -> tuple[int, int]:
        return (50, 50)

    @staticmethod
    def moveTo(*_args: object, **_kwargs: object) -> None:
        raise FakeFailSafeException("corner reached")


def test_pyautogui_fail_safe_pauses_cursor_instead_of_crashing(monkeypatch) -> None:
    fake = SimpleNamespace(
        FAILSAFE=False,
        PAUSE=0.1,
        FailSafeException=FakeFailSafeException,
        position=FakePyAutoGui.position,
        moveTo=FakePyAutoGui.moveTo,
    )
    monkeypatch.setitem(sys.modules, "pyautogui", fake)
    safety = CursorSafety(bounds=ScreenBounds(width=100, height=100), paused=False)
    controller = PyAutoGuiCursorController(safety=safety, movement_duration_sec=0.0)

    controller.move_to(0, 0)

    assert safety.paused is True
