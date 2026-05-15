from __future__ import annotations

from typing import Protocol


class UserInterface(Protocol):
    def show_status(self, message: str) -> None: ...

    def show_error(self, message: str) -> None: ...

