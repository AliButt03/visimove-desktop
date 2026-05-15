from __future__ import annotations


class CliInterface:
    def show_status(self, message: str) -> None:
        print(message)

    def show_error(self, message: str) -> None:
        print(f"ERROR: {message}")

