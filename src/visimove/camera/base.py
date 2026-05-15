from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from visimove.types import Frame


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


class Camera(Protocol):
    def open(self) -> None: ...

    def read(self) -> Frame | None: ...

    def close(self) -> None: ...

