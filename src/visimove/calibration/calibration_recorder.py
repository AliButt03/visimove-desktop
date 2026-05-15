from __future__ import annotations

from dataclasses import dataclass, field
from time import sleep
from time import monotonic
from typing import Protocol

from visimove.calibration.calibration_points import CalibrationPoint
from visimove.calibration.calibration_store import CalibrationSample


class GazeSampleProvider(Protocol):
    def read_gaze(self) -> tuple[float, float] | None: ...


@dataclass
class CalibrationRecorder:
    sample_provider: GazeSampleProvider
    stabilization_seconds: float = 0.75
    sample_seconds: float = 1.7
    samples: list[CalibrationSample] = field(default_factory=list)

    def record_point(self, point: CalibrationPoint) -> list[CalibrationSample]:
        self._wait(self.stabilization_seconds)
        started = monotonic()
        point_samples: list[CalibrationSample] = []
        while monotonic() - started < self.sample_seconds:
            gaze = self.sample_provider.read_gaze()
            if gaze is not None:
                sample = CalibrationSample(
                    raw_gaze=gaze,
                    target_screen=(point.screen_x, point.screen_y),
                    timestamp=monotonic(),
                )
                self.samples.append(sample)
                point_samples.append(sample)
            self._wait(1 / 60)
        return point_samples

    @staticmethod
    def _wait(seconds: float) -> None:
        sleep(max(0.0, seconds))
