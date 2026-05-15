from __future__ import annotations

from typing import Any

from visimove.cursor.dwell import DwellSelector
from visimove.pipeline.runtime import PipelineRuntime
from visimove.utils.performance import PerformanceMonitor


class RealtimePipeline:
    def __init__(self, runtime: PipelineRuntime, config: dict[str, Any]) -> None:
        self.runtime = runtime
        self.config = config
        cursor_config = config.get("cursor", {})
        self.dwell = DwellSelector(
            dwell_time_ms=int(cursor_config.get("dwell_time_ms", 900)),
            radius_px=int(cursor_config.get("dwell_radius_px", 35)),
        )
        self.performance = PerformanceMonitor()

    def process_once(self) -> bool:
        frame = self.runtime.camera.read()
        if frame is None:
            return False

        with self.performance.measure_frame():
            detection = self.runtime.detector.detect(frame)
            gaze = self.runtime.gaze_model.estimate(frame, detection)
            blink = self.runtime.blink_model.infer(frame, detection)
            screen_point = self.runtime.mapper.map_to_screen(gaze)
            smoothed = self.runtime.smoother.update(screen_point)
            target_visible = detection.found and detection.eyes is not None
            self.runtime.cursor.move_to(
                smoothed.x,
                smoothed.y,
                gaze_confidence=gaze.confidence,
                target_visible=target_visible,
            )

            dwell_enabled = bool(self.config.get("cursor", {}).get("dwell_enabled", True))
            should_click = blink.is_blinking or (dwell_enabled and self.dwell.update(smoothed))
            if should_click:
                self.runtime.cursor.left_click()
        return True

    def run(self, max_frames: int | None = None) -> None:
        self.runtime.camera.open()
        try:
            frames = 0
            while max_frames is None or frames < max_frames:
                if not self.process_once():
                    break
                frames += 1
        finally:
            self.runtime.camera.close()
