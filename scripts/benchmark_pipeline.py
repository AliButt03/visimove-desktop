from __future__ import annotations

from visimove.config import load_config
from visimove.pipeline.realtime import RealtimePipeline
from visimove.pipeline.runtime import build_runtime


def main() -> None:
    config = load_config("config/default.yaml", "config/performance.yaml")
    config.setdefault("pipeline", {})["dry_run"] = True
    pipeline = RealtimePipeline(runtime=build_runtime(config), config=config)
    pipeline.run(max_frames=120)
    print(f"FPS: {pipeline.performance.fps:.2f}")
    print(f"Average frame time: {pipeline.performance.average_frame_time_ms:.2f} ms")


if __name__ == "__main__":
    main()

