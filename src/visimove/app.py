from __future__ import annotations

from pathlib import Path

from visimove.config import load_config
from visimove.pipeline.realtime import RealtimePipeline
from visimove.pipeline.runtime import build_runtime


def run(config_path: str | Path = "config/default.yaml", dry_run: bool = True) -> None:
    config = load_config(config_path)
    config.setdefault("pipeline", {})["dry_run"] = dry_run
    runtime = build_runtime(config)
    RealtimePipeline(runtime=runtime, config=config).run()


if __name__ == "__main__":
    run()

