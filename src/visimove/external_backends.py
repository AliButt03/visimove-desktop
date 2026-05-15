from __future__ import annotations

from pathlib import Path

from visimove.config import PROJECT_ROOT


class ExternalBackendUnavailable(RuntimeError):
    """Raised when an optional external backend is configured but not installed."""


def external_repo_path(name: str) -> Path:
    return PROJECT_ROOT / "external" / name


def require_external_repo(name: str) -> Path:
    path = external_repo_path(name)
    installed_files = [child for child in path.iterdir()] if path.exists() else []
    meaningful_files = [child for child in installed_files if child.name != ".gitkeep"]
    if not path.exists() or not meaningful_files:
        raise ExternalBackendUnavailable(
            f"External backend '{name}' is not installed. See external/README.md."
        )
    return path
