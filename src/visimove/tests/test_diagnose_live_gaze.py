from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType


def load_diagnose_live_gaze_module() -> ModuleType:
    project_root = Path(__file__).resolve().parents[3]
    module_path = project_root / "scripts" / "diagnose_live_gaze.py"
    spec = importlib.util.spec_from_file_location("diagnose_live_gaze", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["diagnose_live_gaze"] = module
    spec.loader.exec_module(module)
    return module


def make_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "camera_index": None,
        "detector_backend": None,
        "gaze_backend": None,
        "calibration_profile": None,
        "horizontal_gain": None,
        "vertical_gain": None,
        "horizontal_offset": None,
        "vertical_offset": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_mobilegaze_live_diagnostic_uses_backend_specific_profile(tmp_path: Path) -> None:
    diagnose_live_gaze = load_diagnose_live_gaze_module()
    calibration_dir = tmp_path / "data" / "calibration"
    calibration_dir.mkdir(parents=True)
    mobilegaze_profile = calibration_dir / "user_profile_mobilegaze.json"
    mobilegaze_profile.write_text("{}", encoding="utf-8")
    diagnose_live_gaze.PROJECT_ROOT = tmp_path

    config = {
        "camera": {"index": 0},
        "detection": {"detector_backend": "auto", "backend": "auto"},
        "gaze": {"gaze_backend": "dummy", "backend": "dummy"},
        "calibration": {"profile_path": "data/calibration/user_profile.json"},
    }

    diagnose_live_gaze.apply_cli_overrides(config, make_args(gaze_backend="mobilegaze"))

    assert config["gaze"]["gaze_backend"] == "mobilegaze"
    assert Path(str(config["calibration"]["profile_path"])) == mobilegaze_profile


def test_explicit_live_diagnostic_profile_overrides_backend_default(tmp_path: Path) -> None:
    diagnose_live_gaze = load_diagnose_live_gaze_module()
    explicit_profile = tmp_path / "custom_profile.json"
    diagnose_live_gaze.PROJECT_ROOT = tmp_path
    config = {
        "gaze": {"gaze_backend": "mobilegaze", "backend": "mobilegaze"},
        "calibration": {"profile_path": "data/calibration/user_profile.json"},
    }

    diagnose_live_gaze.apply_cli_overrides(
        config,
        make_args(gaze_backend="mobilegaze", calibration_profile=str(explicit_profile)),
    )

    assert Path(str(config["calibration"]["profile_path"])) == explicit_profile
