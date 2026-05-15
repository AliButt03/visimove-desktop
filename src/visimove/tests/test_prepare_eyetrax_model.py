from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_prepare_module() -> ModuleType:
    project_root = Path(__file__).resolve().parents[3]
    module_path = project_root / "scripts" / "prepare_eyetrax_model.py"
    spec = importlib.util.spec_from_file_location("prepare_eyetrax_model", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_eyetrax_model"] = module
    spec.loader.exec_module(module)
    return module


def test_missing_face_landmarker_blocks_readiness(tmp_path) -> None:
    module = load_prepare_module()
    output = tmp_path / "models" / "gaze" / "eyetrax" / "gaze_model.pkl"

    status = module.check_readiness(face_model=tmp_path / "missing.task", output_path=output)

    assert status.face_landmarker_present is False
    assert status.ready_to_build is False
    assert "face_landmarker.task" in module.next_action(status)


def test_output_model_detection(tmp_path) -> None:
    module = load_prepare_module()
    face_model = tmp_path / "face_landmarker.task"
    output = tmp_path / "models" / "gaze" / "eyetrax" / "gaze_model.pkl"
    face_model.write_bytes(b"task")
    output.parent.mkdir(parents=True)
    output.write_bytes(b"model")

    status = module.check_readiness(face_model=face_model, output_path=output)

    assert status.face_landmarker_present is True
    assert status.output_model_exists is True


def test_build_command_targets_expected_output(tmp_path) -> None:
    module = load_prepare_module()
    output = tmp_path / "models" / "gaze" / "eyetrax" / "gaze_model.pkl"

    command = module.build_eyetrax_command(
        camera=0,
        output=output,
        model="ridge",
        random_points=60,
        retrain_every=10,
        show_predictions=True,
    )

    assert "eyetrax.app.build_model" in command
    assert "--outfile" in command
    assert str(output) in command
    assert not output.exists()


def test_download_skips_existing_face_landmarker_without_force(tmp_path) -> None:
    module = load_prepare_module()
    face_model = tmp_path / "face_landmarker.task"
    face_model.write_bytes(b"existing")

    module.download_face_landmarker(face_model, force=False)

    assert face_model.read_bytes() == b"existing"
