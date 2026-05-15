import numpy as np

from visimove.detection import DummyDetector
from visimove.gaze import EyeTraxAdapter, GazeBackendUnavailable, GazeFollowerAdapter
from visimove.gaze import GazeResult, MovingDummyGazeModel
from visimove.pipeline.realtime_pipeline import build_gaze_backend_config, build_gaze_model


def test_gaze_result_exposes_mapper_compatible_point() -> None:
    result = GazeResult(raw_x=0.25, raw_y=0.75, confidence=0.9)

    assert result.point.x == 0.25
    assert result.point.y == 0.75
    assert result.confidence == 0.9


def test_dummy_gaze_model_returns_standard_gaze_result() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detection = DummyDetector().detect(frame)

    result = MovingDummyGazeModel().estimate(frame, detection)

    assert isinstance(result, GazeResult)
    assert 0.0 <= result.raw_x <= 1.0
    assert 0.0 <= result.raw_y <= 1.0
    assert result.metadata["backend"] == "dummy"


def test_missing_external_gaze_model_falls_back_to_dummy() -> None:
    model = build_gaze_model(
        {
            "gaze_backend": "gazefollower",
            "model_path": "models/gaze/gazefollower/missing.pth",
        }
    )

    assert isinstance(model, MovingDummyGazeModel)


def test_gazefollower_missing_repo_reports_clear_message(tmp_path) -> None:
    status = GazeFollowerAdapter.check_setup(repo_path=tmp_path / "missing-gazefollower")

    assert not status.ready
    assert "external/gazefollower is not set up" in status.reason


def test_gazefollower_missing_model_does_not_claim_real_backend(tmp_path) -> None:
    repo = tmp_path / "gazefollower"
    package = repo / "gazefollower"
    (package / "gaze_estimator").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "GazeFollower.py").write_text("", encoding="utf-8")
    (package / "gaze_estimator" / "MGazeNetGazeEstimator.py").write_text("", encoding="utf-8")

    status = GazeFollowerAdapter.check_setup(repo_path=repo, model_path=tmp_path / "missing.mnn")

    assert not status.ready
    assert "MNN model file not found" in status.reason


def test_eyetrax_missing_repo_reports_clear_setup_message(tmp_path) -> None:
    status = EyeTraxAdapter.check_setup(
        repo_path=tmp_path / "missing-eyetrax",
        model_path=tmp_path / "models" / "gaze_model.pkl",
        face_landmarker_model_path=tmp_path / "face_landmarker.task",
    )

    assert not status.ready
    assert "EyeTrax backend selected" in status.reason


def test_eyetrax_missing_model_does_not_claim_real_backend(tmp_path) -> None:
    repo = tmp_path / "eyetrax"
    package = repo / "src" / "eyetrax"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = 'test'\n", encoding="utf-8")
    (package / "gaze.py").write_text("", encoding="utf-8")

    status = EyeTraxAdapter.check_setup(
        repo_path=repo,
        model_path=tmp_path / "models" / "eyetrax",
        face_landmarker_model_path=tmp_path / "face_landmarker.task",
    )

    assert not status.ready
    assert "model file not found" in status.reason


def test_eyetrax_backend_falls_back_to_dummy_when_allowed(tmp_path) -> None:
    model = build_gaze_model(
        {
            "gaze_backend": "eyetrax",
            "backend": "eyetrax",
            "repo_path": str(tmp_path / "missing-eyetrax"),
            "model_path": str(tmp_path / "models" / "eyetrax"),
            "face_landmarker_model_path": str(tmp_path / "face_landmarker.task"),
            "fallback_to_dummy": True,
        }
    )

    assert isinstance(model, MovingDummyGazeModel)


def test_eyetrax_backend_config_uses_backend_model_path_when_gaze_path_empty() -> None:
    config = build_gaze_backend_config(
        {
            "gaze": {"gaze_backend": "eyetrax", "backend": "eyetrax", "model_path": None},
            "eyetrax": {"model_path": "models/gaze/eyetrax"},
        }
    )

    assert config["model_path"] == "models/gaze/eyetrax"


def test_gazefollower_backend_config_uses_backend_model_path_when_gaze_path_empty() -> None:
    config = build_gaze_backend_config(
        {
            "gaze": {"gaze_backend": "gazefollower", "backend": "gazefollower", "model_path": None},
            "gazefollower": {
                "model_path": "external/gazefollower/gazefollower/res/model_weights/base.mnn"
            },
        }
    )

    assert config["model_path"] == "external/gazefollower/gazefollower/res/model_weights/base.mnn"
