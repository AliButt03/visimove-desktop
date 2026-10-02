import numpy as np

from visimove.detection import DummyDetector
from visimove.gaze import EyeTraxAdapter, GazeBackendUnavailable, GazeFollowerAdapter, MobileGazeAdapter
from visimove.gaze import GazeResult, MovingDummyGazeModel
from visimove.gaze.mobilegaze_adapter import _RawGazeMedianFilter
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


def test_gazefollower_native_coordinates_are_not_treated_as_screen_pixels() -> None:
    raw_x, raw_y, units = GazeFollowerAdapter._normalize_coordinates(0.367, -9.740)

    assert units == "gazefollower_model_coordinates"
    assert 0.5 < raw_x < 0.6
    assert 0.0 < raw_y < 0.5


def test_gazefollower_native_zero_maps_to_center() -> None:
    raw_x, raw_y, units = GazeFollowerAdapter._normalize_coordinates(0.0, 0.0)

    assert units == "gazefollower_model_coordinates"
    assert raw_x == 0.5
    assert raw_y == 0.5


def test_gazefollower_native_coordinate_scale_is_configurable() -> None:
    default_x, _default_y, _units = GazeFollowerAdapter._normalize_coordinates(5.0, 0.0, scale_x=10.0)
    tighter_x, _tighter_y, _units = GazeFollowerAdapter._normalize_coordinates(5.0, 0.0, scale_x=2.0)

    assert tighter_x > default_x


def test_mobilegaze_missing_repo_reports_clear_message(tmp_path) -> None:
    status = MobileGazeAdapter.check_setup(repo_path=tmp_path / "missing-mobilegaze")

    assert not status.ready
    assert "external/mobilegaze is not set up" in status.reason


def test_mobilegaze_missing_model_does_not_claim_real_backend(tmp_path) -> None:
    repo = tmp_path / "mobilegaze"
    (repo / "models").mkdir(parents=True)
    (repo / "utils").mkdir()
    (repo / "onnx_inference.py").write_text("", encoding="utf-8")
    (repo / "models" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "utils" / "helpers.py").write_text("", encoding="utf-8")

    status = MobileGazeAdapter.check_setup(repo_path=repo, model_path=tmp_path / "missing.onnx")

    assert not status.ready
    assert "ONNX model file not found" in status.reason


def test_mobilegaze_angles_map_to_center_and_edges() -> None:
    center = MobileGazeAdapter._angles_to_raw(0.0, 0.0, yaw_range_deg=45.0, pitch_range_deg=35.0)
    left = MobileGazeAdapter._angles_to_raw(np.radians(-45.0), 0.0, yaw_range_deg=45.0)
    right = MobileGazeAdapter._angles_to_raw(np.radians(45.0), 0.0, yaw_range_deg=45.0)
    top = MobileGazeAdapter._angles_to_raw(0.0, np.radians(35.0), pitch_range_deg=35.0)
    bottom = MobileGazeAdapter._angles_to_raw(0.0, np.radians(-35.0), pitch_range_deg=35.0)

    assert center == (0.5, 0.5)
    assert left[0] == 0.0
    assert right[0] == 1.0
    assert top[1] == 0.0
    assert bottom[1] == 1.0


def test_mobilegaze_raw_median_filter_rejects_single_frame_outlier() -> None:
    gaze_filter = _RawGazeMedianFilter(window_size=5)

    for sample in ((0.50, 0.50), (0.51, 0.49), (0.95, 0.05), (0.49, 0.51)):
        filtered = gaze_filter.update(*sample)

    assert filtered == (0.505, 0.495)
    assert gaze_filter.sample_count == 4


def test_mobilegaze_raw_median_filter_resets_after_missing_detection() -> None:
    gaze_filter = _RawGazeMedianFilter(window_size=4)
    gaze_filter.update(0.20, 0.80)
    gaze_filter.update(0.25, 0.75)

    gaze_filter.reset()

    assert gaze_filter.window_size == 5
    assert gaze_filter.update(0.70, 0.30) == (0.70, 0.30)
    assert gaze_filter.sample_count == 1


def test_mobilegaze_backend_config_uses_backend_model_path_when_gaze_path_empty() -> None:
    config = build_gaze_backend_config(
        {
            "gaze": {"gaze_backend": "mobilegaze", "backend": "mobilegaze", "model_path": None},
            "mobilegaze": {
                "model_path": "external/mobilegaze/weights/mobileone_s0_gaze.onnx",
            },
        }
    )

    assert config["model_path"] == "external/mobilegaze/weights/mobileone_s0_gaze.onnx"


def test_legacy_gaze_adapters_module_reexports_real_adapters() -> None:
    from visimove.gaze import adapters as gaze_adapters

    assert gaze_adapters.EyeTraxAdapter is EyeTraxAdapter
    assert gaze_adapters.GazeFollowerAdapter is GazeFollowerAdapter
    assert gaze_adapters.MobileGazeAdapter is MobileGazeAdapter
