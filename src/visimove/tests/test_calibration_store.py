from visimove.calibration import CalibrationPoint, CalibrationSample
from visimove.calibration import load_calibration_profile, new_profile, save_calibration_profile


def test_save_load_calibration_profile() -> None:
    points = [
        CalibrationPoint(
            index=0,
            normalized_x=0.5,
            normalized_y=0.5,
            screen_x=960,
            screen_y=540,
        )
    ]
    samples = [
        CalibrationSample(
            raw_gaze=(0.49, 0.52),
            target_screen=(960, 540),
            timestamp=123.0,
            confidence=0.91,
        )
    ]
    profile = new_profile(
        screen_width=1920,
        screen_height=1080,
        camera_index=0,
        calibration_points=points,
        samples=samples,
        mapping_model_type="linear",
        mapping_parameters={"model_type": "linear", "coefficients": [[1, 0], [2, 0], [0, 2]]},
        gaze_backend="dummy",
        detector_backend="opencv",
        blink_backend="dummy",
        calibration_point_layout="1-point",
        backend_metadata={"provider": "test"},
        sample_count_per_point=[1],
        confidence_statistics={"min": 0.91, "mean": 0.91, "max": 0.91},
        calibration_quality="good",
        calibration_warnings=[],
        quality_metrics={"valid_samples": 1},
    )

    path = "data/calibration/test_user_profile.json"
    try:
        save_calibration_profile(profile, path)
        loaded = load_calibration_profile(path)

        assert loaded.screen_width == 1920
        assert loaded.camera_index == 0
        assert loaded.calibration_points[0].screen_x == 960
        assert loaded.raw_gaze_samples[0].raw_gaze == (0.49, 0.52)
        assert loaded.raw_gaze_samples[0].confidence == 0.91
        assert loaded.target_screen_coordinates == [(960, 540)]
        assert loaded.gaze_backend == "dummy"
        assert loaded.detector_backend == "opencv"
        assert loaded.blink_backend == "dummy"
        assert loaded.calibration_point_layout == "1-point"
        assert loaded.backend_metadata == {"provider": "test"}
        assert loaded.sample_count_per_point == [1]
        assert loaded.confidence_statistics == {"min": 0.91, "mean": 0.91, "max": 0.91}
        assert loaded.calibration_quality == "good"
        assert loaded.calibration_warnings == []
        assert loaded.quality_metrics == {"valid_samples": 1}
    finally:
        from pathlib import Path

        Path(path).unlink(missing_ok=True)
