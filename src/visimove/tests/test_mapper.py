from visimove.calibration import CalibrationMapper, CalibrationProfile, RawCalibrationDomain
from visimove.types import GazeEstimate, Point, ScreenPoint


def test_mapper_clamps_normalized_gaze() -> None:
    mapper = CalibrationMapper(screen_width=100, screen_height=50)
    mapped = mapper.map_to_screen(GazeEstimate(point=Point(2.0, -1.0), confidence=1.0))
    assert mapped == ScreenPoint(99, 0)


def test_mapper_loads_regression_model_from_profile() -> None:
    profile = CalibrationProfile(
        timestamp="2026-05-14T00:00:00+00:00",
        screen_width=100,
        screen_height=50,
        camera_index=0,
        calibration_points=[],
        raw_gaze_samples=[],
        target_screen_coordinates=[],
        mapping_model_type="linear",
        mapping_parameters={"model_type": "linear", "coefficients": [[10, 5], [20, 0], [0, 30]]},
        gaze_backend="dummy",
        detector_backend="opencv",
        blink_backend="dummy",
        calibration_point_layout="9-point",
        backend_metadata={},
    )

    mapper = CalibrationMapper.from_profile(profile)
    mapped = mapper.map_to_screen(GazeEstimate(point=Point(0.5, 0.5), confidence=1.0))

    assert mapped == ScreenPoint(20, 20)


def test_raw_y_below_calibration_min_is_detected() -> None:
    mapper = CalibrationMapper(
        raw_domain=RawCalibrationDomain(raw_x_min=0.2, raw_x_max=0.8, raw_y_min=0.3, raw_y_max=0.9)
    )

    check = mapper.check_raw_domain(0.5, 0.1)

    assert check.status == "outside"
    assert check.violations == ("raw_y_below_min",)


def test_raw_input_clamping_prevents_negative_mapped_y() -> None:
    profile = CalibrationProfile(
        timestamp="2026-05-15T00:00:00+00:00",
        screen_width=1000,
        screen_height=1000,
        camera_index=0,
        calibration_points=[],
        raw_gaze_samples=[],
        target_screen_coordinates=[],
        mapping_model_type="affine",
        mapping_parameters={
            "model_type": "affine",
            "coefficients": [[-500.0, -400.0], [1000.0, 0.0], [0.0, 2000.0]],
        },
        raw_x_min=0.2,
        raw_x_max=0.8,
        raw_y_min=0.3,
        raw_y_max=0.9,
        raw_x_range=0.6,
        raw_y_range=0.6,
    )
    mapper = CalibrationMapper.from_profile(profile)
    mapper.clamp_raw_input_to_calibration_domain = True
    mapper.raw_domain_margin = 0.05

    mapped, debug = mapper.map_with_debug(GazeEstimate(point=Point(0.5, 0.0), confidence=1.0))

    assert debug.raw_domain_status == "outside"
    assert debug.raw_domain_violations == ("raw_y_below_min",)
    assert debug.mapped_input_y == 0.3
    assert debug.before_clamp_y == 200.0
    assert mapped.y == 200


def test_raw_domain_margin_is_detection_tolerance_not_mapping_input() -> None:
    mapper = CalibrationMapper(
        raw_domain=RawCalibrationDomain(raw_x_min=0.2, raw_x_max=0.8, raw_y_min=0.3, raw_y_max=0.9),
        clamp_raw_input_to_calibration_domain=True,
        raw_domain_margin=0.05,
    )

    check = mapper.check_raw_domain(0.5, 0.28)

    assert check.status == "inside"
    assert check.violations == ()
    assert check.mapped_input_y == 0.3
