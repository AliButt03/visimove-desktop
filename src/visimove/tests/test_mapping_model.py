from visimove.calibration import MappingModelType, create_mapping_model
from visimove.calibration.mapping_model import RegressionMappingModel


def test_linear_mapping_fit_predict() -> None:
    model = create_mapping_model(MappingModelType.LINEAR)
    raw = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    targets = [(10.0, 20.0), (110.0, 20.0), (10.0, 220.0), (110.0, 220.0)]

    model.fit(raw, targets)
    prediction = model.predict((0.5, 0.5))

    assert round(prediction.x) == 60
    assert round(prediction.y) == 120


def test_ridge_mapping_serializes_parameters() -> None:
    model = create_mapping_model(MappingModelType.RIDGE, ridge_alpha=0.1)
    raw = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    targets = [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)]

    model.fit(raw, targets)
    restored = RegressionMappingModel.from_parameters(model.to_parameters())

    prediction = restored.predict((1.0, 1.0))
    assert prediction.x > 80
    assert prediction.y > 80


def test_polynomial_mapping_fit_predict() -> None:
    model = create_mapping_model(MappingModelType.POLYNOMIAL)
    raw = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5), (0.25, 0.75)]
    targets = [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0), (50.0, 50.0), (25.0, 75.0)]

    model.fit(raw, targets)
    prediction = model.predict((0.25, 0.75))

    assert round(prediction.x) == 25
    assert round(prediction.y) == 75


def test_affine_mapping_predicts_varied_y_for_varied_raw_y() -> None:
    model = create_mapping_model("affine")
    model.fit(
        raw_gaze=[(0.1, 0.2), (0.9, 0.2), (0.1, 0.9), (0.9, 0.9)],
        targets=[(100.0, 100.0), (900.0, 100.0), (100.0, 900.0), (900.0, 900.0)],
    )

    top = model.predict((0.5, 0.2))
    bottom = model.predict((0.5, 0.9))

    assert round(top.y) == 100
    assert round(bottom.y) == 900
    assert bottom.y > top.y


def test_idw_mapping_is_bounded_by_control_targets() -> None:
    model = create_mapping_model("idw")
    model.fit(
        raw_gaze=[(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)],
        targets=[(100.0, 100.0), (900.0, 100.0), (100.0, 900.0), (900.0, 900.0)],
    )

    exact = model.predict((0.2, 0.2))
    prediction = model.predict((-10.0, 10.0))

    assert round(exact.x) == 100
    assert round(exact.y) == 100
    assert 100 <= prediction.x <= 900
    assert 100 <= prediction.y <= 900


def test_idw_mapping_serializes_control_points() -> None:
    model = create_mapping_model(MappingModelType.IDW)
    model.fit(
        raw_gaze=[(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)],
        targets=[(100.0, 100.0), (900.0, 100.0), (100.0, 900.0), (900.0, 900.0)],
    )

    restored = RegressionMappingModel.from_parameters(model.to_parameters())
    prediction = restored.predict((0.8, 0.8))

    assert restored.model_type == MappingModelType.IDW
    assert round(prediction.x) == 900
    assert round(prediction.y) == 900


def test_grid_mapping_interpolates_columns_and_rows() -> None:
    model = create_mapping_model("grid")
    raw = [
        (0.2, 0.3),
        (0.5, 0.3),
        (0.8, 0.3),
        (0.2, 0.6),
        (0.5, 0.6),
        (0.8, 0.6),
        (0.2, 0.9),
        (0.5, 0.9),
        (0.8, 0.9),
    ]
    targets = [
        (100.0, 100.0),
        (500.0, 100.0),
        (900.0, 100.0),
        (100.0, 500.0),
        (500.0, 500.0),
        (900.0, 500.0),
        (100.0, 900.0),
        (500.0, 900.0),
        (900.0, 900.0),
    ]

    model.fit(raw, targets)

    center = model.predict((0.5, 0.6))
    beyond_bottom_right = model.predict((1.0, 1.0))

    assert round(center.x) == 500
    assert round(center.y) == 500
    assert round(beyond_bottom_right.x) == 900
    assert round(beyond_bottom_right.y) == 900


def test_grid_mapping_serializes_knots() -> None:
    model = create_mapping_model(MappingModelType.GRID)
    model.fit(
        raw_gaze=[
            (0.2, 0.3),
            (0.5, 0.3),
            (0.8, 0.3),
            (0.2, 0.9),
            (0.5, 0.9),
            (0.8, 0.9),
        ],
        targets=[
            (100.0, 100.0),
            (500.0, 100.0),
            (900.0, 100.0),
            (100.0, 900.0),
            (500.0, 900.0),
            (900.0, 900.0),
        ],
    )

    restored = RegressionMappingModel.from_parameters(model.to_parameters())
    prediction = restored.predict((0.8, 0.9))

    assert restored.model_type == MappingModelType.GRID
    assert round(prediction.x) == 900
    assert round(prediction.y) == 900
