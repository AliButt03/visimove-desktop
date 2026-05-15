from visimove.calibration import generate_calibration_points


def test_generates_five_calibration_points() -> None:
    points = generate_calibration_points("5", screen_width=1000, screen_height=500, margin=0.1)

    assert len(points) == 5
    assert (points[0].normalized_x, points[0].normalized_y) == (0.5, 0.5)
    assert (points[0].screen_x, points[0].screen_y) == (500, 250)


def test_generates_nine_calibration_points() -> None:
    points = generate_calibration_points("9", screen_width=100, screen_height=100, margin=0.1)

    assert len(points) == 9
    assert points[0].screen_x == 10
    assert points[-1].screen_y == 89


def test_nine_point_order_is_row_major() -> None:
    points = generate_calibration_points("9", screen_width=1000, screen_height=800)
    coordinates = [(point.normalized_x, point.normalized_y) for point in points]

    assert coordinates == [
        (0.12, 0.12),
        (0.5, 0.12),
        (0.88, 0.12),
        (0.12, 0.5),
        (0.5, 0.5),
        (0.88, 0.5),
        (0.12, 0.88),
        (0.5, 0.88),
        (0.88, 0.88),
    ]


def test_generates_sixteen_calibration_points() -> None:
    points = generate_calibration_points("16", screen_width=1600, screen_height=900)

    assert len(points) == 16
    assert len({point.normalized_x for point in points}) == 4
    assert len({point.normalized_y for point in points}) == 4
