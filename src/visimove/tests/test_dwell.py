from visimove.cursor.dwell import DwellSelector
from visimove.types import ScreenPoint


def test_dwell_fires_once_until_cursor_leaves_anchor() -> None:
    dwell = DwellSelector(dwell_time_ms=900, radius_px=35)
    point = ScreenPoint(100, 100)

    assert dwell.update(point, timestamp=1.0) is False
    assert dwell.update(point, timestamp=1.8) is False
    assert dwell.update(point, timestamp=1.9) is True
    assert dwell.update(point, timestamp=3.0) is False

    assert dwell.update(ScreenPoint(200, 100), timestamp=3.1) is False
    assert dwell.update(ScreenPoint(200, 100), timestamp=4.0) is True


def test_dwell_reset_requires_a_new_full_fixation() -> None:
    dwell = DwellSelector(dwell_time_ms=900, radius_px=35)
    point = ScreenPoint(100, 100)

    dwell.update(point, timestamp=1.0)
    dwell.reset()

    assert dwell.update(point, timestamp=2.0) is False
    assert dwell.update(point, timestamp=2.8) is False
    assert dwell.update(point, timestamp=2.9) is True
