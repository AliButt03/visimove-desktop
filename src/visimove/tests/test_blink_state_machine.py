from visimove.blink import BlinkStateMachine, BlinkStateMachineConfig, ClickEvent


def make_machine(**overrides: object) -> BlinkStateMachine:
    config = BlinkStateMachineConfig(**overrides)
    return BlinkStateMachine(config)


def test_short_blink_is_ignored() -> None:
    machine = make_machine()

    assert machine.update(0.9, 1.00) is None
    assert machine.update(0.1, 1.08) is None
    assert machine.update(0.1, 2.00) is None


def test_valid_blink_produces_left_click_after_double_gap() -> None:
    machine = make_machine(double_blink_gap_ms=450)

    assert machine.update(0.9, 1.00) is None
    assert machine.update(0.1, 1.20) is None

    assert machine.update(0.1, 1.66) is ClickEvent.LEFT_CLICK


def test_long_blink_produces_configured_long_blink_event() -> None:
    machine = make_machine(long_blink_event=ClickEvent.PAUSE_TOGGLE)

    assert machine.update(0.9, 1.00) is None
    assert machine.update(0.9, 1.50) is None

    assert machine.update(0.1, 1.85) is ClickEvent.PAUSE_TOGGLE


def test_double_blink_produces_double_click() -> None:
    machine = make_machine(double_blink_gap_ms=450)

    assert machine.update(0.9, 1.00) is None
    assert machine.update(0.1, 1.20) is None
    assert machine.update(0.9, 1.45) is None

    assert machine.update(0.1, 1.60) is ClickEvent.DOUBLE_CLICK


def test_cooldown_blocks_repeated_clicks() -> None:
    machine = make_machine(click_cooldown_ms=1000)

    assert machine.update(0.9, 1.00) is None
    assert machine.update(0.1, 1.20) is None
    assert machine.update(0.1, 1.70) is ClickEvent.LEFT_CLICK

    assert machine.update(0.9, 1.80) is None
    assert machine.update(0.1, 2.00) is None
    assert machine.update(0.1, 2.50) is None


def test_repeated_closed_frames_do_not_repeatedly_click() -> None:
    machine = make_machine()

    assert machine.update(0.9, 1.00) is None
    assert machine.update(0.95, 1.20) is None
    assert machine.update(0.99, 1.40) is None
    assert machine.update(0.90, 1.60) is None

    assert machine.update(0.1, 1.85) is ClickEvent.RIGHT_CLICK
