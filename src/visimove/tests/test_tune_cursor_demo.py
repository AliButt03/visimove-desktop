from __future__ import annotations

from scripts import tune_cursor_demo


def test_preview_target_wait_pumps_events_until_start(monkeypatch) -> None:
    draw_calls: list[str] = []
    keys = iter((255, 32))
    monkeypatch.setattr(
        tune_cursor_demo,
        "draw_preview",
        lambda **kwargs: draw_calls.append(str(kwargs["phase"])),
    )
    monkeypatch.setattr(tune_cursor_demo.cv2, "waitKey", lambda _delay: next(keys))

    started = tune_cursor_demo.wait_for_target_start(
        name="center",
        target_x=1280,
        target_y=720,
        screen_width=2560,
        screen_height=1440,
        show_preview=True,
    )

    assert started is True
    assert draw_calls == ["ready", "ready"]


def test_preview_target_wait_can_cancel(monkeypatch) -> None:
    monkeypatch.setattr(tune_cursor_demo, "draw_preview", lambda **_kwargs: None)
    monkeypatch.setattr(tune_cursor_demo.cv2, "waitKey", lambda _delay: 27)

    started = tune_cursor_demo.wait_for_target_start(
        name="top_left",
        target_x=0,
        target_y=0,
        screen_width=2560,
        screen_height=1440,
        show_preview=True,
    )

    assert started is False