# Blink Click

VisiMove treats blink detection as a probability problem. A blink model estimates how likely each eye is closed, then the blink state machine turns that probability stream into click events.

## Blink Probability

Blink backends return:

- left eye closed probability
- right eye closed probability
- combined closed probability
- model confidence
- inference time in milliseconds

The real-time pipeline uses the combined closed probability for click decisions. ONNX Runtime is supported when installed and when a valid model path is configured. If the model file or runtime is missing, VisiMove falls back to the dummy blink model so the app can still run safely.

## Thresholds

Blink thresholds are configured in `config/default.yaml`:

```yaml
blink:
  closed_threshold: 0.65
  min_blink_ms: 150
  max_blink_ms: 600
  long_blink_ms: 800
  double_blink_gap_ms: 450
  click_cooldown_ms: 1000
```

`closed_threshold` decides when the eyes count as closed. A closure shorter than `min_blink_ms` is ignored. A closure between `min_blink_ms` and `max_blink_ms` can become a left click. A closure longer than `long_blink_ms` can become the configured long-blink action, such as right click or pause toggle.

## Cooldown

`click_cooldown_ms` prevents repeated clicks immediately after a valid blink event. This is separate from cursor click cooldown and exists to suppress repeated blink-triggered actions.

## False-Click Prevention

VisiMove does not click on every closed-eye frame. The blink state machine waits for a closed-to-open transition, checks duration, checks cooldown, and only then emits a click event.

False-click prevention includes:

- ignoring very short closures
- ignoring ambiguous closures between normal and long blink windows
- waiting for the double-blink window before finalizing a single left click
- suppressing repeated events while eyes remain closed
- freezing blink clicks when no face or eye crops are available
- falling back to dummy inference if optional model backends are unavailable
