# Architecture

VisiMove is organized as a real-time pipeline:

1. Camera capture reads frames.
2. Detector finds face and eye landmarks or regions.
3. Gaze model estimates normalized gaze.
4. Calibration mapper converts gaze to screen coordinates.
5. Smoothing filter stabilizes movement.
6. Cursor controller moves and clicks.
7. Performance monitor tracks latency and FPS.

External repositories stay in `external/` and are accessed only through adapters.

## Model Adapters

VisiMove keeps first-party application code independent from external research repositories. The real-time pipeline depends on stable internal interfaces only:

- `BaseDetector`
- `BaseGazeModel`
- `BaseBlinkModel`
- cursor controller interface

External code is loaded only inside adapter modules selected by config.

## Gaze Adapters

Standard gaze output is `GazeResult`:

- `raw_x`
- `raw_y`
- optional `screen_x`
- optional `screen_y`
- optional `gaze_vector`
- `confidence`
- `inference_time_ms`
- `metadata`

Adapters live in:

```text
src/visimove/gaze/gazefollower_adapter.py
src/visimove/gaze/eyetrax_adapter.py
src/visimove/gaze/mobilegaze_adapter.py
```

External repositories stay isolated:

```text
external/gazefollower/
external/eyetrax/
external/mobilegaze/
```

Expected model paths:

```text
external/gazefollower/gazefollower/res/model_weights/base.mnn
external/gazefollower/gazefollower/res/model_weights/blaze_face.mnn
models/gaze/eyetrax/gaze_model.pkl
models/gaze/mobilegaze/*.onnx
```

Config examples:

```yaml
gaze:
  gaze_backend: dummy
```

```yaml
gaze:
  gaze_backend: gazefollower
  fallback_to_dummy: true

gazefollower:
  repo_path: external/gazefollower
  model_path: external/gazefollower/gazefollower/res/model_weights/base.mnn
  face_model_path: external/gazefollower/gazefollower/res/model_weights/blaze_face.mnn
  face_alignment_backend: blazeface
```

```yaml
gaze:
  gaze_backend: eyetrax
  backend: eyetrax
  fallback_to_dummy: true

eyetrax:
  repo_path: external/eyetrax
  model_path: models/gaze/eyetrax
  face_landmarker_model_path: models/detection/face_landmarker.task
```

```yaml
gaze:
  gaze_backend: mobilegaze
  model_path: models/gaze/mobilegaze/model.onnx
```

EyeTrax is the first real gaze backend target. Its adapter validates `external/eyetrax`, a saved `gaze_model.pkl`, importable dependencies, and a local MediaPipe `face_landmarker.task` before claiming the backend is ready. If setup is incomplete and `fallback_to_dummy` is true, the pipeline prints a clear message and falls back to `MovingDummyGazeModel` for preview/testing only.

When fallback happens, cursor movement remains blocked unless the explicit dummy testing override is used. Real cursor control should wait until EyeTrax preview/debug output is stable and a matching calibration profile exists.

## Adapter Rules

- Do not import external repo code outside adapter modules.
- Do not copy external repo files into `src/visimove/`.
- Validate repo and model paths before inference.
- Return `GazeResult` from every gaze backend.
- Keep exact external integration TODOs inside the adapter file until dependencies are approved.
