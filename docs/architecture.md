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
  native_output_mode: model_coordinates
  native_coordinate_scale_x: 10.0
  native_coordinate_scale_y: 10.0
```

GazeFollower's external `raw_gaze_coordinates` are the first two values of the MNN model output. They are not Windows screen pixels. The adapter exposes them as `native_gaze` with `native_units=gazefollower_model_coordinates` and maps them into VisiMove's 0..1 raw space with a configurable centered transform before any VisiMove calibration.

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

mobilegaze:
  repo_path: external/mobilegaze
  model_path: external/mobilegaze/weights/mobileone_s0_gaze.onnx
  yaw_range_deg: 45.0
  pitch_range_deg: 35.0
  face_crop_scale: 1.15
  providers:
    - CPUExecutionProvider
```

MobileGaze uses ONNX Runtime and VisiMove's detector face crop. Its native output is `(yaw, pitch)` in radians. The adapter exposes that as `native_gaze` with `native_units=radians_yaw_pitch`, plus a 3D `gaze_vector`, then maps the angles into normalized raw gaze for preview and VisiMove calibration.

EyeTrax is the first real gaze backend target. Its adapter validates `external/eyetrax`, a saved `gaze_model.pkl`, importable dependencies, and a local MediaPipe `face_landmarker.task` before claiming the backend is ready. If setup is incomplete and `fallback_to_dummy` is true, the pipeline prints a clear message and falls back to `MovingDummyGazeModel` for preview/testing only.

When fallback happens, cursor movement remains blocked unless the explicit dummy testing override is used. Real cursor control should wait until EyeTrax preview/debug output is stable and a matching calibration profile exists.

## Adapter Rules

- Do not import external repo code outside adapter modules.
- Do not copy external repo files into `src/visimove/`.
- Validate repo and model paths before inference.
- Return `GazeResult` from every gaze backend.
- Keep exact external integration TODOs inside the adapter file until dependencies are approved.
