# VisiMove Test Suite

This file is the living record of VisiMove tests and verification commands. Add every new automated test, manual check, and important command result here as the project grows.

## How To Run

From the project root:

```powershell
cd C:\Users\MNA\Desktop\FYP-backend\visimove-desktop
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

The `-p no:cacheprovider` option avoids pytest cache folders on this Windows setup.

## Current Automated Suite

### Blink

File: `src/visimove/tests/test_blink_models.py`

- `test_dummy_blink_model_returns_probability_result`
- `test_missing_onnx_model_falls_back_to_dummy`
- `test_onnx_preprocess_eye_crop_resizes_and_normalizes`
- `test_empty_blink_result_is_safe_for_missing_crops`

Coverage:

- Blink model output includes left, right, combined closed probability, confidence, and inference timing fields.
- Missing ONNX model paths fall back to the dummy blink model.
- ONNX eye crop preprocessing resizes and normalizes input.
- Missing or invalid blink inputs can safely produce an empty probability result.

File: `src/visimove/tests/test_blink_state_machine.py`

- `test_short_blink_is_ignored`
- `test_valid_blink_produces_left_click_after_double_gap`
- `test_long_blink_produces_configured_long_blink_event`
- `test_double_blink_produces_double_click`
- `test_cooldown_blocks_repeated_clicks`
- `test_repeated_closed_frames_do_not_repeatedly_click`

Coverage:

- Very short blinks are ignored.
- Valid blinks emit left-click events only after the double-blink window expires.
- Long blinks emit the configured long-blink event.
- Double blinks emit double-click events.
- Cooldown prevents repeated click events.
- Holding eyes closed does not repeatedly click.

### Calibration

File: `src/visimove/tests/test_calibration_points.py`

- `test_generates_five_calibration_points`
- `test_generates_nine_calibration_points`
- `test_nine_point_order_is_row_major`
- `test_generates_sixteen_calibration_points`

Coverage:

- 5-point, 9-point, and 16-point calibration point generation.
- Normalized point layout.
- 9-point row-major target order.
- Screen coordinate conversion.

File: `src/visimove/tests/test_mapping_model.py`

- `test_linear_mapping_fit_predict`
- `test_ridge_mapping_serializes_parameters`
- `test_polynomial_mapping_fit_predict`
- `test_affine_mapping_predicts_varied_y_for_varied_raw_y`
- `test_idw_mapping_is_bounded_by_control_targets`
- `test_idw_mapping_serializes_control_points`

Coverage:

- Linear regression fit and prediction.
- Ridge regression parameter serialization and restore.
- Polynomial regression fit and prediction.
- Affine mapping preserves Y variation for varied raw Y values.
- IDW mapping predicts exact calibration controls and stays bounded by calibration target points.

File: `src/visimove/tests/test_mapping_diagnostics.py`

- `test_diagnostics_detects_clamping_to_top_edge`
- `test_calibration_point_means_keep_samples_paired_with_targets`

Coverage:

- Calibration diagnostics detect negative pre-clamp Y, top-edge clamping, and mapped Y stuck at zero.
- Calibration sample means stay paired with their exact target coordinates.

File: `src/visimove/tests/test_live_quality.py`

- `test_live_domain_violation_ratio_is_calculated`
- `test_live_quality_becomes_unsafe_when_violations_cross_threshold`

Coverage:

- Live calibration-domain violation ratios are calculated over a rolling window.
- Live tracking quality becomes `unsafe` when violations cross the configured threshold.

File: `src/visimove/tests/test_realtime_live_safety.py`

- `test_cursor_blocked_when_live_tracking_quality_is_unsafe`
- `test_cursor_allowed_when_live_tracking_quality_is_stable`
- `test_unstable_live_gaze_override_allows_cursor`

Coverage:

- Real cursor movement is blocked when live gaze leaves the calibrated domain.
- Stable live gaze allows cursor safety to proceed.
- `--allow-unstable-live-gaze` can override the live-domain block for careful testing.

File: `src/visimove/tests/test_axis_adjustment.py`

- `test_horizontal_gain_expands_x_around_screen_center`
- `test_offset_shifts_x_and_y`
- `test_gain_output_is_clamped_to_screen_bounds`
- `test_adjustment_result_populates_debug_fields`

Coverage:

- Horizontal gain expands mapped X around the screen center.
- Pixel offsets shift adjusted X/Y.
- Gain-adjusted output is clamped to screen bounds.
- Axis adjustment debug fields are populated for tracking diagnostics.

File: `src/visimove/tests/test_calibration_store.py`

- `test_save_load_calibration_profile`

Coverage:

- Calibration profile JSON save/load.
- Screen size, camera index, calibration points, raw gaze samples, confidence values, target screen coordinates, backend metadata, and calibration quality diagnostics persist correctly.

File: `src/visimove/tests/test_calibration_quality.py`

- `test_quality_detects_pinned_eyetrax_raw_gaze`
- `test_quality_detects_low_sample_count`
- `test_eyetrax_profile_quality_metadata_can_be_saved`

Coverage:

- Pinned EyeTrax-style raw gaze near `raw_x=1.0` and `raw_y=0.0` is marked for review.
- Low sample count and missing calibration points produce warnings.
- Quality reports include sample counts, confidence statistics, and raw-gaze metrics for saved profiles.

### Cursor Safety

File: `src/visimove/tests/test_cursor_safety.py`

- `test_screen_bounds_clamps_coordinates`
- `test_paused_controller_ignores_movement_and_clicks`
- `test_resumed_controller_clamps_movement`
- `test_low_gaze_confidence_freezes_cursor`
- `test_click_cooldown_blocks_immediate_second_click`

Coverage:

- Coordinate clamping to screen bounds.
- Paused cursor ignores movement and clicks.
- Resumed cursor accepts safe movement.
- Low gaze confidence freezes movement.
- Click cooldown blocks immediate repeated clicks.

### Detection

File: `src/visimove/tests/test_detection.py`

- `test_dummy_detector_returns_face_eye_boxes_and_crops`
- `test_safe_crop_clamps_to_frame_bounds`
- `test_clamp_box_rejects_empty_box`
- `test_opencv_detector_handles_no_face_safely`
- `test_mediapipe_tasks_missing_model_falls_back_to_opencv`
- `test_mediapipe_normalized_landmarks_convert_to_pixels`
- `test_auto_detector_uses_safe_available_backend`

Coverage:

- Detector result includes face box, eye boxes, eye crops, landmarks compatibility, and confidence.
- Eye crops are safely clamped to frame bounds.
- Invalid boxes are rejected.
- OpenCV detector safely returns no-face results on blank frames.
- MediaPipe Tasks installs without a configured `.task` model fall back to OpenCV.
- MediaPipe normalized landmarks convert to pixel coordinates.
- Auto detector mode selects a safe available backend without noisy startup warnings.

File: `src/visimove/tests/test_yolo_detector.py`

- `test_missing_yolo_weights_falls_back_to_opencv_or_mediapipe`
- `test_yolo_detector_returns_detector_result_and_crops`
- `test_yolo_detector_reuses_previous_result_between_detections`

Coverage:

- Optional YOLO backend falls back when weights/dependencies are unavailable.
- YOLO class outputs map to `face`, `left_eye`, and `right_eye`.
- YOLO returns the standard `DetectorResult` format with eye crops.
- YOLO detection can run every N frames and reuse previous boxes between runs.

### Gaze

File: `src/visimove/tests/test_gaze_adapters.py`

- `test_gaze_result_exposes_mapper_compatible_point`
- `test_dummy_gaze_model_returns_standard_gaze_result`
- `test_missing_external_gaze_model_falls_back_to_dummy`
- `test_gazefollower_adapter_requires_model_path`

Coverage:

- Standard `GazeResult` exposes mapper-compatible normalized point output.
- Dummy gaze backend returns the standard gaze result contract.
- Missing external gaze setup falls back to the dummy model.
- GazeFollower adapter validates external setup/model path before use.

### Smoothing

File: `src/visimove/tests/test_smoothing.py`

- `test_exponential_smoothing_starts_at_first_point`
- `test_exponential_smoothing_blends_points`
- `test_ema_smoothing_blends_points`
- `test_deadzone_holds_small_movements`
- `test_large_jump_is_clamped`
- `test_missing_gaze_reuses_last_value`
- `test_reset_clears_filter_state`

Coverage:

- Exponential smoothing initializes with the first point.
- Later points are blended by alpha.
- EMA smoothing reduces jitter by blending toward new gaze points.
- Deadzone filtering holds output for small movements inside the configured radius.
- Impossible large jumps are clamped by `max_jump_pixels`.
- Missing gaze values reuse the last stable output.
- Reset clears filter state.

### Mapper

File: `src/visimove/tests/test_mapper.py`

- `test_mapper_clamps_normalized_gaze`

Coverage:

- Normalized gaze coordinates are clamped before mapping to screen coordinates.

## Verification Log

### 2026-05-04

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Result:

```text
21 passed in 0.50s
```

## 2026-05-15 - Calibration Mapping Diagnostics And Affine EyeTrax Mapping

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

Result:

```text
passed
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

```text
71 passed in 3.62s
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_calibration_mapping.py --profile data/calibration/user_profile_eyetrax.json
```

Result:

```text
diagnostic ran; current saved profile uses ridge and has one calibration mean with negative Y before clamp
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_calibration.py --help
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

```text
passed
```

Added coverage:

- Affine mapping model.
- Calibration mapping diagnostics.
- Quality gating for mapped Y stuck at zero.
- Cursor gate allows good/acceptable calibration and blocks low-quality calibration.

## 2026-05-15 - Live Calibration Domain Safety

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

Result:

```text
passed
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

```text
79 passed in 4.21s
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_calibration_mapping.py --profile data/calibration/user_profile_eyetrax.json
```

Result:

```text
diagnostic ran; affine profile has healthy mapped ranges and no diagnostic warnings
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend eyetrax --show-debug --no-preview
```

Result:

```text
smoke run timed out by design after printing startup/debug output; cursor was disabled
```

Added coverage:

- Raw gaze calibration domain storage/load.
- Runtime raw-domain violation detection.
- Mapping input clamping to calibrated raw min/max, with margin used only for violation-detection tolerance.
- Live tracking quality monitor.
- Cursor block for unsafe live gaze.

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

```text
OK - tracking CLI help displayed.
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_calibration.py --help
```

Result:

```text
OK - calibration CLI help displayed.
```

Previously performed checks:

- `python -m compileall -q src scripts` passed after pipeline, cursor, blink, and calibration implementation steps.
- Venv import check passed for `cv2`, `numpy`, `yaml`, `pydantic`, `pytest`, and `visimove`.
- Tracking pipeline import/build check passed for `RealtimeVisiMovePipeline`.
- Cleanup checks removed generated `__pycache__`, pytest temp folders, and editable-install metadata outside `.venv`.

### 2026-05-07

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

Result:

```text
OK - source and scripts compiled.
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Result:

```text
26 passed in 1.76s
```

Added coverage:

- EMA gaze smoothing.
- Deadzone gaze smoothing.
- Missing gaze value handling.
- Large jump clamping.
- Smoothing reset behavior.

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Result:

```text
30 passed in 1.00s
```

Added coverage:

- Dummy detector face/eye boxes and eye crops.
- OpenCV no-face safety behavior.
- Safe crop and box clamping helpers.

Command:

```powershell
.\.venv\Scripts\python.exe -c "from visimove.config import load_config; from visimove.pipeline.realtime_pipeline import build_detector; cfg=load_config('config/default.yaml'); detector=build_detector(cfg['detection']); print(type(detector).__name__)"
```

Result:

```text
MediaPipe detector unavailable; falling back to OpenCV Haar detector. MediaPipe is not installed.
OpenCvHaarDetector
```

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

Result:

```text
OK - source and scripts compiled.
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Result:

```text
34 passed in 0.97s
```

Added coverage:

- Blink result probability model contract.
- ONNX blink model missing-file fallback.
- ONNX eye crop preprocessing.
- Safe empty blink result behavior.

Command:

```powershell
.\.venv\Scripts\python.exe scripts\setup_external_repos.py
```

Result:

```text
OK - setup script reported gazefollower, eyetrax, mobilegaze, and ocec as present.
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\setup_external_repos.py --clone --force --depth 1
```

Result:

```text
OK - shallow cloned external/gazefollower, external/eyetrax, external/mobilegaze, and external/ocec.
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Result:

```text
34 passed in 0.78s
```

Added coverage:

- External repository setup script dry run.
- External repository shallow clone workflow.
- Main app test suite still passes with external repos present.

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

Result:

```text
OK - source and scripts compiled.
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Result:

```text
38 passed in 0.98s
```

Command:

```powershell
.\.venv\Scripts\python.exe -c "from visimove.config import load_config; from visimove.pipeline.realtime_pipeline import build_gaze_model; cfg=load_config('config/default.yaml'); print(type(build_gaze_model(cfg['gaze'])).__name__)"
```

Result:

```text
MovingDummyGazeModel
```

Added coverage:

- Gaze adapter standard result contract.
- Dummy gaze backend through `BaseGazeModel`.
- External gaze backend fallback behavior.
- Default config gaze backend build check.

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q scripts\train_blink_model_colab.py scripts\train_yolo_detector_colab.py scripts\export_onnx.py
```

Result:

```text
OK - cloud training scripts compiled.
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\train_blink_model_colab.py --help
.\.venv\Scripts\python.exe scripts\train_yolo_detector_colab.py --help
.\.venv\Scripts\python.exe scripts\export_onnx.py --help
```

Result:

```text
OK - cloud training and export CLIs displayed help without starting local training.
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Result:

```text
38 passed in 1.24s
```

Added coverage:

- Cloud training scripts stay importable locally.
- Training scripts require `--run` before starting cloud training.
- Existing runtime suite still passes after cloud-training support changes.

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

Result:

```text
OK - source and scripts compiled.
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Result:

```text
41 passed in 2.36s
```

Command:

```powershell
.\.venv\Scripts\python.exe -c "from visimove.pipeline.realtime_pipeline import build_detector; print(type(build_detector({'detector_backend':'yolo','yolo_model_path':'models/detection/missing.onnx','reuse_frames':0})).__name__)"
```

Result:

```text
YOLO detector unavailable; falling back to MediaPipe/OpenCV. YOLO weights not found: models\detection\missing.onnx
MediaPipe detector unavailable; falling back to OpenCV Haar detector. MediaPipe is not installed.
OpenCvHaarDetector
```

Added coverage:

- Optional YOLO detector backend.
- Missing YOLO dependency/weights fallback path.
- YOLO result parsing and previous-box reuse.

Command:

```powershell
.\.venv\Scripts\python.exe -c "from visimove.config import load_config; from visimove.pipeline.realtime_pipeline import build_detector; cfg=load_config('config/default.yaml'); detector=build_detector(cfg['detection']); print(type(detector).__name__)"
```

Result:

```text
MediaPipe detector unavailable; falling back to OpenCV Haar detector. Installed MediaPipe does not expose the classic solutions.face_mesh API (version: 0.10.35). Falling back to OpenCV.
OpenCvHaarDetector
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Result:

```text
41 passed in 4.74s
```

Added coverage:

- MediaPipe package compatibility check for installs without `solutions.face_mesh`.
- Safe fallback to OpenCV instead of crashing at startup.

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Result:

```text
43 passed in 1.65s
```

Added coverage:

- MediaPipe Tasks fallback path when `face_landmarker.task` is not configured.
- MediaPipe normalized landmark conversion.

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Result:

```text
44 passed in 2.39s
```

Command:

```powershell
.\.venv\Scripts\python.exe -c "from visimove.config import load_config; from visimove.pipeline.realtime_pipeline import build_detector; cfg=load_config('config/default.yaml'); print(cfg['detection']['detector_backend']); print(type(build_detector(cfg['detection'])).__name__)"
```

Result:

```text
auto
OpenCvHaarDetector
```

Added coverage:

- Default detector mode changed to `auto`.
- Auto mode avoids noisy MediaPipe fallback when the installed package needs an unconfigured `.task` file.
- Tracking startup warns when the saved calibration profile is degenerate.

## Future Test Areas

- Webcam-open smoke test with camera index handling.
- Dry-run tracking loop with synthetic camera frames.
- Calibration UI smoke test separated from full-screen manual UI.
- Real gaze sample provider integration tests.
- Cursor backend tests with mocked `pyautogui`.
- Performance regression checks for per-stage latency and FPS.

## 2026-05-15 - Live Gaze Bias Diagnostics And Axis Gain

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

Result:

```text
passed
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

```text
85 passed in 4.28s
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\diagnose_live_gaze.py --help
```

Result:

```text
tracking and live gaze diagnostic CLI help passed
```

Added coverage:

- Axis gain/offset adjustment after calibration and before smoothing.
- CLI overrides for horizontal/vertical gain and offset.
- Guided live gaze diagnostic CLI.

## 2026-05-15 - EyeTrax Root-Cause Trace

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

Result:

```text
passed
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

```text
87 passed in 4.10s
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\trace_eyetrax_pipeline.py --help
```

Result:

```text
EyeTrax trace CLI help passed
```

Added coverage:

- VisiMove 9-point target order.
- Calibration sample-target pairing.
- EyeTrax trace CLI compiles and exposes guided mode.

## 2026-05-14 - Backend Startup Summary And Dummy Cursor Safety

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

Result:

```text
passed
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Result:

```text
47 passed in 4.64s
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

```text
run_tracking.py exposes --gaze-backend, --detector-backend, --blink-backend,
--calibration-profile, --allow-dummy-cursor, and --show-debug.
```

Added coverage:

- Dummy gaze blocks real cursor movement when `--enable-cursor` is used without `--allow-dummy-cursor`.
- Dummy gaze can move the cursor only when both `--enable-cursor` and `--allow-dummy-cursor` are present.
- CLI backend/profile/debug overrides update the runtime config.

## 2026-05-14 - Calibration Loading And Debug Output

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

Result:

```text
passed
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Result:

```text
50 passed in 2.27s
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\run_calibration.py --help
```

Result:

```text
tracking and calibration CLI help passed
```

Added coverage:

- Calibration profiles save/load backend metadata.
- Missing calibration profile blocks cursor movement when cursor is requested.
- Calibration mismatch detection reports dummy gaze, detector mismatch, and screen size mismatch.
- Runtime profile mapper can load saved mapping parameters.

## 2026-05-14 - External Repository Setup System

Command:

```powershell
.\.venv\Scripts\python.exe scripts\setup_external_repos.py --help
```

Result:

```text
passed
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\setup_external_repos.py
```

Result:

```text
passed; printed setup instructions and confirmed eyetrax, gazefollower, mobilegaze, and ocec folders
```

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Result:

```text
50 passed in 2.25s
```

Added coverage:

- External setup script help works.
- External setup script creates/verifies folder structure without downloading weights or dependencies.
- Existing external repos are left unchanged unless explicit clone/force options are used.

## 2026-05-14 - EyeTrax Preview Backend Adapter

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

Result:

```text
passed
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

```text
54 passed in 1.76s
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\run_calibration.py --help
```

Result:

```text
tracking and calibration CLI help passed
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_calibration.py --gaze-backend eyetrax
```

Result:

```text
refused safely because models/gaze/eyetrax/gaze_model.pkl is missing
```

Command:

```powershell
.\.venv\Scripts\python.exe -c "from visimove.config import load_config; from visimove.pipeline.realtime_pipeline import build_gaze_backend_config, build_gaze_model; cfg=load_config('config/default.yaml'); cfg['gaze']['gaze_backend']='eyetrax'; cfg['gaze']['backend']='eyetrax'; model=build_gaze_model(build_gaze_backend_config(cfg)); print(type(model).__name__)"
```

Result:

```text
EyeTrax backend unavailable. EyeTrax model file not found. Expected a calibrated model such as models/gaze/eyetrax/gaze_model.pkl.
Falling back to dummy gaze model.
MovingDummyGazeModel
```

Added coverage:

- EyeTrax missing repo behavior reports a clear setup message.
- EyeTrax missing model does not claim real gaze is active.
- EyeTrax backend selection falls back to dummy when fallback is enabled.
- EyeTrax backend config reads backend-specific model paths.

## 2026-05-14 - EyeTrax Model Preparation Inspection

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q scripts
.\.venv\Scripts\python.exe scripts\prepare_eyetrax_model.py --help
```

Result:

```text
passed
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_eyetrax_model.py
```

Result:

```text
refused safely because models/detection/face_landmarker.task is missing
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

```text
54 passed in 2.29s
```

Inspection findings:

- `external/eyetrax` does not include pretrained gaze weights.
- EyeTrax expects a per-user calibration/training flow to create `gaze_model.pkl`.
- The repo's `eyetrax.app.build_model` command trains and saves the `.pkl`.

## 2026-05-14 - EyeTrax Preparation Finalization

Command:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_eyetrax_model.py --download-face-landmarker --check-only
```

Result:

```text
downloaded models/detection/face_landmarker.task
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pip install scikit-learn
```

Result:

```text
scikit-learn available; scipy and screeninfo are also available
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_eyetrax_model.py --run
```

Result:

```text
created models/gaze/eyetrax/gaze_model.pkl; external EyeTrax command ended with a Windows cp1252 Unicode print error after saving, so wrapper now sets PYTHONIOENCODING=utf-8
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend eyetrax --show-debug --no-preview
```

Result:

```text
gaze_backend=eyetrax fallback=no; cursor_enabled=no; face_found=yes
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_calibration.py --gaze-backend eyetrax
```

Result:

```text
refused safely because VisiMove calibration UI is not yet wired to collect EyeTrax features
```

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\prepare_eyetrax_model.py --check-only
```

Result:

```text
58 passed in 4.59s
FaceLandmarker present: yes
Output model exists: yes
Dependencies available: yes
```

## 2026-05-14 - EyeTrax VisiMove Calibration Wiring

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

Result:

```text
passed
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

```text
63 passed in 4.08s
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_eyetrax_model.py --check-only
```

Result:

```text
FaceLandmarker present: yes
Output model exists: yes
Dependencies available: yes
```

Added coverage:

- EyeTrax VisiMove calibration profiles can store backend metadata and quality diagnostics.
- Pinned raw gaze values are detected.
- Low sample count and missing point coverage are detected.
- Tracking blocks real cursor movement for poor/needs_review calibration quality.
- Tracking preview remains allowed when calibration quality is poor.

## 2026-05-15 - GazeFollower Backend Preparation

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

Result:

```text
passed
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

```text
89 passed in 12.49s
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

```text
passed
```

Command:

```powershell
.\.venv\Scripts\python.exe -c "from visimove.config import load_config; from visimove.pipeline.realtime_pipeline import build_gaze_backend_config, build_gaze_model; cfg=load_config('config/default.yaml'); cfg['gaze']['gaze_backend']='gazefollower'; cfg['gaze']['backend']='gazefollower'; model=build_gaze_model(build_gaze_backend_config(cfg)); print(type(model).__name__)"
```

Result:

```text
GazeFollower backend unavailable. GazeFollower dependencies are missing: MNN, pygame.
Falling back to dummy gaze model.
MovingDummyGazeModel
```

Added coverage:

- GazeFollower missing repo reports a clear setup message.
- GazeFollower missing MNN model does not claim a real backend.
- GazeFollower backend config uses the backend-specific bundled `.mnn` model path.
- GazeFollower falls back to dummy safely when optional dependencies are missing.

## 2026-05-15 - GazeFollower Dependency Install

Command:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Result:

```text
passed; MNN 3.5.0 and pygame 2.6.1 are installed
```

Command:

```powershell
.\.venv\Scripts\python.exe -c "import MNN, pygame; print('MNN ok'); print('pygame', pygame.version.ver)"
```

Result:

```text
MNN ok
pygame 2.6.1
```

Command:

```powershell
.\.venv\Scripts\python.exe -c "from visimove.config import load_config; from visimove.pipeline.realtime_pipeline import build_gaze_backend_config, build_gaze_model; cfg=load_config('config/default.yaml'); cfg['gaze']['gaze_backend']='gazefollower'; cfg['gaze']['backend']='gazefollower'; model=build_gaze_model(build_gaze_backend_config(cfg)); print(type(model).__name__)"
```

Result:

```text
GazeFollower backend selected and preflight checks passed.
GazeFollowerAdapter
```

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

```text
compile passed
89 passed in 4.37s
run_tracking help passed
```

## 2026-05-16 - GazeFollower Pandas Dependency Fix

Command:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Result:

```text
passed; pandas 3.0.3 and tzdata 2026.2 installed
```

Command:

```powershell
.\.venv\Scripts\python.exe -c "import pandas; print('pandas', pandas.__version__)"
```

Result:

```text
pandas 3.0.3
```

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

```text
compile passed
89 passed in 5.02s
run_tracking help passed
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend gazefollower --show-debug --no-preview
```

Result:

```text
short smoke run no longer reports missing pandas; ended by timeout because tracking runs continuously
```

## 2026-05-16 - GazeFollower BlazeFace Import Fix

Command:

```powershell
.\.venv\Scripts\python.exe -c "from visimove.config import load_config; from visimove.pipeline.realtime_pipeline import build_gaze_backend_config, build_gaze_model; cfg=load_config('config/default.yaml'); cfg['gaze']['gaze_backend']='gazefollower'; cfg['gaze']['backend']='gazefollower'; model=build_gaze_model(build_gaze_backend_config(cfg)); model._initialize(); print(type(model).__name__, getattr(model, '_initialized', False))"
```

Result:

```text
GazeFollowerAdapter True
```

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

```text
compile passed
89 passed in 5.67s
run_tracking help passed
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend gazefollower --show-debug --no-preview
```

Result:

```text
short smoke run no longer reports the GazeFollower MediaPipe import error; ended by timeout because tracking runs continuously
```

## 2026-05-16 - GazeFollower Native Coordinate Normalization

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

Result:

```text
compile passed
```

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

```text
92 passed in 5.94s
```

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

```text
run_tracking help passed
```

Command:

```powershell
.\.venv\Scripts\python.exe -c "from visimove.config import load_config; from visimove.pipeline.realtime_pipeline import build_gaze_backend_config; cfg=load_config('config/default.yaml'); cfg['gaze']['gaze_backend']='gazefollower'; cfg['gaze']['backend']='gazefollower'; print(build_gaze_backend_config(cfg)['native_output_mode'], build_gaze_backend_config(cfg)['native_coordinate_scale_x'], build_gaze_backend_config(cfg)['native_coordinate_scale_y'])"
```

Result:

```text
model_coordinates 10.0 10.0
```

## 2026-05-16 - MobileGaze ONNX Preview Adapter

Command:

```powershell
.\.venv\Scripts\python.exe -c "from visimove.config import load_config; from visimove.pipeline.realtime_pipeline import build_gaze_backend_config, build_gaze_model; cfg=load_config('config/default.yaml'); cfg['gaze']['gaze_backend']='mobilegaze'; cfg['gaze']['backend']='mobilegaze'; model=build_gaze_model(build_gaze_backend_config(cfg)); model._initialize(); print(type(model).__name__, getattr(model, '_initialized', False), model._input_size, model._output_names)"
```

Result:

```text
MobileGazeAdapter True (448, 448) ['yaw', 'pitch']
```

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

```text
compile passed
96 passed in 4.45s
run_tracking help passed
```

## 2026-05-16 - MobileGaze Horizontal Axis Sign Fix

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

```text
compile passed
96 passed in 4.79s
run_tracking help passed
```

Command:

```powershell
.\.venv\Scripts\python.exe -c "from visimove.gaze.mobilegaze_adapter import MobileGazeAdapter; import numpy as np; print('left', MobileGazeAdapter._angles_to_raw(np.radians(-45),0)); print('center', MobileGazeAdapter._angles_to_raw(0,0)); print('right', MobileGazeAdapter._angles_to_raw(np.radians(45),0))"
```

Result:

```text
left (0.0, 0.5)
center (0.5, 0.5)
right (1.0, 0.5)
```

## 2026-05-16 - MobileGaze Calibration Provider

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_calibration.py --help
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

```text
compile passed
96 passed in 4.32s
run_calibration help passed
run_tracking help passed
```

## 2026-05-16 - MobileGaze Live Diagnostic Profile Selection

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\diagnose_live_gaze.py --help
```

Result:

```text
compile passed
98 passed in 5.90s
diagnose_live_gaze help passed
```

Coverage added:

- `diagnose_live_gaze.py --gaze-backend mobilegaze` auto-loads `data/calibration/user_profile_mobilegaze.json` when present.
- Explicit `--calibration-profile` still overrides backend-specific profile defaults.

## 2026-05-16 - Runtime Mapping Input Domain Clamp

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\diagnose_live_gaze.py --help
```

Result:

```text
compile passed
100 passed in 8.17s
run_tracking help passed
diagnose_live_gaze help passed
```

Coverage added:

- Mapping models serialize fitted raw input min/max for newly trained profiles.
- Existing affine profiles can derive the fitted mapping input domain from coefficients and target coordinates.
- Runtime mapping clamps to the fitted mapping input domain to reduce off-screen affine extrapolation.

## 2026-05-16 - Safe Edge Reach Mapping

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

```text
compile passed
run_tracking help passed
104 passed in 4.89s
```

Coverage added:

- Edge-reach expansion maps the inset calibrated target rectangle to full-screen edges.
- Edge-reach expansion respects a configured edge margin.
- CLI overrides cover `--edge-reach` and `--edge-margin-px`.

## 2026-05-17 - Adaptive Stabilization

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

```text
compile passed
run_tracking help passed
107 passed in 15.52s
```

Coverage added:

- Adaptive smoothing moves faster for large intentional cursor moves.
- Adaptive smoothing holds small fixation jitter after the configured hold time.
- CLI overrides cover adaptive smoothing parameters.

## 2026-05-17 - Edge Boost Tuning

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

```text
compile passed
run_tracking help passed
108 passed in 4.29s
```

Coverage added:

- Edge boost pushes near-edge output toward screen boundaries.
- CLI overrides cover `--edge-boost` and `--edge-boost-gamma`.

## 2026-05-18 - MobileGaze Mapping Quality Guard

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m json.tool docs\PROJECT_STATE.json
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\diagnose_calibration_mapping.py --profile data\calibration\user_profile_mobilegaze.json
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Coverage added:

- MobileGaze calibration can use `auto` mapping selection.
- Auto mapping selection chooses the candidate with the lowest point-mean screen error.
- Mapping diagnostics preserve calibration collection order.
- High mapping error downgrades calibration quality from `good` to `needs_review`.
- Startup rechecks saved profile mapping diagnostics so old inaccurate profiles are blocked from cursor control by default.

Result:

```text
compile passed
PROJECT_STATE.json valid
run_tracking help passed
diagnose_calibration_mapping warning check passed
118 passed in 4.30s
```

## 2026-05-18 - MobileGaze Raw-Sample Mapping Stability Guard

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\run_calibration.py --help
.\.venv\Scripts\python.exe scripts\diagnose_calibration_mapping.py --profile data\calibration\user_profile_mobilegaze.json
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Coverage added:

- Mapping diagnostics now report raw-sample mean absolute error and raw-sample clipping ratio.
- Calibration quality flags profiles whose raw samples map outside the screen even when point-mean diagnostics look good.
- Auto mapping selection penalizes candidates that fit point means but map raw calibration samples off-screen.

Result:

```text
compile passed
run_tracking help passed
run_calibration help passed
diagnose_calibration_mapping now flags current polynomial profile as needs-review-worthy
120 passed in 4.53s
```

## 2026-05-18 - MobileGaze Bounded IDW Mapping

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\run_calibration.py --help
.\.venv\Scripts\python.exe scripts\diagnose_calibration_mapping.py --profile data\calibration\user_profile_mobilegaze.json
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Coverage added:

- Added bounded `idw` calibration mapping for noisy MobileGaze point-mean profiles.
- MobileGaze `auto` mapping now evaluates `idw` before polynomial, linear, and affine.
- IDW serialization/deserialization preserves control raw points and target screen points.
- Current MobileGaze samples select `idw` in dry-run scoring and avoid raw-sample off-screen predictions.

Result:

```text
compile passed
PROJECT_STATE.json valid
run_tracking help passed
run_calibration help passed
diagnose_calibration_mapping confirms the saved profile is still affine/needs review until recalibration
122 passed in 13.89s
```

## 2026-05-18 - MobileGaze Grid Mapping Candidate

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m json.tool docs\PROJECT_STATE.json
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\run_calibration.py --help
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Coverage added:

- Added bounded `grid` calibration mapping.
- MobileGaze `auto` mapping now evaluates `grid` before IDW, polynomial, linear, and affine.
- Grid serialization/deserialization preserves row and column calibration knots.
- Current MobileGaze samples select `grid` in dry-run scoring and keep raw-sample clipping at `0%`.

Result:

```text
compile passed
PROJECT_STATE.json valid
run_tracking help passed
run_calibration help passed
124 passed in 5.79s
```

### 2026-05-25 - Revert Median-Window Cursor Tuning

Commands:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp src\visimove\tests\test_smoothing.py src\visimove\tests\test_run_tracking_startup.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

```text
compile passed
29 focused tests passed in 3.20s
126 passed in 4.75s
```

### 2026-05-25 - Relax Unstable Live-Quality Cursor Gate

Commands:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp src\visimove\tests\test_realtime_live_safety.py src\visimove\tests\test_live_quality.py src\visimove\tests\test_smoothing.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

```text
compile passed
18 focused tests passed in 0.49s
127 passed in 4.98s
run_tracking help passed
```

### 2026-05-25 - Adaptive Large-Jump Confirmation

Commands:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp src\visimove\tests\test_smoothing.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp src\visimove\tests\test_realtime_live_safety.py src\visimove\tests\test_live_quality.py src\visimove\tests\test_run_tracking_startup.py src\visimove\tests\test_smoothing.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Coverage added:

- Adaptive smoothing holds isolated large target spikes as `confirming`.
- Adaptive smoothing accepts a large jump after the target repeats within the confirmation radius.
- Existing immediate large-motion behavior is still available when confirmation samples are set to `1`.

Result:

```text
compile passed
14 smoothing tests passed in 1.02s
37 focused tests passed in 3.43s
129 passed in 6.11s
run_tracking help passed
```

### 2026-05-25 - Adaptive Edge Catch-Up

Commands:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp src\visimove\tests\test_smoothing.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp src\visimove\tests\test_realtime_live_safety.py src\visimove\tests\test_live_quality.py src\visimove\tests\test_run_tracking_startup.py src\visimove\tests\test_smoothing.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Coverage added:

- Adaptive smoothing uses faster catch-up for confirmed physical edge targets.
- Adaptive smoothing snaps the final few pixels when already close to a physical edge.
- Edge behavior is only enabled when screen bounds are supplied by the real-time pipeline.

Result:

```text
compile passed
16 smoothing tests passed in 0.26s
39 focused tests passed in 2.90s
131 passed in 4.67s
run_tracking help passed
```

## Velocity Cursor And Dwell Tests

Files:

- src/visimove/tests/test_velocity_control.py
- src/visimove/tests/test_dwell.py
- src/visimove/tests/test_realtime_live_safety.py
- src/visimove/tests/test_run_tracking_startup.py

Coverage includes center deadzone hold, bounded incremental motion, long-frame teleport prevention, edge-margin clamping, missing-gaze hold, pause/resume position synchronization, velocity pipeline selection, one-shot dwell behavior, dwell reset, dwell blocking while steering, and release-default configuration.
