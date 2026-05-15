# VisiMove Project Context

This is the permanent project memory file for VisiMove. Future coding tasks must read this file and `docs/PROJECT_STATE.json` before making changes, then update both files at the end of the task.

## Project Goal

VisiMove is a final-year accessibility project for Windows desktop interaction. It uses a webcam to estimate user gaze for mouse cursor movement and uses blink detection or dwell selection for clicking. The goal is a safe, real-time, webcam-based eye-controlled cursor system for users who benefit from hands-free desktop control.

## Current Implementation Status

- Python project structure exists under `visimove-desktop/`.
- First-party application code lives under `src/visimove/`.
- `scripts/run_tracking.py` runs the real-time webcam pipeline.
- `scripts/run_tracking.py --enable-cursor` runs and can move the cursor.
- `scripts/run_tracking.py` prints a startup summary for backend, calibration, camera, cursor, and smoothing settings.
- Dummy gaze is blocked from moving the real cursor unless `--allow-dummy-cursor` is explicitly passed with `--enable-cursor`.
- Tracking loads calibration profiles by CLI path, config path, or saved default profile path.
- Tracking warns when calibration is missing, dummy-created, backend-mismatched, or screen-size-mismatched.
- `--show-debug` prints throttled raw/mapped/smoothed gaze coordinates, blink probability, click event, cursor skip reason, and detector status.
- `scripts/run_calibration.py` runs calibration and saves a JSON profile.
- Webcam capture exists through OpenCV.
- Real-time pipeline exists with camera, detector, gaze, blink, calibration, smoothing, cursor, and performance stages.
- Dummy gaze backend is active by default.
- Dummy blink backend is active by default.
- Detector backend uses safe fallback behavior; current MediaPipe install has compatibility issues, so OpenCV Haar fallback works.
- Calibration JSON save/load support exists.
- A calibration profile has been saved to `data/calibration/user_profile.json`.
- Smoothing filters exist: EMA, deadzone, fixation, and Kalman.
- Cursor safety and Windows cursor controllers exist.
- Blink state machine exists.
- Blink model adapter shells exist for dummy, ONNX, and OCEC.
- Gaze adapter shells exist for EyeTrax, GazeFollower, and MobileGaze.
- EyeTrax adapter is implemented for preview/debug backend selection with strict setup validation and dummy fallback.
- EyeTrax repo does not include pretrained gaze weights; it requires per-user calibration/training to create `models/gaze/eyetrax/gaze_model.pkl`.
- `scripts/prepare_eyetrax_model.py` exists as a safe wrapper around EyeTrax model preparation.
- `models/detection/face_landmarker.task` has been downloaded.
- `models/gaze/eyetrax/gaze_model.pkl` has been generated through the EyeTrax interactive calibration flow.
- EyeTrax preview smoke test showed `gaze_backend=eyetrax` and `fallback=no` with cursor disabled.
- VisiMove calibration can now collect real EyeTrax gaze samples and save `data/calibration/user_profile_eyetrax.json`.
- Calibration profiles include quality diagnostics: sample counts, raw gaze ranges, confidence statistics, warnings, and `calibration_quality`.
- `scripts/diagnose_calibration_mapping.py` diagnoses calibration mapping before and after screen clamping.
- EyeTrax VisiMove calibration now defaults to affine mapping instead of ridge to reduce top-edge overfit/clamping.
- Calibration profiles store the calibrated raw gaze domain.
- Runtime mapping can clamp raw input to the actual calibrated raw min/max while preserving original raw gaze in debug output. The margin is only used as violation-detection tolerance.
- Tracking reports live tracking quality as `stable`, `unstable`, or `unsafe` and blocks cursor movement when live gaze is outside the calibrated domain unless `--allow-unstable-live-gaze` is passed.
- Live EyeTrax tracking has reached stable quality and can drive the cursor, but current cursor coverage is biased toward center/right in the user's latest test.
- Axis gain/offset tuning is implemented after calibration mapping and before smoothing.
- `scripts/diagnose_live_gaze.py` can collect guided left/center/right/top/bottom live gaze stats and diagnose horizontal bias.
- `scripts/trace_eyetrax_pipeline.py` traces EyeTrax FaceLandmarker, features, raw model prediction, adapter output, and VisiMove mapping without applying gain/offset correction.
- Static inspection found no VisiMove 9-point target-order bug and no sample-target pairing bug.
- EyeTrax `gaze_model.pkl` predicts absolute screen pixel coordinates; the adapter normalizes those pixels to `raw_x/raw_y` before VisiMove calibration mapping.
- Tracking blocks cursor movement when calibration quality is `poor` or `needs_review` unless `--allow-low-quality-calibration` is explicitly passed.
- Optional YOLO detector support exists for face/left_eye/right_eye only.
- External repositories are cloned or prepared under `external/`.
- External repository setup docs and `scripts/setup_external_repos.py` are prepared.
- Tests have passed.
- Current limitation: EyeTrax is not suitable enough for the current setup because guided tracing showed unreliable/saturated Y-axis behavior.
- GazeFollower has been inspected as the next real gaze backend candidate.
- GazeFollower includes bundled MNN weights at `external/gazefollower/gazefollower/res/model_weights/base.mnn` and `external/gazefollower/gazefollower/res/model_weights/blaze_face.mnn`.
- GazeFollower adapter setup validation is implemented with safe dummy fallback when dependencies or weights are missing.
- GazeFollower optional dependencies `MNN`, `pygame`, and `pandas` are installed in the selected `.venv`.
- GazeFollower adapter setup now passes preflight and can be selected for preview/debug mode.
- GazeFollower adapter avoids the upstream MediaPipe face-alignment import path when `face_alignment_backend=blazeface`.
- Tracking debug output now includes native backend gaze output when provided by the backend.

## Important Architecture Rules

- `src/visimove/` is our own product/application code.
- `external/` contains external GitHub repositories only.
- External code must be accessed only through adapter classes.
- Do not copy external repo code directly into `src/visimove/`.
- Do not commit large models, datasets, checkpoints, logs, or generated artifacts to GitHub.
- Prefer config-driven thresholds, paths, backends, and safety settings.
- Cursor must remain disabled by default.
- Dummy gaze must not be allowed to control the real cursor unless explicitly enabled for testing.
- YOLO is optional only for face/left_eye/right_eye detection.
- YOLO must not be used as the main gaze-estimation model.
- Preserve existing working functionality while improving the system.

## Current Problem

EyeTrax real gaze works but is not reliable enough for the current setup because Y-axis output can become saturated/pinned. The next real gaze backend candidate is GazeFollower. GazeFollower is present under `external/gazefollower/`, required dependencies are installed, and the next step is preview/debug testing with cursor disabled.

## Next Required Steps

1. Run `python scripts/run_tracking.py --gaze-backend gazefollower --show-debug`.
2. Confirm startup shows `gazefollower initialized: yes` and debug output shows `gaze_backend=gazefollower fallback=no`.
3. Inspect live raw output in preview mode only; do not enable cursor yet.
4. Add/adjust VisiMove calibration support for GazeFollower if preview output is usable.
5. Recalibrate after selecting GazeFollower.
6. Test cursor only after preview shows usable X/Y coverage, `calibration_quality=good` or `acceptable`, and `live_tracking_quality=stable`.
7. If GazeFollower is not reliable or dependencies conflict, inspect MobileGaze next.
8. Integrate OCEC/ONNX blink backend.
9. Add dwell click.
10. Optimize performance.
11. Prepare final demo mode.

## Commands Already Tested

```powershell
python scripts/run_tracking.py
python scripts/run_tracking.py --enable-cursor
python scripts/run_calibration.py
python -m pytest
```

Notes:

- `run_tracking.py` works.
- `run_tracking.py --enable-cursor` works.
- Tests pass.
- Calibration saves JSON.
- `--enable-cursor` moves cursor randomly because dummy gaze is active.
- Current detector default should avoid noisy MediaPipe warnings and use OpenCV fallback when MediaPipe Tasks model is unavailable.

## Safe Testing Rules

- Do not enable cursor with dummy gaze by default.
- Do not enable click until blink backend is verified.
- Always test preview/debug mode before enabling cursor.
- Recalibrate after switching gaze backend.
- Keep Ctrl+C ready during cursor tests.
- Keep cursor safety enabled.
- Keep cursor disabled by default in config.
- Use `p` pause/resume behavior during live cursor tests.

## External Backend Plan

- EyeTrax/EyePy: first real gaze backend target.
- GazeFollower: alternate high-accuracy gaze backend.
- MobileGaze: PyTorch/ONNX gaze backend for later experiments.
- OCEC: blink detection backend.
- YOLO11n: optional only for face/left_eye/right_eye detection if MediaPipe/OpenCV are unstable.

## Coding Style Rules

- Keep code clean.
- Keep modules small and focused.
- Use typed code where practical.
- Optimize for real-time performance.
- Avoid huge files.
- Avoid duplicate logic.
- Avoid hardcoded paths.
- Use clear errors and logs.
- Preserve existing working functionality.
- Inspect files before editing.
- Update tests when behavior changes.
- Update documentation when workflow or architecture changes.

## Standard Task Workflow

Every future coding task should follow this workflow:

1. Read `docs/PROJECT_CONTEXT.md`.
2. Read `docs/PROJECT_STATE.json`.
3. Inspect relevant source/config/test files before editing.
4. Make focused changes.
5. Run appropriate checks:
   - compile check: `python -m compileall -q src scripts`
   - tests: `python -m pytest -q -p no:cacheprovider`
   - relevant CLI help or smoke check
6. Clean generated `__pycache__` folders outside `.venv` and `external/`.
7. Update `docs/test_suite.md` when new tests/checks are added.
8. Update this file and `docs/PROJECT_STATE.json`.
9. Report changed files, commands run, known issues, and next step.

## Running Context Update Log

Every future task must append an entry here with:

- date
- what changed
- files modified
- commands to run
- known issues
- next step

### 2026-05-14

What changed:

- Created persistent project context system.
- Added machine-readable project state.
- Added current status documentation.
- Updated README to point future work at the context files.

Files modified:

- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/current_status.md`
- `README.md`

Commands to run:

```powershell
python -m pytest -q -p no:cacheprovider
python scripts/run_tracking.py
python scripts/run_tracking.py --enable-cursor
```

Known issues:

- Dummy gaze is still active and causes random cursor movement if allowed to control the cursor.
- Calibration profile was saved, but it may be degenerate until real gaze samples are collected.
- MediaPipe install may require a `.task` model for Tasks API; OpenCV fallback works.

Next step:

- Add backend startup summary/logging and block dummy gaze from real cursor control unless `--allow-dummy-cursor` is passed.

### 2026-05-14 - Backend Startup Summary And Dummy Cursor Safety

What changed:

- Added startup summary logging to `scripts/run_tracking.py`.
- Added CLI overrides for gaze, detector, blink, calibration profile, dummy cursor allowance, and debug output.
- Blocked dummy gaze from moving the real cursor when `--enable-cursor` is used without `--allow-dummy-cursor`.
- Added a calibration warning for real gaze backend changes.
- Added a default calibration profile path to config.
- Documented safe dummy gaze and cursor testing behavior.
- Added focused tests for dummy cursor safety and CLI overrides.

Files modified:

- `scripts/run_tracking.py`
- `config/default.yaml`
- `README.md`
- `src/visimove/tests/test_run_tracking_startup.py`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
python scripts/run_tracking.py
python scripts/run_tracking.py --enable-cursor
python scripts/run_tracking.py --enable-cursor --allow-dummy-cursor
python scripts/run_tracking.py --gaze-backend eyetrax --show-debug
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

- Compile check passed.
- `47 passed in 4.64s`.
- `run_tracking.py --help` shows the new CLI arguments.

Known issues:

- Dummy gaze is still the default gaze backend and is not suitable for real cursor control.
- Existing calibration may be degenerate because it was captured before real gaze integration.
- MediaPipe Tasks still needs a valid `.task` model path or OpenCV fallback is used.

Next step:

- Improve calibration loading and debug coordinate output before integrating the first real gaze backend.

### 2026-05-14 - Calibration Loading And Debug Output

What changed:

- Added backend-aware calibration profile metadata for newly saved profiles.
- Added calibration profile loading priority for tracking: CLI path, config path, saved default, then no profile.
- Added startup calibration metadata reporting: model type, saved gaze/detector backend, saved screen size, current screen size, and timestamp.
- Added calibration mismatch warnings for missing profiles, dummy-gaze profiles, backend mismatches, and screen-size mismatches.
- Added cursor safety gating when calibration is missing or mismatched unless `--allow-dummy-cursor` is used for controlled testing.
- Added loaded-profile mapping support in `CalibrationMapper`.
- Added throttled `--show-debug` coordinate output in the real-time pipeline.
- Made performance reports show camera, detector, gaze, blink, calibration, smoothing, cursor, and FPS consistently.
- Updated calibration docs and tests.

Files modified:

- `scripts/run_tracking.py`
- `scripts/run_calibration.py`
- `src/visimove/calibration/calibration_store.py`
- `src/visimove/calibration/calibration_ui.py`
- `src/visimove/calibration/mapper.py`
- `src/visimove/pipeline/realtime_pipeline.py`
- `src/visimove/pipeline/performance_monitor.py`
- `src/visimove/tests/test_calibration_store.py`
- `src/visimove/tests/test_mapper.py`
- `src/visimove/tests/test_run_tracking_startup.py`
- `docs/calibration.md`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/current_status.md`

Commands to run:

```powershell
python scripts/run_tracking.py
python scripts/run_tracking.py --show-debug
python scripts/run_calibration.py --gaze-backend dummy
python scripts/run_tracking.py --calibration-profile data/calibration/user_profile.json --show-debug
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\run_calibration.py --help
```

Result:

- Compile check passed.
- `50 passed in 2.27s`.
- Tracking and calibration CLI help passed.

Known issues:

- Dummy gaze is still the default gaze backend and is not suitable for real cursor control.
- Existing calibration may have old or dummy backend metadata and should be regenerated before real cursor use.
- Real gaze backends are still adapter shells and have not been integrated in this task.
- MediaPipe Tasks still needs a valid `.task` model path or OpenCV fallback is used.

Next step:

- Prepare/verify external repo setup, then integrate EyeTrax as the first real gaze backend with preview/debug mode before cursor control.

### 2026-05-14 - External Repository Setup System

What changed:

- Rebuilt `scripts/setup_external_repos.py` as a safe setup helper.
- The setup script now creates/verifies external folders, prints clone commands, prints expected model locations, avoids large downloads, and does not hard-fail if cloning is unavailable.
- Updated `external/README.md` with isolation rules and adapter mapping.
- Rewrote `docs/setup_external_repos.md` with per-repo purpose, paths, clone commands, dependencies, model weight locations, config keys, licensing warnings, troubleshooting, and fallback behavior.
- Updated `.gitignore` for external model folders, datasets, checkpoints, runs, caches, image/video files, and model weights.
- Verified the expected external folders exist: `eyetrax`, `gazefollower`, `mobilegaze`, and `ocec`.

Files modified:

- `scripts/setup_external_repos.py`
- `external/README.md`
- `docs/setup_external_repos.md`
- `.gitignore`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/current_status.md`

Commands to run:

```powershell
python scripts/setup_external_repos.py
python scripts/setup_external_repos.py --help
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe scripts\setup_external_repos.py --help
.\.venv\Scripts\python.exe scripts\setup_external_repos.py
.\.venv\Scripts\python.exe -m compileall -q scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Result:

- Setup script help passed.
- Setup script printed external setup instructions and confirmed folder structure.
- Compile check passed.
- `50 passed in 2.25s`.

Known issues:

- External repos are isolated and prepared, but no real backend adapter is wired to perform real gaze or blink inference yet.
- Model weights still need manual download after license review.
- External dependency compatibility still needs inspection before installing into the main `.venv`.

Next step:

- Integrate EyeTrax as the first real gaze backend in preview/debug mode, without enabling cursor control until gaze output is verified and recalibrated.

### 2026-05-14 - EyeTrax Preview Backend Adapter

What changed:

- Inspected the actual `external/eyetrax` repository and confirmed it exposes `GazeEstimator`, `extract_features(frame)`, `predict(...)`, and `load_model(...)`.
- Implemented `src/visimove/gaze/eyetrax_adapter.py` as the first real gaze backend adapter.
- EyeTrax setup validation now checks repo files, model path, local `face_landmarker.task`, and importable dependencies.
- Tracking can select `--gaze-backend eyetrax` and reports EyeTrax setup status, effective fallback backend, model paths, and debug metadata.
- EyeTrax falls back to dummy gaze when configured to do so, but does not pretend dummy output is real gaze.
- Cursor movement remains blocked when EyeTrax fails and fallback dummy is active unless the explicit dummy override is used.
- `run_calibration.py --gaze-backend eyetrax` refuses to write a misleading calibration profile when EyeTrax is not ready or when the current VisiMove calibration UI cannot collect real EyeTrax features.
- Added EyeTrax config defaults and updated README/setup/architecture/calibration docs.
- Added tests for missing repo, missing model, fallback behavior, and backend-specific config merging.

Files modified:

- `src/visimove/gaze/eyetrax_adapter.py`
- `src/visimove/gaze/dummy_gaze_model.py`
- `src/visimove/pipeline/realtime_pipeline.py`
- `scripts/run_tracking.py`
- `scripts/run_calibration.py`
- `config/default.yaml`
- `src/visimove/tests/test_gaze_adapters.py`
- `docs/setup_external_repos.md`
- `docs/architecture.md`
- `docs/calibration.md`
- `README.md`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/current_status.md`

Commands to run:

```powershell
python scripts/run_tracking.py --gaze-backend eyetrax --show-debug
python scripts/run_calibration.py --gaze-backend eyetrax
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\run_calibration.py --help
.\.venv\Scripts\python.exe scripts\run_calibration.py --gaze-backend eyetrax
```

Result:

- Compile check passed.
- `54 passed in 1.76s`.
- CLI help passed.
- EyeTrax calibration refused safely because `models/gaze/eyetrax/gaze_model.pkl` is missing.

Known issues:

- EyeTrax repo exists, but real EyeTrax tracking cannot run until a calibrated `gaze_model.pkl` and local `models/detection/face_landmarker.task` are available.
- VisiMove calibration UI is not yet wired to collect EyeTrax features, so EyeTrax calibration is intentionally refused rather than writing a fake profile.
- EyeTrax dependencies may need version review because its `pyproject.toml` requests `numpy<2` while the current environment reports NumPy 2.x.
- Cursor control should remain disabled until EyeTrax preview output is real, stable, and recalibrated.

Next step:

- Add a guided EyeTrax model/calibration preparation path or integrate EyeTrax feature collection into VisiMove calibration, then verify real EyeTrax preview/debug output before cursor control.

### 2026-05-14 - EyeTrax Model Preparation Inspection

What changed:

- Inspected `external/eyetrax` for pretrained weights, training scripts, calibration scripts, model save/load APIs, and export instructions.
- Confirmed EyeTrax does not include pretrained gaze weights or a ready `gaze_model.pkl`.
- Confirmed EyeTrax uses per-user calibration/training through `GazeEstimator.train(...)` and `GazeEstimator.save_model(...)`.
- Confirmed `external/eyetrax/src/eyetrax/app/build_model.py` launches adaptive calibration and saves the `.pkl`.
- Added `scripts/prepare_eyetrax_model.py` to print or launch the EyeTrax model-preparation command without copying external code.
- Updated `docs/setup_external_repos.md` with exact EyeTrax findings and setup flow.

Files modified:

- `scripts/prepare_eyetrax_model.py`
- `docs/setup_external_repos.md`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
python scripts/prepare_eyetrax_model.py --help
python scripts/prepare_eyetrax_model.py
python scripts/prepare_eyetrax_model.py --run
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q scripts
.\.venv\Scripts\python.exe scripts\prepare_eyetrax_model.py --help
.\.venv\Scripts\python.exe scripts\prepare_eyetrax_model.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

- Compile check passed.
- `prepare_eyetrax_model.py --help` passed.
- `prepare_eyetrax_model.py` refused safely because `models/detection/face_landmarker.task` is missing.
- `54 passed in 2.29s`.

Known issues:

- EyeTrax is not plug-and-play; it needs a per-user calibrated `gaze_model.pkl`.
- `models/detection/face_landmarker.task` is missing and must be manually downloaded/placed first.
- EyeTrax may require dependency/version cleanup because it specifies `numpy<2` while the current environment has NumPy 2.x.
- VisiMove calibration UI still does not directly collect EyeTrax features.

Next step:

- Add/download the FaceLandmarker task model and run `scripts/prepare_eyetrax_model.py --run`, or switch to GazeFollower/MobileGaze if a more pretrained backend is needed.

### 2026-05-14 - EyeTrax Preparation Finalization

What changed:

- Enhanced `scripts/prepare_eyetrax_model.py` with `--check-only`, `--download-face-landmarker`, `--force`, and `--output`.
- Added readiness checks for FaceLandmarker, EyeTrax repo, `build_model.py`, `gaze.py`, output model, and dependencies.
- Downloaded the official MediaPipe FaceLandmarker task file to `models/detection/face_landmarker.task`.
- Installed/confirmed EyeTrax runtime dependencies needed for model preparation: `scikit-learn`, `scipy`, and `screeninfo`.
- Launched EyeTrax interactive calibration and generated `models/gaze/eyetrax/gaze_model.pkl`.
- Patched the wrapper to set `PYTHONIOENCODING=utf-8` after the external EyeTrax script hit a Windows cp1252 Unicode print error after saving the model.
- Verified EyeTrax tracking preview in no-cursor/no-preview smoke mode: `gaze_backend=eyetrax`, `fallback=no`, `cursor_enabled=no`.
- Confirmed `run_calibration.py --gaze-backend eyetrax` still refuses safely because VisiMove calibration UI is not wired to EyeTrax feature collection yet.
- Updated docs, dependency metadata, tests, and ignore rules for `.task` model files.

Files modified:

- `scripts/prepare_eyetrax_model.py`
- `src/visimove/tests/test_prepare_eyetrax_model.py`
- `docs/setup_external_repos.md`
- `docs/calibration.md`
- `README.md`
- `requirements.txt`
- `pyproject.toml`
- `.gitignore`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/current_status.md`

Generated local model files:

- `models/detection/face_landmarker.task`
- `models/gaze/eyetrax/gaze_model.pkl`

Commands to run:

```powershell
python scripts/prepare_eyetrax_model.py --check-only
python scripts/run_tracking.py --gaze-backend eyetrax --show-debug
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_eyetrax_model.py --download-face-landmarker --check-only
.\.venv\Scripts\python.exe scripts\prepare_eyetrax_model.py --run
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend eyetrax --show-debug --no-preview
.\.venv\Scripts\python.exe scripts\run_calibration.py --gaze-backend eyetrax
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

- FaceLandmarker downloaded.
- EyeTrax `gaze_model.pkl` exists.
- EyeTrax preview smoke test showed `fallback=no`.
- VisiMove EyeTrax calibration refused safely because it is not wired yet.
- `58 passed in 4.59s`.

Known issues:

- Current EyeTrax debug output was real but pinned near `raw_gaze=(1.000,0.000)`, so calibration quality may be poor and should be verified interactively.
- `data/calibration/user_profile.json` is still old/degenerate and not a valid real EyeTrax calibration profile.
- VisiMove calibration UI still needs EyeTrax feature collection or an EyeTrax-aware calibration path.
- Cursor must remain disabled until EyeTrax gaze output is stable and calibration metadata is valid.

Next step:

- Wire VisiMove calibration to real EyeTrax output/features, or run a careful EyeTrax preview session and assess whether the generated model is usable before any cursor test.

### 2026-05-14 - EyeTrax VisiMove Calibration Wiring

What changed:

- Wired `scripts/run_calibration.py --gaze-backend eyetrax` to collect real EyeTrax adapter output from the webcam.
- Added an EyeTrax calibration gaze provider that reads camera frames and calls the EyeTrax adapter without using dummy gaze.
- Added calibration quality diagnostics for sample count, raw gaze range, pinned values, confidence statistics, and mapping model success.
- EyeTrax calibration now saves `data/calibration/user_profile_eyetrax.json` by default.
- Tracking startup/debug output now reports calibration quality and warnings.
- Real cursor movement is blocked when calibration quality is `poor` or `needs_review` unless `--allow-low-quality-calibration` is passed.
- Updated README, calibration docs, current status, and the living test suite.

Files modified:

- `scripts/run_calibration.py`
- `scripts/run_tracking.py`
- `src/visimove/calibration/calibration_store.py`
- `src/visimove/calibration/calibration_ui.py`
- `src/visimove/calibration/calibration_quality.py`
- `src/visimove/calibration/eyetrax_provider.py`
- `src/visimove/calibration/__init__.py`
- `src/visimove/pipeline/realtime_pipeline.py`
- `src/visimove/tests/test_calibration_store.py`
- `src/visimove/tests/test_calibration_quality.py`
- `src/visimove/tests/test_run_tracking_startup.py`
- `docs/calibration.md`
- `README.md`
- `docs/test_suite.md`
- `docs/current_status.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
python scripts/run_calibration.py --gaze-backend eyetrax
python scripts/run_tracking.py --gaze-backend eyetrax --show-debug
python scripts/run_tracking.py --gaze-backend eyetrax --enable-cursor
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\prepare_eyetrax_model.py --check-only
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\run_calibration.py --help
```

Result:

- Compile check passed.
- `63 passed in 4.08s`.
- EyeTrax preparation check shows FaceLandmarker, EyeTrax model, and dependencies are present.
- Tracking and calibration CLI help passed.

Known issues:

- EyeTrax live calibration still must be run interactively by the user because it opens a full-screen UI and uses the webcam.
- Existing `data/calibration/user_profile.json` is not a real EyeTrax calibration profile.
- Previous EyeTrax preview looked pinned near `raw_gaze=(1.000,0.000)`, so the first EyeTrax VisiMove calibration may be marked `poor` or `needs_review`.
- Cursor must remain disabled until EyeTrax calibration quality is acceptable.

Next step:

- Run `python scripts/run_calibration.py --gaze-backend eyetrax`, inspect the printed quality report and saved `data/calibration/user_profile_eyetrax.json`, then rerun EyeTrax preview/debug before any cursor test.

### 2026-05-15 - Calibration Mapping Diagnostics And Affine EyeTrax Mapping

What changed:

- Added `scripts/diagnose_calibration_mapping.py` for inspecting calibration profiles.
- Added mapped prediction diagnostics before and after clamping in tracking debug output.
- Added affine mapping support and made affine the default VisiMove calibration mapping model.
- Calibration training now fits mapping models on per-point sample means so uneven sample counts do not overweight one point.
- Calibration quality can now consider mapped-screen diagnostics, including stuck Y, clipping, negative pre-clamp Y, and high mapping error.
- Added tests for affine Y variation, mapping diagnostics, broken mapped Y quality, and cursor gating for good/acceptable profiles.
- Updated README, calibration docs, current status, and the test suite log.

Files modified:

- `config/default.yaml`
- `config/calibration.yaml`
- `scripts/diagnose_calibration_mapping.py`
- `scripts/run_calibration.py`
- `scripts/run_tracking.py`
- `src/visimove/calibration/__init__.py`
- `src/visimove/calibration/calibration_quality.py`
- `src/visimove/calibration/calibration_ui.py`
- `src/visimove/calibration/mapper.py`
- `src/visimove/calibration/mapping_diagnostics.py`
- `src/visimove/calibration/mapping_model.py`
- `src/visimove/pipeline/realtime_pipeline.py`
- `src/visimove/tests/test_calibration_quality.py`
- `src/visimove/tests/test_mapping_diagnostics.py`
- `src/visimove/tests/test_mapping_model.py`
- `src/visimove/tests/test_run_tracking_startup.py`
- `docs/calibration.md`
- `README.md`
- `docs/test_suite.md`
- `docs/current_status.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
python scripts/run_calibration.py --gaze-backend eyetrax --verbose-quality
python scripts/diagnose_calibration_mapping.py --profile data/calibration/user_profile_eyetrax.json
python scripts/run_tracking.py --gaze-backend eyetrax --show-debug
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\diagnose_calibration_mapping.py --profile data/calibration/user_profile_eyetrax.json
.\.venv\Scripts\python.exe scripts\run_calibration.py --help
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

- Compile check passed.
- `71 passed in 3.62s`.
- Diagnostic script ran on the current EyeTrax profile.
- The current saved profile still uses `ridge`; the diagnostic showed one top calibration point predicts negative Y before clamping.

Known issues:

- Existing `data/calibration/user_profile_eyetrax.json` still uses ridge mapping and should be regenerated.
- Live tracking was not run to completion in this task because it is an interactive webcam loop.
- Cursor must remain disabled until the regenerated affine profile shows mapped Y changing properly.

Next step:

- Re-run EyeTrax calibration with affine mapping, diagnose the new profile, then preview tracking. Cursor testing is only safe after mapped Y varies correctly and calibration quality is `good` or `acceptable`.

### 2026-05-15 - Live Calibration Domain Safety

What changed:

- Added raw calibration domain fields to saved/loaded calibration profiles.
- Added runtime raw-domain checks before mapping.
- Added mapping-input clamping to the calibrated raw domain plus `raw_domain_margin`.
- Extended mapping debug output with original raw gaze, calibrated raw domain, domain violations, domain-clamped mapping input, pre-screen-clamp mapping, and final screen clamp.
- Added live tracking quality states: `stable`, `unstable`, and `unsafe`.
- Added live-domain rolling-window violation tracking and cursor blocking for unstable/unsafe live gaze.
- Added `--allow-unstable-live-gaze` for careful testing only.
- Updated calibration diagnostics, docs, README, current status, and tests.

Files modified:

- `config/default.yaml`
- `config/calibration.yaml`
- `scripts/diagnose_calibration_mapping.py`
- `scripts/run_tracking.py`
- `src/visimove/calibration/__init__.py`
- `src/visimove/calibration/calibration_store.py`
- `src/visimove/calibration/live_quality.py`
- `src/visimove/calibration/mapper.py`
- `src/visimove/pipeline/realtime_pipeline.py`
- `src/visimove/tests/test_live_quality.py`
- `src/visimove/tests/test_mapper.py`
- `src/visimove/tests/test_realtime_live_safety.py`
- `src/visimove/tests/test_run_tracking_startup.py`
- `docs/calibration.md`
- `README.md`
- `docs/test_suite.md`
- `docs/current_status.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
python scripts/diagnose_calibration_mapping.py --profile data/calibration/user_profile_eyetrax.json
python scripts/run_tracking.py --gaze-backend eyetrax --show-debug
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\diagnose_calibration_mapping.py --profile data/calibration/user_profile_eyetrax.json
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend eyetrax --show-debug --no-preview
```

Result:

- Compile check passed.
- `79 passed in 4.21s`.
- Diagnostic confirms the affine EyeTrax profile has healthy mapped ranges and no diagnostic warnings.
- Tracking smoke printed the new live-domain debug fields with cursor disabled.

Known issues:

- Live smoke run was ended by timeout because tracking is an interactive webcam loop.
- In the smoke run, no face was detected, so EyeTrax returned its safe empty gaze result. A face-visible preview session is still needed to verify real live-domain behavior.
- Cursor must remain disabled until live debug shows `live_tracking_quality=stable` during real face detection.

Next step:

- Run `python scripts/run_tracking.py --gaze-backend eyetrax --show-debug`, keep cursor disabled, and verify live raw gaze stays mostly inside the calibrated raw domain before any cursor test.

### 2026-05-15 - Raw Domain Clamp Correction

What changed:

- Corrected runtime raw-domain clamping so mapping input clamps to the actual calibrated raw min/max.
- Kept `raw_domain_margin` only as tolerance for deciding whether a live raw sample counts as a domain violation.
- Updated below-min raw Y tests to assert clamping to `raw_y_min`, not `raw_y_min - margin`.
- Updated calibration/current-status/test docs.

Files modified:

- `src/visimove/calibration/mapper.py`
- `src/visimove/tests/test_mapper.py`
- `docs/calibration.md`
- `docs/current_status.md`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
python scripts/run_tracking.py --gaze-backend eyetrax --show-debug
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

- Compile check passed.
- `80 passed in 4.49s`.

Known issues:

- Cursor remains blocked when live tracking quality is unstable or unsafe.
- Face-visible live preview is still needed to verify the corrected domain clamp with real EyeTrax samples.

Next step:

- Run EyeTrax preview with `--show-debug` and confirm below-min live `raw_y` maps using `mapped_input_after_domain_clamp` at the calibrated `raw_y_min` rather than `raw_y_min - margin`.

### 2026-05-15 - Live Gaze Bias Diagnostics And Axis Gain

What changed:

- Added post-mapping axis gain/offset adjustment after calibration and before smoothing.
- Added CLI tuning options: `--horizontal-gain`, `--vertical-gain`, `--horizontal-offset`, and `--vertical-offset`.
- Added debug output for `adjusted_after_gain` and current gain/offset values.
- Added `scripts/diagnose_live_gaze.py` with guided left/center/right/top/bottom collection and horizontal bias warnings.
- Added tests for gain expansion, offsets, clamping, debug field population, and CLI override behavior.
- Updated README, calibration docs, and test suite log.

Files modified:

- `config/default.yaml`
- `scripts/run_tracking.py`
- `scripts/diagnose_live_gaze.py`
- `src/visimove/calibration/__init__.py`
- `src/visimove/calibration/axis_adjustment.py`
- `src/visimove/pipeline/realtime_pipeline.py`
- `src/visimove/tests/test_axis_adjustment.py`
- `src/visimove/tests/test_run_tracking_startup.py`
- `README.md`
- `docs/calibration.md`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
python scripts/diagnose_live_gaze.py --gaze-backend eyetrax --guided
python scripts/run_tracking.py --gaze-backend eyetrax --show-debug --horizontal-gain 1.2
python scripts/run_tracking.py --gaze-backend eyetrax --show-debug --horizontal-gain 1.3
python scripts/run_tracking.py --gaze-backend eyetrax --enable-cursor --horizontal-gain 1.3
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\diagnose_live_gaze.py --help
```

Result:

- Compile check passed.
- `85 passed in 4.28s`.
- Tracking and live diagnostic help commands passed.

Known issues:

- EyeTrax cursor movement is still reported as biased toward center/right until the guided diagnostic is run and tuning is verified in preview.
- High horizontal gain can amplify jitter; values above `2.0` trigger a warning and should be avoided for normal use.
- Blink click and dwell click are still not integrated for final use.

Next step:

- Run the guided live gaze diagnostic, then preview `--horizontal-gain 1.2`, `1.3`, and possibly `1.4` before enabling cursor again.

### 2026-05-15 - EyeTrax Root-Cause Trace

What changed:

- Inspected EyeTrax `gaze.py`, `build_model.py`, calibration code, model classes, adapter, calibration provider, mapping model, tracking, calibration, and the saved EyeTrax profile.
- Confirmed EyeTrax trains and predicts absolute screen pixel coordinates.
- Confirmed the VisiMove adapter normalizes EyeTrax screen-pixel predictions to `raw_x/raw_y`, then VisiMove calibration maps them back to screen coordinates.
- Confirmed VisiMove 9-point calibration target order is row-major and the saved profile order matches that expectation.
- Confirmed calibration sample means remain paired with their exact target coordinates.
- Added `scripts/trace_eyetrax_pipeline.py` to trace EyeTrax FaceLandmarker output, features, raw prediction, adapter output, and VisiMove mapping without applying gain/offset correction.
- Added tests for target order and sample-target pairing.
- Updated calibration docs and test suite log.

Files modified:

- `scripts/trace_eyetrax_pipeline.py`
- `src/visimove/gaze/eyetrax_adapter.py`
- `src/visimove/tests/test_calibration_points.py`
- `src/visimove/tests/test_mapping_diagnostics.py`
- `docs/calibration.md`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
python scripts/trace_eyetrax_pipeline.py --guided
python scripts/trace_eyetrax_pipeline.py --frames 30
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\trace_eyetrax_pipeline.py --help
```

Result:

- Compile check passed.
- `87 passed in 4.10s`.
- EyeTrax trace CLI help passed.

Known issues:

- Live guided trace still needs to be run with the user looking far-left/center/far-right to confirm whether the EyeTrax model itself is weak on the left side.
- The saved EyeTrax profile shows calibration-time left/center/right raw X separation, but left points are not uniformly strong across all rows and right-side raw X can saturate near `1.0`.
- Axis gain exists from the previous task, but it should not be used as a fix until the trace confirms the root cause.

Next step:

- Run the guided EyeTrax trace and decide whether to recalibrate/rebuild EyeTrax, adjust VisiMove mapping, or switch to GazeFollower/MobileGaze.

### 2026-05-15 - EyeTrax Trace Return Fix

What changed:

- Fixed `EyeTraxAdapter.trace_frame()` success path so it returns `EyeTraxTrace` instead of returning `GazeResult` directly.
- Verified compile, tests, and trace CLI help.

Files modified:

- `src/visimove/gaze/eyetrax_adapter.py`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
python scripts/trace_eyetrax_pipeline.py --guided
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\trace_eyetrax_pipeline.py --help
```

Result:

- Compile check passed.
- `87 passed in 3.94s`.
- EyeTrax trace CLI help passed.

Known issues:

- Guided live trace still needs to be rerun to capture far-left/center/far-right EyeTrax predictions.

Next step:

- Rerun the guided EyeTrax trace command.

### 2026-05-15 - GazeFollower Backend Preparation

What changed:

- Inspected `external/gazefollower` and confirmed it is an MNN-based backend with bundled weights.
- Confirmed GazeFollower is closer to plug-and-play than EyeTrax because it includes `base.mnn` and `blaze_face.mnn`, but it still requires optional runtime dependencies.
- Implemented GazeFollower adapter setup validation and lazy preview inference wiring.
- Added GazeFollower startup summary details and safe dummy fallback when dependencies or weights are missing.
- Updated config defaults and docs to use the real GazeFollower `.mnn` model paths instead of an incorrect `.pth` placeholder.
- Kept EyeTrax in the project as a documented experimental backend and diagnostic reference instead of deleting it.
- Removed generated `__pycache__` folders and `.pytest-tmp`; `.pytest_cache` could not be removed because Windows denied access.

Files modified:

- `src/visimove/gaze/gazefollower_adapter.py`
- `src/visimove/pipeline/realtime_pipeline.py`
- `scripts/run_tracking.py`
- `config/default.yaml`
- `src/visimove/tests/test_gaze_adapters.py`
- `docs/setup_external_repos.md`
- `docs/architecture.md`
- `docs/calibration.md`
- `README.md`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
.\.venv\Scripts\python.exe -m pip install MNN pygame
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend gazefollower --show-debug
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe -c "from visimove.config import load_config; from visimove.pipeline.realtime_pipeline import build_gaze_backend_config, build_gaze_model; cfg=load_config('config/default.yaml'); cfg['gaze']['gaze_backend']='gazefollower'; cfg['gaze']['backend']='gazefollower'; model=build_gaze_model(build_gaze_backend_config(cfg)); print(type(model).__name__)"
```

Result:

- Compile check passed.
- `89 passed in 12.49s`.
- Tracking CLI help passed.
- GazeFollower smoke check reported missing `MNN` and `pygame`, then safely fell back to `MovingDummyGazeModel`.

Known issues:

- GazeFollower cannot run real inference until `MNN` and `pygame` are installed in the selected venv.
- GazeFollower uses a CC BY-NC-SA license; keep usage aligned with license/research constraints.
- EyeTrax remains implemented but is not recommended for the current setup because Y-axis output is unreliable/saturated.
- `.pytest_cache` remains because Windows denied deletion.

Next step:

- Install the missing GazeFollower dependencies, then run GazeFollower preview/debug mode with cursor disabled.

### 2026-05-15 - GazeFollower Dependency Install

What changed:

- Cleaned the GazeFollower dependency entries in `requirements.txt`.
- Installed `pygame 2.6.1` and `MNN 3.5.0` into the project `.venv`.
- Verified `MNN` and `pygame` imports.
- Verified GazeFollower adapter preflight now passes and returns `GazeFollowerAdapter`.

Files modified:

- `requirements.txt`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/test_suite.md`
- `docs/current_status.md`

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend gazefollower --show-debug
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import MNN, pygame; print('MNN ok'); print('pygame', pygame.version.ver)"
.\.venv\Scripts\python.exe -c "from visimove.config import load_config; from visimove.pipeline.realtime_pipeline import build_gaze_backend_config, build_gaze_model; cfg=load_config('config/default.yaml'); cfg['gaze']['gaze_backend']='gazefollower'; cfg['gaze']['backend']='gazefollower'; model=build_gaze_model(build_gaze_backend_config(cfg)); print(type(model).__name__)"
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

- Requirements install passed.
- `MNN` and `pygame` imports passed.
- GazeFollower preflight returned `GazeFollowerAdapter`.
- Compile check passed.
- `89 passed in 4.37s`.
- Tracking CLI help passed.

Known issues:

- GazeFollower still needs live webcam preview validation.
- Cursor must remain disabled until GazeFollower output is verified and a matching calibration profile exists.
- EyeTrax remains implemented but is not recommended for this setup due Y-axis saturation.

Next step:

- Run GazeFollower preview/debug mode and inspect whether it produces real, stable gaze output with `fallback=no`.

### 2026-05-16 - GazeFollower Pandas Dependency Fix

What changed:

- Added `pandas>=2.2` to `requirements.txt`.
- Installed `pandas 3.0.3` and `tzdata 2026.2` into the project `.venv`.
- Updated GazeFollower adapter preflight to check for `pandas`.
- Verified GazeFollower preflight still returns `GazeFollowerAdapter`.
- Ran a short no-preview GazeFollower smoke check; the previous `No module named 'pandas'` runtime error is resolved.

Files modified:

- `requirements.txt`
- `src/visimove/gaze/gazefollower_adapter.py`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/test_suite.md`
- `docs/current_status.md`

Commands to run:

```powershell
python scripts\run_tracking.py --gaze-backend gazefollower --show-debug
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import pandas; print('pandas', pandas.__version__)"
.\.venv\Scripts\python.exe -c "from visimove.config import load_config; from visimove.pipeline.realtime_pipeline import build_gaze_backend_config, build_gaze_model; cfg=load_config('config/default.yaml'); cfg['gaze']['gaze_backend']='gazefollower'; cfg['gaze']['backend']='gazefollower'; model=build_gaze_model(build_gaze_backend_config(cfg)); print(type(model).__name__)"
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend gazefollower --show-debug --no-preview
```

Result:

- Requirements install passed.
- `pandas 3.0.3` import passed.
- GazeFollower preflight returned `GazeFollowerAdapter`.
- Compile check passed.
- `89 passed in 5.02s`.
- Tracking CLI help passed.
- Short no-preview smoke run no longer reports missing `pandas`; it timed out because tracking is a continuous loop.

Known issues:

- The no-preview smoke run reported `no face from VisiMove detector`; a manual preview run with the face visible is needed.
- Current loaded calibration profile is old/degenerate and not valid for GazeFollower.
- Cursor must remain disabled until GazeFollower output and a matching calibration are verified.

Next step:

- Run GazeFollower preview/debug mode with your face visible and confirm real gaze values appear with `gaze_conf > 0`.

### 2026-05-16 - GazeFollower BlazeFace Import Fix

What changed:

- Patched the GazeFollower adapter to avoid importing GazeFollower's MediaPipe face-alignment initializer when using BlazeFace.
- Added lightweight import shims for the GazeFollower root package, `face_alignment`, and `misc` modules so BlazeFace and MGazeNet can load without triggering upstream UI/MediaPipe side effects.
- Added native backend gaze output to tracking debug logs when available.
- Verified GazeFollower adapter initialization without the `mediapipe.solutions` error.

Files modified:

- `src/visimove/gaze/gazefollower_adapter.py`
- `src/visimove/pipeline/realtime_pipeline.py`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/test_suite.md`
- `docs/current_status.md`

Commands to run:

```powershell
python scripts\run_tracking.py --gaze-backend gazefollower --show-debug
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -c "from visimove.config import load_config; from visimove.pipeline.realtime_pipeline import build_gaze_backend_config, build_gaze_model; cfg=load_config('config/default.yaml'); cfg['gaze']['gaze_backend']='gazefollower'; cfg['gaze']['backend']='gazefollower'; model=build_gaze_model(build_gaze_backend_config(cfg)); model._initialize(); print(type(model).__name__, getattr(model, '_initialized', False))"
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend gazefollower --show-debug --no-preview
```

Result:

- GazeFollower adapter initialization passed without `module 'mediapipe' has no attribute 'solutions'`.
- Compile check passed.
- `89 passed in 5.67s`.
- Tracking CLI help passed.
- Short no-preview smoke run no longer showed the GazeFollower MediaPipe import error; it reported no face visible in the no-preview run.

Known issues:

- The manual preview still needs to verify whether GazeFollower native gaze output moves usefully or stays pinned.
- Current default calibration profile is old/degenerate and not valid for GazeFollower.
- Cursor must remain disabled until GazeFollower output and matching calibration are verified.

Next step:

- Run GazeFollower preview/debug mode again and inspect `native_gaze`, `native_units`, `raw_gaze`, and `gaze_conf`.

### 2026-05-16 - GitHub Push Preparation

What changed:

- Reviewed `.gitignore` before first GitHub push.
- Added ignores for `.vscode/`, `models/**/*.pkl`, and `external/**/*.pkl`.
- Changed external repo ignore rules so cloned external repositories are not staged as embedded git repositories.
- Verified `models/gaze/eyetrax/gaze_model.pkl`, `models/detection/face_landmarker.task`, calibration JSON, `.venv`, `.vscode`, and external repo folders are ignored.

Files modified:

- `.gitignore`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
git add .
git status
git commit -m "Initial VisiMove desktop project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/visimove-desktop.git
git push -u origin main
```

Known issues:

- External repos are intentionally not pushed; setup docs/scripts explain how to clone them.
- Local model/calibration files are intentionally not pushed.

Next step:

- Push the clean first-party project to GitHub.
