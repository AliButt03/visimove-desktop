# VisiMove Desktop

VisiMove is a real-time webcam-based eye-controlled cursor system for Windows accessibility use. This project starts with a clean, modular architecture using OpenCV/MediaPipe-style detection interfaces, calibration, smoothing, blink/dwell click strategies, and Windows cursor control.

This scaffold intentionally does not start with model training, large datasets, or YOLO. External gaze and detection projects can be added later as isolated backends under `external/` and wrapped by adapters in `src/visimove/`.

## Status

- Dummy backends run without external repositories or trained models.
- Config files drive thresholds, smoothing, dwell timing, and performance settings.
- Real cursor movement is guarded behind configuration and backend selection.
- Dummy gaze is for pipeline testing only; it does not represent real eye movement.
- Training scripts and notebooks are placeholders for later cloud workflows.

## Project Memory

Persistent project context lives in:

- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/current_status.md`

Future coding tasks must read `docs/PROJECT_CONTEXT.md` and `docs/PROJECT_STATE.json` before making changes. At the end of each task, update both files with what changed, files modified, current status, known issues, and the next recommended action.

## Quick Start

```powershell
cd visimove-desktop
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts\run_tracking.py
python scripts\run_calibration.py
```

## Safe Cursor Testing

By default, `scripts/run_tracking.py` opens the webcam preview and keeps real cursor movement disabled. The startup summary prints the selected detector, gaze, blink, calibration, camera, cursor, and smoothing settings.

Dummy gaze moves a fake point for testing the pipeline. It must not control the Windows cursor during normal use because the movement is not based on real eye gaze.

Safe commands:

```powershell
python scripts\run_tracking.py
python scripts\run_tracking.py --enable-cursor
python scripts\run_tracking.py --enable-cursor --allow-dummy-cursor
```

`--enable-cursor` is blocked automatically when the active gaze backend is `dummy`. Use `--allow-dummy-cursor` only for short pipeline tests. For real cursor use, select a real gaze backend such as `--gaze-backend eyetrax`, `--gaze-backend gazefollower`, or `--gaze-backend mobilegaze` after that backend is configured.

Recalibrate every time the gaze backend changes. A calibration created with dummy gaze is not valid for real gaze backends.

## EyeTrax Preview

EyeTrax is the first real gaze backend target. Keep cursor movement disabled while verifying it:

```powershell
python scripts\prepare_eyetrax_model.py --check-only
python scripts\prepare_eyetrax_model.py --download-face-landmarker
python scripts\prepare_eyetrax_model.py
python scripts\run_tracking.py --gaze-backend eyetrax --show-debug
python scripts\run_calibration.py --gaze-backend eyetrax
python scripts\diagnose_calibration_mapping.py --profile data/calibration/user_profile_eyetrax.json
python scripts\run_tracking.py --gaze-backend eyetrax --show-debug
```

EyeTrax expects the repo at `external/eyetrax/`, a calibrated model such as `models/gaze/eyetrax/gaze_model.pkl`, and a local `models/detection/face_landmarker.task`. If setup is incomplete, VisiMove prints a clear warning and falls back to dummy gaze for preview/testing only. Do not enable real cursor control until EyeTrax output is stable and calibration matches the selected backend.

EyeTrax uses two calibration layers. First, EyeTrax creates the per-user `gaze_model.pkl`. Then VisiMove screen calibration collects real EyeTrax samples and saves `data/calibration/user_profile_eyetrax.json` with quality diagnostics. If tracking reports `calibration_quality=poor` or `needs_review`, cursor movement is blocked by default. Re-run calibration or inspect the pinned/low-range warnings before testing cursor control.

If preview shows raw gaze changing but `mapped_after_clamp` is stuck at the top edge, run the calibration diagnostic command above. It reports mapped coordinates before clamping, prediction error, clipping, and whether mapped Y is stuck at `0`.

Live tracking also checks whether current EyeTrax raw gaze is inside the raw domain captured during calibration. A saved profile can be `good` while live tracking is `unsafe` if live `raw_y` drops below the calibrated minimum. VisiMove clamps the mapping input for safer preview output, but blocks cursor movement when live gaze is repeatedly outside the calibrated domain. `--allow-unstable-live-gaze` exists only for careful testing.

If the cursor is smooth but biased toward the center/right side, diagnose live gaze coverage before enabling cursor again:

```powershell
python scripts\diagnose_live_gaze.py --gaze-backend eyetrax --guided
python scripts\run_tracking.py --gaze-backend eyetrax --show-debug --horizontal-gain 1.3
python scripts\run_tracking.py --gaze-backend eyetrax --enable-cursor --horizontal-gain 1.3
```

`--horizontal-gain` expands mapped X around the screen center after calibration and before smoothing. Start with preview values around `1.2` to `1.4`; gains above `2.0` are noisy and VisiMove warns before running.

Current EyeTrax tracing showed usable horizontal separation at times, but the Y axis can saturate badly on this setup. Keep EyeTrax in the repo for reproducible diagnostics, but prefer GazeFollower as the next real gaze backend candidate before adding more correction logic.

## GazeFollower Preview

GazeFollower is isolated under `external/gazefollower/` and uses bundled MNN weights:

```text
external/gazefollower/gazefollower/res/model_weights/base.mnn
external/gazefollower/gazefollower/res/model_weights/blaze_face.mnn
```

It requires optional dependencies such as `MNN` and `pygame` in the selected virtual environment. If those are missing, VisiMove falls back to dummy gaze and keeps cursor movement safe.

```powershell
python scripts\run_tracking.py --gaze-backend gazefollower --show-debug
```

Do not enable cursor with GazeFollower until preview shows real output, a GazeFollower-specific VisiMove calibration profile has been created, and live tracking is stable.

## Design Principles

- Keep first-party app code in `src/visimove/`.
- Keep external repositories isolated in `external/`.
- Wrap external code through adapter classes.
- Prefer MediaPipe/OpenCV-style lightweight detection first.
- Use YOLO later only if face/eye detection is unstable.
- Keep thresholds and model paths config-driven.
