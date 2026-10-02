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
- `CLAUDE.md`
- `docs/CLAUDE_HANDOFF.md`

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

Current EyeTrax tracing showed usable horizontal separation at times, but the Y axis can saturate badly on this setup. GazeFollower runs, but guided diagnostics showed weak left/center/right separation. MobileGaze is now the next preview backend candidate.

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

GazeFollower's `raw_gaze_coordinates` are not Windows screen pixels. The upstream code sets them from the first two values of the MNN model output, then uses a separate SVR calibration over the full feature vector before converting calibrated values to screen pixels. VisiMove therefore labels them as `gazefollower_model_coordinates` and maps them into 0..1 preview space with a config-driven centered transform.

Do not enable cursor with GazeFollower until preview shows real output, a GazeFollower-specific VisiMove calibration profile has been created, and live tracking is stable.

## MobileGaze Preview

MobileGaze is isolated under `external/mobilegaze/`. The lightweight ONNX model should be placed here:

```text
external/mobilegaze/weights/mobileone_s0_gaze.onnx
```

MobileGaze predicts `yaw` and `pitch` gaze angles in radians, not screen pixels. VisiMove converts those angles into normalized raw gaze for preview and calibration. MobileGaze is currently the strongest backend on this setup: the saved MobileGaze profile loads with `calibration_quality=good`, and preview maps gaze to varied screen coordinates.

MobileGaze calibration uses an auto-selected point-mean mapping model. The auto selector includes bounded grid and IDW interpolation models to avoid affine/polynomial off-screen extrapolation from noisy yaw/pitch samples. If startup or `diagnose_calibration_mapping.py` reports high mapping error, rerun calibration before cursor testing; raw range alone is not enough for accurate pointer placement.

```powershell
python scripts\run_tracking.py --gaze-backend mobilegaze --show-debug
python scripts\diagnose_live_gaze.py --gaze-backend mobilegaze --guided
python scripts\run_calibration.py --gaze-backend mobilegaze --verbose-quality
python scripts\diagnose_calibration_mapping.py --profile data\calibration\user_profile_mobilegaze.json
python scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor
```

Run the cursor command only after one more cursor-disabled preview shows `fallback=no`, `calibration_quality=good`, useful mapped movement, and mostly `live_tracking_quality=stable`. If debug output shows `cursor_skip=none` but the pointer does not visibly move, compare cursor backends:

```powershell
python scripts\test_cursor_backend.py --backend pyautogui
python scripts\test_cursor_backend.py --backend win32
python scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --allow-unstable-live-gaze --show-debug --cursor-backend win32
```

If the cursor follows gaze but shakes or drifts during fixation, first test the calibrated baseline without edge expansion:

```powershell
python scripts\run_tracking.py --gaze-backend mobilegaze --show-debug --no-edge-reach --no-edge-boost --vertical-offset 0
```

Safe edge reach expands the inset calibration target rectangle to the full screen after safe raw-domain mapping and is enabled by default for calibrated cursor testing. Edge boost can push near-edge gaze harder toward taskbar/corner controls, but it is disabled by default because it can amplify noisy gaze.

Absolute cursor mode is the release default because it maps gaze to the corresponding screen location. Edge reach and speed-adaptive One Euro stabilization remain active. Velocity mode is retained only as an optional directional-steering experiment; it is not appropriate when the cursor must land where the user is looking.

Blink clicking is not ready yet; natural blinks are not detected until a real OCEC/ONNX blink backend is integrated.

## Design Principles

- Keep first-party app code in `src/visimove/`.
- Keep external repositories isolated in `external/`.
- Wrap external code through adapter classes.
- Prefer MediaPipe/OpenCV-style lightweight detection first.
- Use YOLO later only if face/eye detection is unstable.
- Keep thresholds and model paths config-driven.

## Direct Gaze Cursor Mode

The release cursor mode is absolute. MobileGaze output is calibrated to screen coordinates, expanded through safe edge reach, and stabilized by a bounded One Euro filter before cursor movement. The filter applies stronger smoothing to slow fixation jitter, responds to intentional gaze movement, and limits output speed and acceleration so noisy edge targets cannot reverse the cursor abruptly.

    .\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --show-debug --cursor-mode absolute

Press `p` to resume or pause. Absolute mode does not require neutral recentering. Dwell selection remains available after the cursor settles inside its configured radius.
