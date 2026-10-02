# VisiMove Development Handoff

Updated: 2026-10-02

## 1. Purpose

VisiMove is a Windows accessibility desktop application that uses a webcam to estimate gaze, calibrate gaze to screen coordinates, stabilize the result, and optionally move and click the Windows cursor.

This repository contains the Python desktop/backend application. The completed website frontend discussed in project documentation exists separately and is not part of this Git repository.

## 2. Current Reality

### Working and integrated

- OpenCV webcam capture.
- MediaPipe FaceLandmarker support when its task model is present.
- OpenCV Haar fallback detection.
- Face and eye/landmark detection paths.
- MobileGaze ONNX inference using a detected face crop.
- MobileGaze yaw/pitch conversion to normalized raw gaze.
- Five-sample temporal median preprocessing inside the MobileGaze adapter.
- Backend-aware calibration profile generation.
- Grid, IDW, affine, linear, ridge, and polynomial mapping implementations.
- Calibration mapping diagnostics and quality warnings.
- Raw calibration-domain monitoring during live tracking.
- Edge-reach mapping from the inset calibration rectangle to the screen.
- Axis gain and offset adjustment.
- EMA, deadzone, fixation, adaptive, Kalman, and bounded One Euro filters.
- PyAutoGUI, Win32, and dry-run cursor controllers.
- Cursor pause/resume, confidence checks, speed limiting, and live-domain safety.
- Absolute cursor mode and an experimental velocity mode.
- Dwell selection logic in code.
- Startup summaries, debug output, and performance timings.

### Partial or not release-ready

- Accurate absolute gaze placement: blocked by inconsistent calibration signals.
- Live cursor use: controller works, but current calibration safety correctly disables it.
- Dwell click: implemented and unit-tested, not sufficiently validated with reliable live gaze.
- Blink click: state machine/adapters exist, but no real blink backend is validated; runtime normally uses dummy blink.
- EyeTrax: integrated but current setup showed Y-axis saturation.
- GazeFollower: integrated but current setup showed weak directional separation.
- MobileGaze: best available backend, but not yet accurate enough for pinpoint absolute control.
- Windows executable packaging: not implemented.

## 3. Verified Baseline

The full suite passed on 2026-10-02:

```text
172 passed in 17.04s
```

The working tree contains a large set of intentional implementation changes beyond commit `e0e0a88`. Do not discard or reset them.

## 4. Runtime Data Flow

```text
OpenCV camera
  -> detector
     MediaPipe FaceLandmarker when available
     OpenCV Haar fallback
  -> gaze adapter
     MobileGaze face crop
     ONNX yaw and pitch inference
     normalized raw gaze
     five-sample temporal median
  -> CalibrationMapper
     profile compatibility check
     calibrated raw-domain clamp
     selected mapping model
     screen clamp
  -> edge reach
  -> axis gain and offset
  -> smoothing
     bounded One Euro by default
  -> cursor safety
     pause state
     confidence
     target visibility
     live-domain quality
     speed limiting
  -> cursor controller
     PyAutoGUI by default
  -> optional dwell click
```

The principal construction and loop code is in:

- `scripts/run_tracking.py`
- `src/visimove/pipeline/realtime_pipeline.py`

## 5. Calibration Data Flow

```text
scripts/run_calibration.py
  -> backend setup and compatibility metadata
  -> MobileGazeCalibrationGazeProvider
  -> CalibrationUi full-screen targets
  -> raw samples grouped by target
  -> point-mean training data
  -> auto mapper selection
  -> temporary profile
  -> mapping diagnostics
  -> calibration quality report
  -> JSON profile
```

Important files:

- `scripts/run_calibration.py`: CLI, backend setup, profile metadata.
- `src/visimove/calibration/mobilegaze_provider.py`: webcam, detection, MobileGaze provider.
- `src/visimove/calibration/calibration_ui.py`: target timing, sample collection, fit, save.
- `src/visimove/calibration/calibration_quality.py`: profile-level quality classification.
- `src/visimove/calibration/mapping_model.py`: mapping implementations.
- `src/visimove/calibration/mapping_diagnostics.py`: point/sample errors and warnings.
- `src/visimove/calibration/calibration_store.py`: JSON schema and persistence.

## 6. Current Blocking Defect

The latest 9-point MobileGaze profile is not usable:

- Mapping model: grid.
- Mean absolute error: about `170.9px X`, `112.2px Y`.
- Raw-sample mean absolute error: about `287.7px X`, `209.3px Y`.
- Worst calibration point X error: `457px`.
- Worst calibration point Y error: `676px`.
- Quality: `needs_review`.

The top-left raw mean was approximately `0.402, 0.572`, while middle-left was approximately `0.375, 0.547`. Their vertical ordering is contradictory. Similar target overlap occurs elsewhere.

Because the profile is `needs_review`, startup forces:

```text
cursor_enabled=no
cursor_skip=cursor disabled
```

This is correct behavior. It is not a cursor-controller failure.

## 7. Root Cause Boundary

The dominant error exists before smoothing:

```text
unstable/overlapping target raw gaze
  -> incorrect mapped target
  -> smooth but inaccurate cursor
```

Smoothing can reduce visible shake but cannot infer which target the user intended when multiple targets generated overlapping raw gaze values.

Likely contributors to investigate:

- Head movement between calibration targets.
- Face crop position/scale changes.
- Natural MobileGaze model noise for this camera/user.
- Samples accepted solely because count/confidence pass.
- No target-local robust outlier rejection before point aggregation.
- No cross-target row/column consistency gate before profile activation.

## 8. Already Tried and Rejected

Do not repeat these without new evidence:

- Large horizontal or vertical offsets.
- High axis gain.
- Aggressive edge boost.
- Repeated smoothing parameter changes.
- Velocity mode as a replacement for accurate absolute mapping.
- Switching blindly between affine, polynomial, IDW, and grid.
- More calibration points without stronger sample acceptance.
- Enabling low-quality calibration for normal operation.

Prior held-out comparison found grid to be the best existing mapping option. The limitation was target signal overlap, not lack of mapper complexity.

## 9. Next Implementation Slice

Implement this in small tested stages:

### Stage A: target-local robust samples

- Group samples by target.
- Compute robust center using median.
- Reject large X/Y excursions with a configurable MAD or percentile rule.
- Record original, retained, and rejected counts.
- Mark a point weak when too few stable samples remain.

### Stage B: face/head stability

- Make the calibration provider expose normalized face-box center and size in gaze metadata.
- Establish a neutral face-pose baseline.
- Reject samples or restart the current target when translation/scale exceeds configurable limits.
- Avoid hard-coded command-line tuning.

### Stage C: cross-target consistency

- Check that horizontal target columns have useful raw-X separation.
- Check that vertical target rows have useful raw-Y separation.
- Detect inversions, overlaps, and collapsed rows/columns.
- Classify severe violations as poor, not merely good sample count.

### Stage D: profile activation

- Fit and diagnose a candidate profile.
- If quality is poor or `needs_review`, save it to a rejected/diagnostic path.
- Do not overwrite a previously acceptable active profile.
- Show the reason clearly in CLI and calibration UI.

### Stage E: validation

- Add synthetic regression tests for outliers, head-pose drift, row/column overlap, and profile preservation.
- Run the full suite.
- Perform a new 9-point live calibration.
- Diagnose before enabling cursor.

## 10. Safety Invariants

These must remain true:

- Cursor disabled by default.
- Dummy gaze cannot control the real cursor without explicit test override.
- Poor or `needs_review` calibration blocks cursor movement.
- Repeated live raw-domain violations block movement.
- Pause/resume key remains available.
- PyAutoGUI fail-safe remains enabled.
- Screen-edge margin avoids triggering the fail-safe corner.
- External backend failure falls back safely and never masquerades as valid real gaze.

The CLI exposes override flags for careful diagnosis. Do not make them default behavior.

## 11. Configuration Ownership

Runtime defaults:

- `config/default.yaml`

Calibration defaults:

- `config/calibration.yaml`

Keep new thresholds in these files and pass them through dataclass/config objects. Do not bury user-specific corrections in launch commands.

## 12. External Assets Not in Git

The following are intentionally ignored:

- `.venv/`
- `external/eyetrax/`
- `external/gazefollower/`
- `external/mobilegaze/`
- `external/ocec/`
- model files such as ONNX, MNN, PKL, TASK, PT, and PTH
- `data/calibration/*.json`
- datasets and logs

The new PC must obtain the MobileGaze repository and model separately. Expected runtime model:

```text
external/mobilegaze/weights/mobileone_s0_gaze.onnx
```

Calibration is user, camera, posture, monitor, and machine specific. Do not copy the current rejected profile as a release profile; recalibrate on the new PC.

## 13. New-PC Bootstrap

```powershell
git clone https://github.com/AliButt03/visimove-desktop.git
cd visimove-desktop
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python scripts\setup_external_repos.py --clone --depth 1
```

Then place the MobileGaze ONNX model at the expected path and run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --show-debug
```

Do not enable the cursor until a new calibration passes.

## 14. Normal Development Commands

Calibration:

```powershell
.\.venv\Scripts\python.exe scripts\run_calibration.py --gaze-backend mobilegaze --mode 9 --verbose-quality
```

Diagnostics:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_calibration_mapping.py --profile data\calibration\user_profile_mobilegaze.json
```

Safe preview:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --show-debug
```

Cursor test only after quality passes:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --show-debug --cursor-mode absolute
```

Verification:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src scripts
git diff --check
```

## 15. Tests by Area

- Calibration quality: `test_calibration_quality.py`
- Mapper behavior: `test_mapper.py`, `test_mapping_model.py`
- Diagnostics: `test_mapping_diagnostics.py`, `test_diagnose_live_gaze.py`
- MobileGaze/adapters: `test_gaze_adapters.py`
- Startup and profile safety: `test_run_tracking_startup.py`
- Runtime live safety: `test_realtime_live_safety.py`
- Smoothing: `test_smoothing.py`
- Cursor backends: `test_pyautogui_controller.py`, `test_cursor_safety.py`
- Dwell: `test_dwell.py`
- Velocity experiment: `test_velocity_control.py`
- Detection: `test_detection.py`, `test_yolo_detector.py`

## 16. Suggested First Claude Prompt

```text
Read CLAUDE.md and docs/CLAUDE_HANDOFF.md first. Do not scan the whole repository.
Run the full tests to establish the baseline. Then implement Stage A only:
config-driven robust per-target calibration outlier rejection before mapping,
with retained/rejected metrics and focused regression tests. Preserve all cursor
safety gates and do not change smoothing, mapper selection, offsets, or edge reach.
Update the handoff and project memory after verification.
```

