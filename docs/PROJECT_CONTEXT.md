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
- Tracking reports live tracking quality as `stable`, `unstable`, or `unsafe` and blocks cursor movement only when live gaze becomes `unsafe`, unless `--allow-unstable-live-gaze` is passed for careful testing.
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
- GazeFollower preview now runs with `fallback=no` and `face_found=yes`.
- GazeFollower external code confirms `raw_gaze_coordinates` is `res[:2]` from the MNN model output, not Windows screen pixels.
- GazeFollower adapter now exposes `native_units=gazefollower_model_coordinates` and maps the uncalibrated native output into VisiMove 0..1 preview raw gaze with a config-driven centered `tanh` transform.
- GazeFollower guided live diagnostics showed weak left/center/right separation and horizontal range too small, so GazeFollower is not suitable enough for cursor control in the current setup.
- MobileGaze has been inspected and includes ONNX inference support, model definitions, and a `weights/` folder.
- `external/mobilegaze/weights/mobileone_s0_gaze.onnx` has been downloaded from the official MobileGaze GitHub release.
- `onnxruntime` is installed in the selected `.venv`.
- MobileGaze ONNX adapter is implemented for preview/debug mode using VisiMove's detector face crop.
- MobileGaze adapter decodes native `(yaw, pitch)` radians, exposes `native_units=radians_yaw_pitch`, and maps angles into normalized raw gaze for preview/calibration.
- MobileGaze live preview and guided diagnostics show `fallback=no`, face detection, usable raw gaze movement, and much stronger separation than GazeFollower.
- MobileGaze horizontal axis was initially inverted; the adapter has been fixed so left maps to lower `raw_x`, center to about `0.5`, and right to higher `raw_x`.
- MobileGaze calibration provider is now wired to collect real MobileGaze ONNX gaze output and save `data/calibration/user_profile_mobilegaze.json`.
- Tracking auto-selects `data/calibration/user_profile_mobilegaze.json` for `--gaze-backend mobilegaze` when it exists.
- MobileGaze calibration has been run and the saved MobileGaze profile loads with `calibration_quality=good`.
- MobileGaze live preview now shows `fallback=no`, `face_found=yes`, a calibrated raw domain around `x[0.264,0.650], y[0.381,0.913]`, and mapped screen coordinates that move across the display.
- MobileGaze latest preview shows real gaze and good calibration, but live tracking can become `unstable` or `unsafe` because mapped output clips to screen edges, especially right/bottom.
- MobileGaze cursor control should remain disabled until clipping/live-quality behavior is corrected or a better calibration/tuning pass is completed.
- `scripts/diagnose_live_gaze.py` now auto-loads `data/calibration/user_profile_mobilegaze.json` for `--gaze-backend mobilegaze`, matching `scripts/run_tracking.py`.
- Corrected MobileGaze guided diagnostics show usable directional separation: left mean mapped X around `88`, center around `1146`, right around `2373`, top mean mapped Y around `149`, and bottom around `1215`.
- MobileGaze guided diagnostics no longer report strong horizontal bias after the profile-loading fix.
- Runtime calibration mapping now clamps raw input to the affine mapper's fitted input domain, not only the broad calibration sample raw domain. This prevents MobileGaze values inside the sample domain from extrapolating to off-screen coordinates such as `x=3108` before screen clamping.
- Post-clamp MobileGaze preview confirms `mapped_raw_before_screen_clamp` is now bounded to the calibrated target area, roughly `x<=2252` and `y>=173`, instead of jumping far outside the screen.
- MobileGaze live preview can still report `unstable` when raw gaze briefly exceeds the calibrated domain/margin, but this is now a safety signal rather than dangerous off-screen extrapolation.
- First MobileGaze cursor test was attempted. The cursor did not visibly move; likely reasons are safety gating from `live_tracking_quality=unstable`, cursor pause state, or no `--show-debug` output to confirm `cursor_skip`.
- Real blink detection is not integrated yet. Current blink backend is dummy/keyboard-triggered and does not detect natural eye blinks.
- MobileGaze cursor now follows gaze, but latest debug showed the mapped/adjusted target can be much closer to the intended location than the final `smoothed` cursor output.
- Adaptive smoothing has been fixed so it releases into fast movement when the current smoothed cursor is far from the mapped target, even if the fixation anchor is already near that target.
- Adaptive smoothing defaults were tuned for faster release and steadier fixation: fast alpha `0.45`, slow alpha `0.05`, fixation radius `48`, release radius `85`, and hold `80ms`.
- Latest MobileGaze live debug confirms the adaptive release fix is active: `smoothing_state=moving` appears when `smoothed` is far from `adjusted_after_gain`.
- The remaining live cursor issue is now the mapped target itself: `edge_boost=yes` plus `vertical_offset=-220` can push `edge_reach_after`/`adjusted_after_gain` to extremes, causing shaky cursor movement and target overshoot.
- Inspection of `data/calibration/user_profile_mobilegaze.json` shows the saved MobileGaze profile is internally consistent: center calibration mean is about `raw=(0.428, 0.665)`, left-side calibration means are about `raw_x=0.34-0.37`, and bottom calibration means are about `raw_y=0.73-0.79`.
- The latest live "center" values supplied by the user are around `raw_x=0.22-0.33` and `raw_y=0.70-0.90`, which the saved calibration correctly interprets as left/bottom. This indicates calibration/posture/camera drift, not an adapter sign bug.
- Edge boost is no longer enabled by default. It should only be enabled after a stable baseline calibration is confirmed.
- After recalibration, MobileGaze quality was `good`, but live preview still mapped many samples to top/edges because runtime mapping input was clamped to the fitted calibration point-mean domain. This was too tight for noisy MobileGaze live yaw/pitch.
- Runtime mapping input domain clamping is now disabled by default. Calibration raw-domain clamping remains enabled, so truly outside-domain raw gaze is still bounded without forcing normal in-domain samples to the nearest calibration point mean.
- Calibration quality per-point diagnostics now preserve collection order instead of sorting targets by `(x,y)`, avoiding misleading column-major output.
- Follow-up MobileGaze preview confirmed clamp behavior was fixed, but the affine min/max mapper itself still extrapolated hard from noisy/overlapping MobileGaze samples, producing negative screen X or off-screen Y for in-domain raw values.
- MobileGaze calibration now defaults to `mapping_model=auto` with `mapping_fit_strategy=point_means`, while the general/default calibration behavior remains `affine` with `point_means`. Auto tries candidate mapping models and selects the lowest point-mean screen error.
- The latest saved MobileGaze profile loaded with `calibration model type: linear`, but mapping diagnostics show it is not accurate enough: mean absolute error is about `480px` X and `235px` Y.
- Runtime startup now re-checks saved profile mapping diagnostics. A profile stored as `calibration_quality=good` is downgraded to `needs_review` at startup if its mapped calibration-point errors are high.
- The latest MobileGaze preview also showed safe edge reach can over-amplify near-edge targets. For example, a mapped point near `(2252,150)` can become `(2559,0)` after edge reach.
- Edge reach is now disabled by default. It should be treated as an opt-in tuning mode after the baseline MobileGaze mapping is stable.
- A later MobileGaze recalibration selected a `polynomial` mapping. Point-mean diagnostics looked acceptable, but live preview showed normal in-domain samples mapping far off-screen and clamping to edges.
- Mapping diagnostics now include raw-sample stability metrics. The current polynomial MobileGaze profile maps `46%` of raw calibration samples outside the screen, with raw-sample MAE around `563.7px` X and `407.9px` Y.
- MobileGaze auto mapping selection now penalizes candidates that fit calibration point means but map individual raw samples outside the screen. On the current profile data it would select `affine` instead of `polynomial`.
- Cursor should not be enabled with the current polynomial MobileGaze profile.
- After recalibration, the latest MobileGaze profile now uses `affine`, but diagnostics are still not cursor-safe: point-mean MAE is about `170.8px` X and `86.6px` Y, raw-sample MAE is about `548.7px` X and `249.9px` Y, and `36%` of raw calibration samples map outside the screen.
- The latest affine MobileGaze profile should still not be used for cursor control. A preview-only run is acceptable for collecting debug, but the next real fix is improving MobileGaze calibration stability rather than enabling cursor.
- Preview-only debug with the latest affine MobileGaze profile confirmed the mapping is unsafe even with edge reach and edge boost disabled: in-domain raw gaze such as `(0.249,0.641)` can map before clamp to `(-417.5,-88.5)` and then clamp to `(0,0)`.
- The issue is the calibration mapping/model, not edge reach, edge boost, vertical offset, smoothing, or cursor backend. Cursor must remain disabled.
- MobileGaze `auto` mapping now includes a bounded `idw` point-mean interpolation model before polynomial, linear, and affine.
- A dry-run on the current MobileGaze calibration samples selects `idw`; raw-sample clipped prediction ratio becomes `0%`, avoiding affine/polynomial off-screen extrapolation, though raw-sample MAE remains high because the gaze samples are noisy.
- The current saved MobileGaze profile is still affine until the user reruns calibration, so cursor should remain disabled until recalibration, diagnostics, and preview confirm stable mapping.
- The user reran MobileGaze calibration and diagnostics now show `mapping model type: idw`, `calibration_quality=good`, no warnings, point-mean error `0px`, raw-sample clipped prediction ratio `0%`, and raw-sample MAE around `316.7px` X / `158.1px` Y.
- Preview with cursor disabled and `--no-edge-reach --no-edge-boost --vertical-offset 0` shows `live_tracking_quality=stable`, no raw-domain violations, and mapped output no longer snapping to corners.
- First cursor-enabled IDW baseline shows `cursor_skip=none`, `live_tracking_quality=stable`, and real cursor movement, but the cursor is shaky and does not reach physical screen edges or exact intended targets.
- Edge non-reach is expected in that baseline because `--no-edge-reach` keeps output inside the calibration target rectangle. With the current 2560x1440 screen and 9-point calibration, the IDW target rectangle is about `x[307,2252]`, `y[173,1266]`.
- Remaining shake is mainly target noise: logs show `smoothing_state=moving`, so the adaptive filter is still chasing a moving MobileGaze target rather than holding fixation.
- Implemented a bounded `grid` mapping candidate for MobileGaze. It maps calibrated left/center/right columns and top/middle/bottom rows independently, staying bounded while reducing IDW center-pull.
- Dry-run scoring on the current MobileGaze samples selects `grid` over IDW: raw-sample MAE improves to about `262.2px` X and `120.6px` Y with `0%` raw-sample clipping and no warnings.
- Latest MobileGaze grid/edge-reach cursor testing is much better: the cursor gets close to intended targets and edge reach can produce full-screen targets.
- The remaining top-right corner issue is primarily smoothing lag rather than mapping failure. The debug output shows `mapped_after_clamp=(2252,173)` expanding to `edge_reach_after=(2559,0)` and `adjusted_after_gain=(2559,0)`, while `smoothed` remains behind.
- The remaining shake is still target noise plus adaptive smoothing staying in `moving` state.
- User's latest cursor test confirms the cursor is now working much better overall. Remaining issues are slower movement, occasional loss of eye tracking, and mild shaking.
- A revised Chapter 5 implementation document was generated at `docs/VisiMove_Chapter_5_Implementation_Revised.docx`. It keeps the original chapter material while updating the implementation status to MobileGaze, grid/IDW mapping, edge reach, adaptive smoothing, and current development limitations.
- Cursor tuning has been updated for smoother default behavior: edge reach is enabled by default, PyAutoGUI movement duration is `0`, cursor speed cap is `3000px/s`, adaptive smoothing uses a wider fixation lock, and the smoother holds the last position when face/eyes are lost or live gaze quality is invalid.
- The separate frontend project at `C:\Users\MNA\Desktop\FYP\visimove` has been inspected. It is a completed Next.js/React/TypeScript frontend with Tailwind CSS, Three.js/React Three Fiber/Drei, GSAP, Lenis, Lucide icons, a 3D VisiMove eye model, responsive sections, and a Windows-focused download section.
- A scoped 30% implementation chapter document was generated at `docs/VisiMove_Chapter_5_Implementation_30_Percent.docx`. It presents 20% frontend completion and only a 10% backend slice covering MobileGaze integration, calibration/mapping, diagnostics, smoothing, and current backend limitations.
- A user-provided Chapter 5 Word document on the Desktop was updated as `C:\Users\MNA\Desktop\VisiMove_Chapter_5_Implementation_Backend_Updated.docx`. The existing frontend content was preserved, and backend text was added only up to webcam capture plus face/eye detection for the 10% backend implementation scope.
- The in-repo Chapter 5 document `VisiMove_Chapter_5_Implementation_Backend_Updated.docx` now includes the 10% backend scope under the required `5.1` to `5.4` format and embeds a backend face/eye detection preview figure. A backup of the previous version exists at `VisiMove_Chapter_5_Implementation_Backend_Updated.backup.docx`.
- The attempted median-window/demo-stable cursor tuning was rejected because it made the cursor feel worse and still did not land on intended targets. Those code/config changes have been reverted to the previous working smoothing behavior.
- Latest debug showed cursor stops because `live_tracking_quality=unstable` caused `cursor_skip=live gaze outside calibrated domain` and made the smoother report `missing`, even for very low violation ratios such as `0.03`.
- Runtime gating now blocks cursor/smoothing only when live quality becomes `unsafe`; `unstable` remains visible in debug but no longer freezes movement by default.
- Adaptive smoothing now includes large-jump confirmation. Isolated one-frame MobileGaze target spikes are held as `smoothing_state=confirming`, while repeated large targets are accepted and released into normal movement.
- Adaptive smoothing now includes edge catch-up. When `adjusted_after_gain` reaches a physical screen edge or corner, the smoother uses faster edge movement and snaps the final few pixels instead of holding short of the edge.

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

EyeTrax real gaze works but is not reliable enough for the current setup because Y-axis output can become saturated/pinned. GazeFollower runs with `fallback=no`, but guided diagnostics showed weak directional separation. MobileGaze is currently the best real gaze backend. The latest saved MobileGaze profile now uses bounded IDW mapping and preview is stable with cursor disabled. Cursor can be tested carefully with edge reach and edge boost disabled, but raw-sample error is still high enough that pinpoint accuracy should not be expected yet.

## Next Required Steps

1. Use `docs/VisiMove_Chapter_5_Implementation_30_Percent.docx` when the report must show only 30% completed work: 20% frontend and 10% backend.
2. Test the current MobileGaze profile with the restored previous smoothing settings and edge reach enabled.
3. Watch whether `cursor_skip` still appears. It should not stop for `live_tracking_quality=unstable`; it should only stop when quality reaches `unsafe`, when the cursor is paused, or when face/eyes are lost.
4. Keep edge boost disabled unless edge reach alone fails after smoothing is fixed.
5. If cursor remains systematically off target after smoothing is stable, add a small guided bias-correction or calibration-offset step instead of hardcoded gain/offset tuning.
6. Re-run `diagnose_calibration_mapping.py` after any new calibration and confirm no warnings and `0%` raw-sample clipping.
7. Watch `smoothing_state`; it should settle/hold during fixation instead of staying mostly `moving`.
8. After cursor movement is usable, integrate real blink or dwell click.
9. Integrate OCEC/ONNX blink backend.
10. Add dwell click.
11. Optimize performance.
12. Prepare final demo mode.

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

### 2026-05-16 - GazeFollower Native Output Review

What changed:

- Recorded the latest GazeFollower preview result.
- Confirmed GazeFollower starts with `fallback=no` and `face_found=yes`.
- Confirmed the previous `mediapipe.solutions` and `pandas` runtime errors are resolved.
- Captured that GazeFollower native output is small/negative, for example `native_gaze=(0.367,-9.740)`, while normalized `raw_gaze` remains pinned at `(0.000,0.000)`.
- Marked GazeFollower as not ready for calibration or cursor control until native output units are understood.

Files modified:

- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/current_status.md`
- `docs/calibration.md`

Commands to run:

```powershell
python scripts\run_tracking.py --gaze-backend gazefollower --show-debug
```

Known issues:

- GazeFollower native coordinate units are not yet understood.
- Current adapter labels the output as `screen_pixels`, but the values look more like pre-calibration or physical-coordinate output.
- `raw_gaze` is pinned at `(0,0)` after normalization, so cursor and calibration are unsafe.
- Existing `data/calibration/user_profile.json` is old/degenerate and not valid for GazeFollower.

Next step:

- Inspect GazeFollower calibration/model code to identify the correct interpretation of `raw_gaze_coordinates` before changing normalization or adding calibration.

### 2026-05-16 - GazeFollower Native Coordinate Normalization

What changed:

- Inspected GazeFollower `MGazeNetGazeEstimator`, `SVRCalibration`, `CalibrationController`, and runtime sampling flow.
- Confirmed `raw_gaze_coordinates` is `res[:2]` from the MNN model output.
- Confirmed GazeFollower's own screen-coordinate path uses SVR calibration over the full feature vector and only then converts calibrated values to pixels.
- Fixed the VisiMove GazeFollower adapter so native output is no longer treated as Windows screen pixels.
- Added config-driven centered `tanh` normalization for GazeFollower model coordinates.
- Added tests proving native GazeFollower values are not normalized as screen pixels and that scale is configurable.
- Updated README, architecture, calibration, current status, and test suite docs.

Files modified:

- `src/visimove/gaze/gazefollower_adapter.py`
- `config/default.yaml`
- `src/visimove/tests/test_gaze_adapters.py`
- `README.md`
- `docs/architecture.md`
- `docs/calibration.md`
- `docs/current_status.md`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
python scripts\run_tracking.py --gaze-backend gazefollower --show-debug
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe -c "from visimove.config import load_config; from visimove.pipeline.realtime_pipeline import build_gaze_backend_config; cfg=load_config('config/default.yaml'); cfg['gaze']['gaze_backend']='gazefollower'; cfg['gaze']['backend']='gazefollower'; print(build_gaze_backend_config(cfg)['native_output_mode'], build_gaze_backend_config(cfg)['native_coordinate_scale_x'], build_gaze_backend_config(cfg)['native_coordinate_scale_y'])"
```

Result:

- Compile check passed.
- `92 passed in 5.94s`.
- Tracking CLI help passed.
- GazeFollower config smoke check printed `model_coordinates 10.0 10.0`.

Known issues:

- GazeFollower preview still needs a manual face-visible run after the normalization fix.
- GazeFollower is not calibrated for VisiMove cursor control yet.
- Existing `data/calibration/user_profile.json` is old/degenerate and not valid for GazeFollower.
- Cursor must remain disabled until preview output is separable and a matching calibration profile is created.

Next step:

- Run GazeFollower preview and verify `native_units=gazefollower_model_coordinates`, `raw_gaze` is not pinned, and raw gaze changes meaningfully when looking left/right/up/down.

### 2026-05-16 - MobileGaze Weight Preparation

What changed:

- Ran the external repository setup helper and confirmed all external backend folders are present.
- Verified `external/mobilegaze/weights` only contained `.gitkeep`.
- Downloaded the official `mobileone_s0_gaze.onnx` MobileGaze release weight into `external/mobilegaze/weights/`.
- Checked runtime dependencies and found `onnxruntime`, `torch`, `torchvision`, and `uniface` are not installed in the current `.venv`.
- Inspected the current `MobileGazeAdapter` shell and confirmed it still needs ONNX inference wiring.

Files modified:

- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Local generated files:

- `external/mobilegaze/weights/mobileone_s0_gaze.onnx` (ignored by git)

Commands to run:

```powershell
.\.venv\Scripts\python.exe -m pip install onnxruntime
```

Known issues:

- MobileGaze adapter is still a shell and will not run real inference yet.
- MobileGaze ONNX output is yaw/pitch in radians, not screen coordinates.
- Need to decide whether to use VisiMove face crops or MobileGaze's upstream `uniface` detector path.
- Cursor must remain disabled.

Next step:

- Install `onnxruntime`, then implement the MobileGaze ONNX adapter in preview/debug mode only.

### 2026-05-16 - MobileGaze ONNX Preview Adapter

What changed:

- Implemented `src/visimove/gaze/mobilegaze_adapter.py` as an ONNX Runtime preview adapter.
- Added MobileGaze setup validation for repo files, ONNX model path, and `onnxruntime`/OpenCV dependencies.
- Added MobileGaze config defaults pointing at `external/mobilegaze/weights/mobileone_s0_gaze.onnx`.
- Updated gaze backend selection so `--gaze-backend mobilegaze` uses the MobileGaze config and safe dummy fallback.
- Updated startup summary and warnings for MobileGaze readiness.
- Converted MobileGaze native yaw/pitch radians into normalized raw gaze for preview and later VisiMove calibration.
- Added tests for missing repo/model behavior, angle-to-raw conversion, and MobileGaze backend config merging.
- Updated README, architecture, calibration, setup, current status, and test suite docs.

Files modified:

- `src/visimove/gaze/mobilegaze_adapter.py`
- `src/visimove/pipeline/realtime_pipeline.py`
- `scripts/run_tracking.py`
- `config/default.yaml`
- `pyproject.toml`
- `src/visimove/tests/test_gaze_adapters.py`
- `README.md`
- `docs/architecture.md`
- `docs/calibration.md`
- `docs/current_status.md`
- `docs/setup_external_repos.md`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
python scripts\run_tracking.py --gaze-backend mobilegaze --show-debug
python scripts\diagnose_live_gaze.py --gaze-backend mobilegaze --guided
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -c "from visimove.config import load_config; from visimove.pipeline.realtime_pipeline import build_gaze_backend_config, build_gaze_model; cfg=load_config('config/default.yaml'); cfg['gaze']['gaze_backend']='mobilegaze'; cfg['gaze']['backend']='mobilegaze'; model=build_gaze_model(build_gaze_backend_config(cfg)); model._initialize(); print(type(model).__name__, getattr(model, '_initialized', False), model._input_size, model._output_names)"
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

- MobileGaze ONNX session initialized with input size `(448, 448)` and outputs `['yaw', 'pitch']`.
- Compile check passed.
- `96 passed in 4.45s`.
- Tracking CLI help passed.

Known issues:

- MobileGaze still needs live webcam preview validation.
- MobileGaze is not calibrated for VisiMove cursor control yet.
- Existing default calibration profile is old/degenerate and not valid for MobileGaze.
- Cursor must remain disabled until MobileGaze preview and calibration quality are verified.

Next step:

- Run MobileGaze preview and guided diagnostics with cursor disabled.

### 2026-05-16 - MobileGaze Live Diagnostic And X Sign Fix

What changed:

- Reviewed the user's MobileGaze preview/debug output.
- Confirmed MobileGaze runs with `fallback=no`, `face_found=yes`, and native `radians_yaw_pitch`.
- Guided diagnostics showed useful raw gaze movement:
  - left mean raw_x around `0.657`
  - center mean raw_x around `0.571`
  - right mean raw_x around `0.477`
  - top/bottom raw_y separation was also visible.
- Identified a true horizontal sign bug because left gaze produced larger `raw_x` than right gaze.
- Fixed MobileGaze yaw-to-raw mapping so left becomes lower `raw_x`, center stays near `0.5`, and right becomes higher `raw_x`.
- Updated tests and docs.

Files modified:

- `src/visimove/gaze/mobilegaze_adapter.py`
- `src/visimove/tests/test_gaze_adapters.py`
- `docs/calibration.md`
- `docs/current_status.md`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
python scripts\run_tracking.py --gaze-backend mobilegaze --show-debug
python scripts\diagnose_live_gaze.py --gaze-backend mobilegaze --guided
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe -c "from visimove.gaze.mobilegaze_adapter import MobileGazeAdapter; import numpy as np; print('left', MobileGazeAdapter._angles_to_raw(np.radians(-45),0)); print('center', MobileGazeAdapter._angles_to_raw(0,0)); print('right', MobileGazeAdapter._angles_to_raw(np.radians(45),0))"
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

- Compile check passed.
- `96 passed in 4.79s`.
- Angle mapping sanity check printed `left (0.0, 0.5)`, `center (0.5, 0.5)`, `right (1.0, 0.5)`.
- Tracking CLI help passed.

Known issues:

- MobileGaze needs one more guided diagnostic after the sign fix.
- MobileGaze calibration profile has not been created yet.
- Existing default calibration profile is old/degenerate and not valid for MobileGaze.
- Cursor must remain disabled.

Next step:

- Rerun MobileGaze guided diagnostics and confirm left/center/right order is correct before calibration.

### 2026-05-16 - MobileGaze Calibration Wiring

What changed:

- Added `MobileGazeCalibrationGazeProvider` to read webcam frames, run the selected detector, and collect real MobileGaze ONNX gaze output.
- Updated `scripts/run_calibration.py` so `--gaze-backend mobilegaze` refuses calibration if MobileGaze is not ready.
- MobileGaze calibration now saves to `data/calibration/user_profile_mobilegaze.json` by default.
- Tracking now auto-selects `user_profile_mobilegaze.json` for `--gaze-backend mobilegaze` when the profile exists.
- Updated calibration/current-status/test docs.

Files modified:

- `src/visimove/calibration/mobilegaze_provider.py`
- `scripts/run_calibration.py`
- `scripts/run_tracking.py`
- `docs/calibration.md`
- `docs/current_status.md`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
python scripts\run_calibration.py --gaze-backend mobilegaze --verbose-quality
python scripts\run_tracking.py --gaze-backend mobilegaze --show-debug
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_calibration.py --help
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

- Compile check passed.
- `96 passed in 4.32s`.
- Calibration and tracking CLI help passed.

Known issues:

- MobileGaze calibration still needs to be run by the user.
- Cursor must remain disabled until MobileGaze calibration quality is good or acceptable and live preview is stable.

Next step:

- Run MobileGaze calibration, then preview using the saved MobileGaze profile.

### 2026-05-16 - MobileGaze Good Calibration Preview

What changed:

- Recorded the user's MobileGaze post-calibration preview output.
- Confirmed MobileGaze runs with `fallback=no`, `face_found=yes`, and `calibration_quality=good`.
- Confirmed tracking auto-loads the MobileGaze calibration profile and reports calibrated raw domain around `x[0.264,0.650], y[0.381,0.913]`.
- Confirmed mapped screen coordinates now vary across the screen, for example around `(389,561)`, `(951,687)`, `(1268,1061)`, and `(2017,1068)`.
- Kept cursor disabled by default; the next step is a careful cursor test only after another stable preview.

Files modified:

- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/current_status.md`
- `docs/calibration.md`
- `README.md`

Commands to run:

```powershell
python scripts\run_tracking.py --gaze-backend mobilegaze --show-debug
python scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor
```

Known issues:

- Some preview lines still report `live_tracking_quality=unstable` even when raw domain violations are `0.00`; watch this before enabling cursor.
- Blink click is still dummy/disabled for real clicking.
- EyeTrax and GazeFollower are not recommended for this setup.

Next step:

- Run one more MobileGaze preview with cursor disabled. If it stays mostly stable and mapped coordinates move naturally, test cursor carefully with `--enable-cursor`.

### 2026-05-16 - MobileGaze Live Clipping Unsafe Preview

What changed:

- Recorded the user's latest MobileGaze preview output after good calibration.
- Confirmed MobileGaze still runs with `fallback=no`, `face_found=yes`, and `calibration_quality=good`.
- Confirmed live raw gaze is usually inside the calibrated raw domain, but mapped output often clips at the screen boundaries.
- Observed mapped output examples clipping to right/bottom edges, such as `mapped_after_clamp=(2559,1132)`, `(2512,1439)`, `(0,1439)`, and frequent `live_tracking_quality=unsafe`.
- Updated the recommended next step: do not enable cursor yet; diagnose mapping/clipping or recalibrate first.

Files modified:

- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/current_status.md`
- `docs/calibration.md`

Commands to run:

```powershell
python scripts\diagnose_calibration_mapping.py --profile data\calibration\user_profile_mobilegaze.json
python scripts\diagnose_live_gaze.py --gaze-backend mobilegaze --guided
```

Known issues:

- MobileGaze raw output is real and calibration quality is good, but live mapped coordinates can extrapolate beyond the screen and get clamped.
- `live_tracking_quality=unsafe` blocks cursor control correctly.
- Blink click is still not integrated.

Next step:

- Diagnose the MobileGaze calibration mapping and guided live gaze coverage before any cursor test.

### 2026-05-16 - MobileGaze Live Diagnostic Profile Fix

What changed:

- Reviewed the user's MobileGaze calibration mapping diagnostics.
- Confirmed the saved MobileGaze calibration profile itself is healthy: affine mapping, no warnings, no clipping on calibration sample means, and mapped X/Y ranges covering the expected screen targets.
- Identified that `scripts/diagnose_live_gaze.py --gaze-backend mobilegaze --guided` was loading the old default `data/calibration/user_profile.json` instead of `data/calibration/user_profile_mobilegaze.json`.
- Fixed `diagnose_live_gaze.py` to auto-load the backend-specific MobileGaze profile, matching `run_tracking.py`.
- Added regression tests for MobileGaze backend-specific profile selection and explicit profile override behavior.

Files modified:

- `scripts/diagnose_live_gaze.py`
- `src/visimove/tests/test_diagnose_live_gaze.py`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/current_status.md`
- `docs/calibration.md`
- `docs/test_suite.md`

Commands to run:

```powershell
python scripts\diagnose_live_gaze.py --gaze-backend mobilegaze --guided
python scripts\run_tracking.py --gaze-backend mobilegaze --show-debug
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\diagnose_live_gaze.py --help
```

Result:

- Compile check passed.
- `98 passed in 5.90s`.
- Live gaze diagnostic CLI help passed.

Known issues:

- MobileGaze cursor should remain disabled until the corrected guided diagnostic confirms stable mapped coverage.
- Previous guided diagnostic output with mapped `(1283,721)` for every target should be ignored because it used the wrong profile.
- Blink click is still not integrated.

Next step:

- Rerun MobileGaze guided diagnostics with the fixed script.

### 2026-05-16 - MobileGaze Corrected Guided Diagnostic

What changed:

- Recorded the corrected MobileGaze guided diagnostic output after fixing profile selection.
- Confirmed MobileGaze now maps guided targets differently instead of staying at the old center point.
- Observed useful horizontal separation: left mean mapped X around `88`, center around `1146`, and right around `2373`.
- Observed useful vertical separation: top mean mapped Y around `149` and bottom around `1215`.
- Confirmed the diagnostic no longer reports strong horizontal bias.

Files modified:

- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/current_status.md`
- `docs/calibration.md`

Commands to run:

```powershell
python scripts\run_tracking.py --gaze-backend mobilegaze --show-debug
python scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor
```

Known issues:

- Mapped output is usable but noisy and can touch screen edges.
- Cursor should be enabled only after one final cursor-disabled preview shows mostly stable live tracking.
- Blink click is still not integrated.

Next step:

- Run MobileGaze preview with cursor disabled. If `live_tracking_quality` stays mostly stable, perform a careful cursor test.

### 2026-05-16 - Mapping Input Domain Clamp

What changed:

- Reviewed the user's final MobileGaze preview where raw gaze was usually inside the saved raw calibration domain, but affine mapping still produced off-screen values before clamping.
- Identified the root cause: the saved raw calibration domain comes from all raw samples, while affine mapping is fit from calibration point means. Values inside the broad sample domain can still be outside the affine model's fitted input range.
- Added fitted input-domain storage to mapping model parameters for newly trained profiles.
- Added fallback fitted-domain derivation for existing affine profiles using coefficients and target coordinates.
- Updated runtime mapping to clamp raw input to the fitted mapping domain before prediction.
- Added tests for old-profile affine extrapolation and mapping model input-domain serialization.

Files modified:

- `src/visimove/calibration/mapping_model.py`
- `src/visimove/calibration/mapper.py`
- `src/visimove/pipeline/realtime_pipeline.py`
- `config/default.yaml`
- `src/visimove/tests/test_mapper.py`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/current_status.md`
- `docs/calibration.md`
- `docs/test_suite.md`

Commands to run:

```powershell
python scripts\run_tracking.py --gaze-backend mobilegaze --show-debug
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
- `100 passed in 8.17s`.
- Tracking and live diagnostic CLI help passed.

Known issues:

- Need a live webcam preview to confirm edge clipping is reduced on the current setup.
- Cursor should remain disabled until that preview is stable.
- Blink click is still not integrated.

Next step:

- Run MobileGaze preview with cursor disabled and check `mapped_raw_before_screen_clamp` no longer jumps far beyond screen bounds.

### 2026-05-16 - MobileGaze Post-Clamp Preview

What changed:

- Recorded the user's MobileGaze preview after the mapping input-domain clamp.
- Confirmed dangerous off-screen affine extrapolation is fixed: mapped predictions now stay bounded to the calibrated target area instead of jumping to values such as `x=3108`.
- Observed mapped input clamping to fitted ranges, for example `mapped_input_after_domain_clamp=(0.556,0.528)` mapping to `(2252,173)`.
- Confirmed `fallback=no`, `face_found=yes`, and `calibration_quality=good` remain intact.
- Noted that `live_tracking_quality` can still become `unstable` because raw gaze sometimes exceeds the calibrated domain/margin, but this is now a controlled safety warning.

Files modified:

- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/current_status.md`
- `docs/calibration.md`

Commands to run:

```powershell
python scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor
```

Known issues:

- Cursor movement may be blocked on frames where live quality is `unstable`.
- Use `--allow-unstable-live-gaze` only for a short controlled experiment if normal cursor mode remains blocked.
- Blink click is still not integrated.

Next step:

- Carefully test MobileGaze cursor movement.

### 2026-05-16 - MobileGaze Cursor Test Blocked

What changed:

- Recorded the user's first MobileGaze cursor-enabled run.
- The pipeline started with MobileGaze preflight passing, and pause/resume hotkey events worked.
- The cursor did not visibly move.
- Because the run did not include `--show-debug`, the exact `cursor_skip` reason was not visible.
- Real blink detection did not work because no real blink backend is integrated yet; the active blink backend remains dummy/keyboard-triggered.

Files modified:

- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/current_status.md`
- `docs/calibration.md`

Commands to run:

```powershell
python scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --allow-unstable-live-gaze --show-debug
```

Known issues:

- Cursor may be skipped due pause state, live-quality gating, low confidence, or no face/eyes.
- Need `--show-debug` to see `cursor_skip`.
- Real blink detection is not implemented; use `b` only for fake blink testing until ONNX/OCEC blink backend is integrated.

Next step:

- Run the debug cursor command, press `p` once to resume, and inspect `cursor_skip`.

### 2026-05-16 - MobileGaze Cursor And Blink Follow-Up

What changed:

- Recorded the user's MobileGaze cursor-enabled output with repeated cursor pause/resume hotkey messages.
- Confirmed the cursor controller is receiving hotkey state changes, but the run did not include `--show-debug`, so the exact cursor movement skip reason is still unknown.
- Confirmed natural blink detection is not expected to work yet because the active blink backend remains dummy; real OCEC/ONNX blink integration is still pending.

Files modified:

- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
python scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --allow-unstable-live-gaze --show-debug
```

Known issues:

- Cursor may still be skipped because it is paused, live gaze is unstable, confidence is low, no face is detected, or the Windows cursor backend needs debugging.
- Real blink detection is not integrated yet; dummy blink does not detect natural blinks.

Next step:

- Run the debug cursor command, press `p` once to resume, and copy the lines containing `cursor_skip=...`.

### 2026-05-16 - Cursor Backend Debug CLI

What changed:

- Reviewed the user's MobileGaze cursor debug output and confirmed `cursor_skip=none` after pressing `p`, so the real-time pipeline is issuing cursor movement requests.
- Added `--cursor-backend pyautogui/win32/dryrun` to `scripts/run_tracking.py`.
- Added `scripts/test_cursor_backend.py` to smoke-test PyAutoGUI vs Win32 cursor movement directly.
- Updated README, calibration docs, and current status with the cursor backend troubleshooting flow.

Files modified:

- `scripts/run_tracking.py`
- `scripts/test_cursor_backend.py`
- `src/visimove/tests/test_run_tracking_startup.py`
- `README.md`
- `docs/calibration.md`
- `docs/current_status.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\test_cursor_backend.py --backend pyautogui
.\.venv\Scripts\python.exe scripts\test_cursor_backend.py --backend win32
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --allow-unstable-live-gaze --show-debug --cursor-backend win32
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\test_cursor_backend.py --help
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

- Compile check passed.
- Tracking help shows `--cursor-backend`.
- Cursor backend test help passed.
- `100 passed in 7.11s`.

Known issues:

- Natural blink detection is still not integrated; dummy blink does not detect real blinks.
- Need the user to test whether PyAutoGUI or Win32 visibly moves the Windows pointer.

Next step:

- Run the cursor backend smoke tests, then use `--cursor-backend win32` if Win32 moves correctly.

### 2026-05-16 - MobileGaze Cursor Jitter Tuning

What changed:

- Recorded the user's latest MobileGaze cursor output: cursor now follows eye movement with `cursor_skip=none`, `fallback=no`, `face_found=yes`, and `calibration_quality=good`.
- Confirmed the remaining issue is not backend failure but movement quality: cursor is not pinpoint accurate and jumps.
- Added CLI tuning options for quick stability tests without editing YAML:
  - `--smoothing-filter`
  - `--ema-alpha`
  - `--max-speed-px-per-sec`
- Updated README, calibration docs, and current status with the recommended smoother MobileGaze cursor command.

Files modified:

- `scripts/run_tracking.py`
- `src/visimove/tests/test_run_tracking_startup.py`
- `README.md`
- `docs/calibration.md`
- `docs/current_status.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --allow-unstable-live-gaze --show-debug --cursor-backend win32 --smoothing-filter ema --ema-alpha 0.18 --max-speed-px-per-sec 700
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

- Compile check passed.
- Tracking help shows the new smoothing and speed CLI flags.
- `101 passed in 7.12s`.

Known issues:

- MobileGaze cursor is usable but still noisy; it needs tuning and possibly another calmer calibration pass.
- Natural blink detection is still not integrated.

Next step:

- Test stronger smoothing and slower cursor speed; then compare lag versus jitter.

### 2026-05-16 - MobileGaze Edge Reach Observation

What changed:

- Recorded the user's observation that MobileGaze cursor motion is shaky rather than violently jumpy.
- Confirmed cursor movement follows gaze but does not land exactly at the looked-at location.
- Noted that the cursor appears blocked near screen extremes because current calibration targets and fitted mapping-domain clamps bound output to the inner calibration area rather than the full physical screen edges.

Files modified:

- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --allow-unstable-live-gaze --show-debug --cursor-backend win32 --smoothing-filter ema --ema-alpha 0.18 --max-speed-px-per-sec 700
```

Known issues:

- MobileGaze cursor is real but not yet precise enough for final use.
- Current mapping-domain clamp and calibration target layout can prevent reaching exact screen edges.
- Natural blink detection is still not integrated.

Next step:

- Add a safe edge-reach calibration/mapping mode: distinguish protective raw-input clamping from full-screen output scaling, then add diagnostics for edge reach.

### 2026-05-16 - Safe Edge Reach Mapping Mode

What changed:

- Implemented safe edge-reach mapping mode.
- The mapper still clamps raw gaze input to the safe calibrated/fitted domain, then expands the calibrated target rectangle to the full screen as a separate output step.
- Added config options:
  - `calibration.edge_reach_enabled`
  - `calibration.edge_margin_px`
- Added CLI options:
  - `--edge-reach`
  - `--no-edge-reach`
  - `--edge-margin-px`
- Added debug output for `edge_reach_after`, `edge_reach_enabled`, and `edge_margin_px`.
- Updated docs and tests.

Files modified:

- `config/default.yaml`
- `scripts/run_tracking.py`
- `src/visimove/calibration/axis_adjustment.py`
- `src/visimove/pipeline/realtime_pipeline.py`
- `src/visimove/tests/test_axis_adjustment.py`
- `src/visimove/tests/test_run_tracking_startup.py`
- `README.md`
- `docs/calibration.md`
- `docs/current_status.md`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --allow-unstable-live-gaze --show-debug --cursor-backend win32 --smoothing-filter ema --ema-alpha 0.18 --max-speed-px-per-sec 700 --edge-reach --edge-margin-px 0
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

- Compile check passed.
- Tracking help shows edge-reach CLI flags.
- `104 passed in 4.89s`.

Known issues:

- MobileGaze cursor may still be shaky because the gaze model output is noisy and FPS is modest.
- Edge reach improves boundary coverage but does not make gaze pinpoint accurate by itself.
- Natural blink detection is still not integrated.

Next step:

- Test MobileGaze cursor with edge reach and smoothing, then decide whether to tune smoothing further or recalibrate with steadier point collection.

### 2026-05-17 - MobileGaze Accuracy And Stability Observation

What changed:

- Recorded the user's safe edge-reach test observation.
- Cursor now moves closer to desired locations, but still does not reliably reach the taskbar or the symmetric top edge area.
- Cursor is still slightly off target and shaky: it approaches the looked-at target but can overshoot or drift, and it does not stay still when the eye is still.
- The next problem is no longer backend selection. It is runtime stabilization plus edge/taskbar reach validation.

Files modified:

- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --allow-unstable-live-gaze --show-debug --cursor-backend win32 --smoothing-filter ema --ema-alpha 0.12 --max-speed-px-per-sec 500 --edge-reach --edge-margin-px 0
```

Known issues:

- MobileGaze cursor is usable but not pinpoint accurate.
- Cursor still has micro-shake when gaze is still.
- Taskbar/top-edge reach needs a dedicated diagnostic because Windows screen bounds, calibration target area, and edge-reach output may not match perceived usable screen edges.
- Natural blink detection is still not integrated.

Next step:

- Implement adaptive stabilization/fixation hold and a taskbar/edge reach diagnostic before working on blink clicks.

### 2026-05-17 - Adaptive Stabilization Filter

What changed:

- Added `AdaptiveSmoothingFilter`.
- Adaptive smoothing uses faster smoothing for clear intentional movement and slow/hold behavior for fixation jitter.
- Added CLI options:
  - `--smoothing-filter adaptive`
  - `--adaptive-fast-alpha`
  - `--adaptive-slow-alpha`
  - `--adaptive-fixation-radius`
  - `--adaptive-release-radius`
  - `--adaptive-hold-ms`
- Added debug output field `smoothing_state`.
- Updated default smoothing filter to `adaptive`.
- Updated README, calibration docs, current status, test suite, and tests.

Files modified:

- `src/visimove/smoothing/adaptive_filter.py`
- `src/visimove/smoothing/__init__.py`
- `src/visimove/pipeline/realtime_pipeline.py`
- `scripts/run_tracking.py`
- `config/default.yaml`
- `src/visimove/tests/test_smoothing.py`
- `src/visimove/tests/test_run_tracking_startup.py`
- `README.md`
- `docs/calibration.md`
- `docs/current_status.md`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --allow-unstable-live-gaze --show-debug --cursor-backend win32 --smoothing-filter adaptive --adaptive-fast-alpha 0.34 --adaptive-slow-alpha 0.08 --adaptive-fixation-radius 32 --adaptive-release-radius 90 --adaptive-hold-ms 120 --max-speed-px-per-sec 850 --edge-reach --edge-margin-px 0
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

- Compile check passed.
- Tracking help shows adaptive smoothing flags.
- `107 passed in 15.52s`.

Known issues:

- Adaptive stabilization still needs live validation.
- Taskbar/top-edge reach may still need a dedicated diagnostic after stabilization.
- Natural blink detection is still not integrated.

Next step:

- Test adaptive stabilization live and watch `smoothing_state`; if top-right/taskbar reach remains poor, add the edge reach diagnostic next.

### 2026-05-17 - Edge Boost For On-Screen Edge Reach

What changed:

- Recorded the user's observation that the cursor reaches extreme edges only when looking off-screen.
- Added edge boost after safe edge reach so on-screen near-edge gaze can move farther toward top/bottom/left/right boundaries.
- Added config options:
  - `calibration.edge_boost_enabled`
  - `calibration.edge_boost_gamma`
- Added CLI options:
  - `--edge-boost`
  - `--no-edge-boost`
  - `--edge-boost-gamma`
- Tuned adaptive defaults to hold fixation more firmly.
- Updated docs and tests.

Files modified:

- `config/default.yaml`
- `scripts/run_tracking.py`
- `src/visimove/calibration/axis_adjustment.py`
- `src/visimove/pipeline/realtime_pipeline.py`
- `src/visimove/tests/test_axis_adjustment.py`
- `src/visimove/tests/test_run_tracking_startup.py`
- `README.md`
- `docs/calibration.md`
- `docs/current_status.md`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --allow-unstable-live-gaze --show-debug --cursor-backend win32 --smoothing-filter adaptive --adaptive-fast-alpha 0.34 --adaptive-slow-alpha 0.06 --adaptive-fixation-radius 42 --adaptive-release-radius 110 --adaptive-hold-ms 90 --max-speed-px-per-sec 850 --edge-reach --edge-margin-px 0 --edge-boost --edge-boost-gamma 0.78
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

- Compile check passed.
- Tracking help shows edge boost flags.
- `108 passed in 4.29s`.

Known issues:

- Edge boost still needs live validation.
- Cursor may still not be pixel-perfect because MobileGaze is approximate.
- Natural blink detection is still not integrated.

Next step:

- Test edge boost live; use gamma `0.82` if too aggressive or `0.72` if edge controls still require off-screen gaze.

### 2026-05-17 - MobileGaze Vertical Bias Observation

What changed:

- Recorded the user's observation that the cursor consistently stays below the looked-at target by roughly 3-4 cm.
- Looking off-screen is still needed to make the cursor reach the desired vertical location.
- Cursor remains shaky/jumpy.
- This points to a systematic vertical calibration bias plus noisy MobileGaze output, not only an edge-reach problem.

Files modified:

- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --allow-unstable-live-gaze --show-debug --cursor-backend win32 --smoothing-filter adaptive --adaptive-fast-alpha 0.30 --adaptive-slow-alpha 0.05 --adaptive-fixation-radius 48 --adaptive-release-radius 120 --adaptive-hold-ms 70 --max-speed-px-per-sec 750 --edge-reach --edge-margin-px 0 --edge-boost --edge-boost-gamma 0.84 --vertical-offset -220
```

Known issues:

- Vertical offset is a temporary runtime correction; a better long-term fix is either a cleaner MobileGaze recalibration or a guided bias-correction calibration step.
- Edge boost can amplify noise if too strong.
- Natural blink detection is still not integrated.

Next step:

- Test vertical offset and reduced edge boost. If it improves alignment, add a guided bias-calibration command to learn vertical/horizontal offsets from center/top/bottom targets.

### 2026-05-17 - MobileGaze Linear Baseline And Edge Reach Reset

What changed:

- Recorded the user's latest MobileGaze startup and debug output.
- Confirmed the new MobileGaze profile now loads with `calibration model type: linear` and `calibration_quality=good`.
- Confirmed MobileGaze raw gaze is usually inside the saved calibration raw domain and mapped screen coordinates are active.
- Identified safe edge reach as the remaining amplification source: it expands near-edge calibrated outputs such as `(2252,150)` to full-screen corners such as `(2559,0)`.
- Disabled edge reach by default so the next test can evaluate the linear MobileGaze baseline without edge expansion.

Files modified:

- `config/default.yaml`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/current_status.md`

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --show-debug --no-edge-reach --no-edge-boost --vertical-offset 0
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m json.tool docs\PROJECT_STATE.json
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\run_calibration.py --help
.\.venv\Scripts\python.exe scripts\diagnose_calibration_mapping.py --profile data\calibration\user_profile_mobilegaze.json
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

- Compile check passed.
- `PROJECT_STATE.json` is valid JSON.
- Tracking and calibration help passed.
- Diagnostics correctly still flag the saved affine profile as needs-review-worthy until recalibration.
- `122 passed in 13.89s`.

Known issues:

- MobileGaze cursor is still not pinpoint accurate.
- The latest log had cursor disabled, so it was a preview/diagnostic run, not a cursor movement test.
- Edge reach may still be useful later, but only after the no-edge-reach baseline is stable.
- Real blink detection is still not integrated.

Next step:

- Run MobileGaze preview without edge reach or edge boost; if stable, test cursor with the same baseline settings and press `p` once to resume.

### 2026-05-18 - MobileGaze Cursor Mapping Logic Review

What changed:

- Inspected the MobileGaze adapter, calibration provider, calibration UI, mapping model, mapper, axis adjustment, smoothing filter, cursor safety, and cursor backends.
- Ran calibration mapping diagnostics on `data/calibration/user_profile_mobilegaze.json`.
- Found the main cursor placement issue: the current saved MobileGaze profile is stored as `good`, but its learned linear mapping misses calibration point means by about `480px` X and `235px` Y.
- Fixed mapping diagnostics to preserve calibration collection order, matching the live calibration UI output.
- Added mapping-error thresholds to calibration quality so high mapped point error causes `needs_review` or `poor` instead of `good`.
- Added startup-time mapping diagnostics so old saved profiles can be downgraded to `needs_review` even if their JSON quality field says `good`.
- Changed MobileGaze calibration defaults from `linear/all_samples` to `auto/point_means`.
- Added auto mapping selection in calibration: candidate models are trained and the model with the lowest normalized point-mean screen error is saved.
- Confirmed the current saved profile is now treated as `needs_review` at startup because of high mapping error.

Files modified:

- `config/default.yaml`
- `config/calibration.yaml`
- `scripts/run_calibration.py`
- `scripts/run_tracking.py`
- `src/visimove/calibration/calibration_ui.py`
- `src/visimove/calibration/calibration_quality.py`
- `src/visimove/calibration/mapping_diagnostics.py`
- `src/visimove/tests/test_calibration_quality.py`
- `src/visimove/tests/test_run_tracking_startup.py`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/current_status.md`
- `docs/calibration.md`
- `docs/test_suite.md`

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_calibration.py --gaze-backend mobilegaze --verbose-quality
.\.venv\Scripts\python.exe scripts\diagnose_calibration_mapping.py --profile data\calibration\user_profile_mobilegaze.json
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --show-debug --no-edge-reach --no-edge-boost --vertical-offset 0
```

Known issues:

- The current saved MobileGaze profile should not be used for cursor control because mapping diagnostics show high screen error.
- Cursor placement depends on recalibrating with the new auto point-mean mapper.
- MobileGaze raw output remains noisy; even with better mapping it may need smoothing and possibly guided bias correction.
- Real blink detection is still not integrated.

Next step:

- Recalibrate MobileGaze, verify mapping diagnostics warnings are gone or much smaller, then preview before enabling cursor.

### 2026-05-18 - MobileGaze Polynomial Instability Guard

What changed:

- Reviewed the user's latest MobileGaze polynomial profile diagnostics and live debug output.
- Confirmed the cursor should not be enabled because normal in-domain live samples were mapping far outside the screen before clamping.
- Added raw-sample stability diagnostics in addition to calibration point-mean diagnostics.
- Current polynomial profile is now flagged: `46%` of raw calibration samples map outside the screen, raw-sample MAE is about `563.7px` X and `407.9px` Y.
- Updated MobileGaze auto mapping scoring to penalize candidates that fit point means but map individual raw samples off-screen.
- Verified the updated auto selector would pick `affine` instead of `polynomial` on the current MobileGaze profile data.
- Updated calibration quality handling, diagnostics output, tests, and docs.

Files modified:

- `scripts/diagnose_calibration_mapping.py`
- `src/visimove/calibration/calibration_ui.py`
- `src/visimove/calibration/calibration_quality.py`
- `src/visimove/calibration/mapping_diagnostics.py`
- `src/visimove/tests/test_calibration_quality.py`
- `src/visimove/tests/test_mapping_diagnostics.py`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/current_status.md`
- `docs/calibration.md`
- `docs/test_suite.md`

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_calibration.py --gaze-backend mobilegaze --verbose-quality
.\.venv\Scripts\python.exe scripts\diagnose_calibration_mapping.py --profile data\calibration\user_profile_mobilegaze.json
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --show-debug --no-edge-reach --no-edge-boost --vertical-offset 0
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\run_calibration.py --help
.\.venv\Scripts\python.exe scripts\diagnose_calibration_mapping.py --profile data\calibration\user_profile_mobilegaze.json
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

- Compile check passed.
- Tracking help passed.
- Calibration help passed.
- Diagnostics now flag the current polynomial profile raw-sample instability.
- `120 passed in 4.53s`.

Known issues:

- The current polynomial MobileGaze profile should not be used for cursor control.
- MobileGaze raw output remains noisy and may still need careful adaptive smoothing after a stable mapping profile exists.
- Real blink detection is still not integrated.

Next step:

- Recalibrate MobileGaze with the updated auto selector, then require clean point-mean and raw-sample diagnostics before any cursor-enabled test.

### 2026-05-18 - MobileGaze Affine Recalibration Still Unsafe

What changed:

- Recorded the user's latest MobileGaze calibration mapping diagnostics after recalibration.
- The new profile selected `affine`, but diagnostics are still not cursor-safe.
- Point-mean diagnostics improved compared with the older bad linear profile, but raw-sample instability remains high.
- Current affine profile metrics: point-mean MAE about `170.8px` X and `86.6px` Y, raw-sample MAE about `548.7px` X and `249.9px` Y, and raw-sample clipped prediction ratio `36%`.
- Confirmed cursor should remain disabled. Preview/debug is acceptable for observation only.

Files modified:

- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`
- `docs/current_status.md`

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --show-debug --no-edge-reach --no-edge-boost --vertical-offset 0
```

Known issues:

- Current affine MobileGaze profile still maps too many raw calibration samples outside the screen.
- Cursor should not be enabled until raw-sample clipped ratio and raw-sample MAE are much lower.
- Real blink detection is still not integrated.

Next step:

- Run preview-only debug to observe live mapping behavior, then improve calibration stability before any cursor-enabled run.

### 2026-05-18 - MobileGaze Preview Confirms Mapping Offscreen

What changed:

- Recorded the user's preview-only MobileGaze debug output with cursor disabled, edge reach disabled, edge boost disabled, and zero vertical offset.
- Confirmed live raw gaze is inside the calibrated raw domain, but affine mapping still predicts off-screen values before clamp.
- Example: `raw_gaze=(0.249,0.641)` mapped to `mapped_raw_before_screen_clamp=(-417.5,-88.5)` and then clamped to `(0,0)`.
- Confirmed this is a calibration mapping/model problem rather than edge reach, edge boost, smoothing, or cursor backend.
- Cursor must remain disabled.

Files modified:

- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_calibration_mapping.py --profile data\calibration\user_profile_mobilegaze.json
```

Known issues:

- The latest affine MobileGaze profile maps in-domain raw samples off-screen.
- Live tracking quality is `unsafe`.
- Real blink detection is still not integrated.

Next step:

- Implement or test a safer MobileGaze mapping strategy before any cursor-enabled run.

### 2026-05-18 - MobileGaze Bounded IDW Mapping

What changed:

- Implemented a bounded inverse-distance (`idw`) mapping model for calibration point-mean profiles.
- Wired MobileGaze `auto` mapping selection to consider `idw` before polynomial, linear, and affine.
- Added IDW serialization/deserialization and regression tests.
- Dry-ran the current MobileGaze calibration samples through auto selection. The selector chooses `idw`; raw-sample clipped prediction ratio drops to `0%`, so noisy samples no longer extrapolate off-screen before clamping.
- Raw-sample MAE is still high because the underlying MobileGaze samples are noisy, so this is a safer baseline, not a final cursor-accuracy fix.

Files modified:

- `src/visimove/calibration/mapping_model.py`
- `src/visimove/calibration/calibration_ui.py`
- `src/visimove/calibration/__init__.py`
- `src/visimove/tests/test_mapping_model.py`
- `src/visimove/tests/test_calibration_quality.py`
- `README.md`
- `docs/calibration.md`
- `docs/current_status.md`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_calibration.py --gaze-backend mobilegaze --verbose-quality
.\.venv\Scripts\python.exe scripts\diagnose_calibration_mapping.py --profile data\calibration\user_profile_mobilegaze.json
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --show-debug --no-edge-reach --no-edge-boost --vertical-offset 0
```

Known issues:

- The currently saved MobileGaze profile is still affine until recalibration is rerun.
- IDW prevents off-screen extrapolation but does not remove MobileGaze raw yaw/pitch noise.
- Cursor should stay disabled until a fresh IDW/auto profile passes diagnostics and preview.
- Real blink detection is still not integrated.

Next step:

- Recalibrate MobileGaze, confirm `mapping model type: idw` or another warning-free auto choice, then preview with cursor disabled before any cursor-enabled test.

### 2026-05-18 - MobileGaze IDW Calibration Clean Baseline

What changed:

- Recorded the user's post-recalibration diagnostics and preview.
- The saved MobileGaze profile now uses `mapping model type: idw`.
- Diagnostics are clean: no warnings, point-mean error `0px`, raw-sample clipped prediction ratio `0%`, and raw-sample negative-Y ratio `0%`.
- Raw-sample MAE remains nontrivial at about `316.7px` X and `158.1px` Y, so IDW is safer but not yet pinpoint.
- Cursor-disabled preview with edge reach disabled, edge boost disabled, and vertical offset `0` shows `calibration_quality=good`, `live_tracking_quality=stable`, and mapped output moving without corner snapping.

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --show-debug --cursor-backend win32 --no-edge-reach --no-edge-boost --vertical-offset 0
```

Known issues:

- Cursor movement may still be approximate because raw-sample MAE is still high.
- Natural blink detection is still not integrated.
- Edge reach and edge boost should stay off until baseline cursor behavior is observed.

Next step:

- Run a short controlled cursor test, press `p` once to resume, and report whether the cursor is stable, offset, or shaky.

### 2026-05-18 - MobileGaze IDW Cursor Baseline Still Shaky

What changed:

- Recorded the first cursor-enabled IDW baseline result.
- Cursor movement is active: debug shows `cursor_enabled=yes` and `cursor_skip=none`.
- Live tracking remains stable with no domain violations.
- The cursor is still shaky and does not reach screen edges or the exact intended target.
- Edge non-reach is expected because the baseline used `--no-edge-reach`, so IDW remains bounded to the calibration target rectangle.
- Logs show `smoothing_state=moving`, indicating the adaptive smoother is still tracking noisy target movement rather than locking into a steady fixation.

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --show-debug --cursor-backend win32 --edge-reach --no-edge-boost --vertical-offset 0 --smoothing-filter adaptive --adaptive-fast-alpha 0.24 --adaptive-slow-alpha 0.035 --adaptive-fixation-radius 70 --adaptive-release-radius 150 --adaptive-hold-ms 120 --max-speed-px-per-sec 650
```

Known issues:

- Cursor is real but still approximate and shaky.
- Natural blink detection is still not integrated.
- Edge boost should remain disabled until edge reach alone is validated.

Next step:

- Test edge reach plus calmer adaptive smoothing. If edge reach solves boundary access but shake remains, implement/tune stronger fixation hold. If alignment is stable but offset, implement guided bias correction.

### 2026-05-18 - MobileGaze Grid Mapper Candidate

What changed:

- Implemented a bounded `grid` mapping model.
- `grid` builds independent raw-to-screen calibration knots for columns and rows, then uses clamped interpolation.
- Wired MobileGaze `auto` mapping to try `grid` before IDW, polynomial, linear, and affine.
- Added grid serialization/deserialization and tests.
- Dry-run on the current MobileGaze samples selects `grid`; raw-sample MAE improves from IDW's about `316.7px` X / `158.1px` Y to about `262.2px` X / `120.6px` Y, still with `0%` clipping and no warnings.
- Verification passed: compile, JSON validation, tracking help, calibration help, and `124` tests.

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_calibration.py --gaze-backend mobilegaze --verbose-quality
.\.venv\Scripts\python.exe scripts\diagnose_calibration_mapping.py --profile data\calibration\user_profile_mobilegaze.json
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --show-debug --cursor-backend win32 --edge-reach --no-edge-boost --vertical-offset 0 --smoothing-filter adaptive --adaptive-fast-alpha 0.24 --adaptive-slow-alpha 0.035 --adaptive-fixation-radius 70 --adaptive-release-radius 150 --adaptive-hold-ms 120 --max-speed-px-per-sec 650
```

Known issues:

- The currently saved profile remains IDW until recalibration is rerun.
- Grid should reduce center-pull, but MobileGaze raw output is still noisy.
- Natural blink detection is still not integrated.

Next step:

- Recalibrate and confirm `grid`; then test cursor with edge reach and edge boost disabled.

### 2026-05-19 - MobileGaze Grid Edge Reach Better But Smoothing Lag

What changed:

- Recorded the user's latest MobileGaze grid/edge-reach cursor test.
- The cursor behavior is now much better and gets very close to the intended targets.
- Two issues remain: top-right corner reach is incomplete, and movement is still shaky.
- Debug indicates the mapper and edge reach can already produce the physical top-right target: `mapped_after_clamp=(2252,173)` expands to `edge_reach_after=(2559,0)` and `adjusted_after_gain=(2559,0)`.
- The final cursor output lags because `smoothed` remains behind the adjusted target while `smoothing_state=moving`.
- Next step is smoothing tuning: increase fast catch-up/max speed and strengthen fixation hold, with edge boost still disabled.

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --show-debug --cursor-backend win32 --edge-reach --no-edge-boost --vertical-offset 0 --smoothing-filter adaptive --adaptive-fast-alpha 0.34 --adaptive-slow-alpha 0.025 --adaptive-fixation-radius 90 --adaptive-release-radius 220 --adaptive-hold-ms 180 --max-speed-px-per-sec 1100
```

Known issues:

- MobileGaze raw target noise still makes the cursor shaky.
- The top-right target reaches `adjusted_after_gain`, but smoothing can prevent the cursor from landing there quickly.
- Natural blink detection is still not integrated.

Next step:

- If this tuned command improves top-right reach but shake remains, implement edge-aware catch-up and a stronger fixation lock inside the adaptive smoother.

### 2026-05-25 - Chapter 5 Implementation Document Revised

What changed:

- Reviewed `VisiMove_Chapter_5_Implementation_Plain.docx`.
- Found that the chapter structure is correct, but the content was outdated because it still emphasized EyeTrax as the current real backend and did not include the latest MobileGaze cursor progress.
- Created `docs/VisiMove_Chapter_5_Implementation_Revised.docx`.
- The revised document keeps the original Chapter 5 material and adds the latest completed work: MobileGaze ONNX backend, calibration profile loading, grid/IDW bounded mapping, edge reach, adaptive smoothing, runtime validation, diagnostics, cursor movement status, and current limitations.
- Recorded the latest user observation that cursor movement is much better, but still slow, can lose eye tracking sometimes, and shakes slightly.

Known issues:

- The product is still in development and not complete.
- Natural blink-click and dwell click are still pending.
- Cursor movement still needs smoothing, tracking-loss handling, and fixation stability improvements.

Next step:

- Use the revised Chapter 5 document as the updated implementation chapter draft, then add final screenshots once the frontend and desktop screens are finalized.

### 2026-05-25 - MobileGaze Smooth Cursor Tuning

What changed:

- Tuned the live cursor path for smoother MobileGaze control and full-screen reach.
- Enabled edge reach by default so calibrated target-area output expands to the full screen without requiring long command-line flags.
- Kept edge boost disabled because it can over-amplify noisy near-edge gaze.
- Set PyAutoGUI cursor movement duration to `0` and increased the cursor safety speed cap to `3000px/s` to reduce visible lag.
- Increased smoothing `max_jump_pixels` to `420` so legitimate screen-wide movement is not artificially slow.
- Retuned adaptive smoothing defaults: fast alpha `0.38`, slow alpha `0.025`, fixation radius `90`, release radius `220`, and hold `180ms`.
- Updated adaptive smoothing so it can hold jitter around the fixation anchor, not only tiny changes around the previous smoothed cursor point.
- Updated the real-time pipeline so the smoother holds the last good position when face/eyes are lost, gaze confidence is low, or live gaze quality is blocked as unstable/unsafe.

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q src\visimove\tests\test_smoothing.py src\visimove\tests\test_axis_adjustment.py src\visimove\tests\test_run_tracking_startup.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

- Compile check passed.
- Focused tests passed: `36 passed`.
- Full tests passed: `126 passed`.
- Tracking CLI help passed.

Commands to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --show-debug --cursor-backend win32 --no-edge-boost --vertical-offset 0
```

Known issues:

- Cursor accuracy still depends on MobileGaze raw signal quality and calibration posture.
- Real blink detection is still not integrated.

Next step:

- Run a short cursor test and watch `smoothing_state`, `cursor_skip`, and whether cursor reaches all screen edges without shaking.

### 2026-05-25 - Frontend Review and 30 Percent Implementation Scope

What changed:

- Inspected the separate frontend project at `C:\Users\MNA\Desktop\FYP\visimove`.
- Confirmed the frontend is complete for the current report scope and is a Next.js 16, React 19, TypeScript, Tailwind CSS 4, Three.js/React Three Fiber/Drei, GSAP, Lenis, and Lucide React implementation.
- Confirmed frontend coverage includes a responsive landing page, navigation, hero section, about section, how-it-works flow, feature cards, technology/trust section, download section, smooth scrolling, scroll choreography, and a 3D VisiMove eye model from `public/models/visimove-eye.glb`.
- Created `docs/VisiMove_Chapter_5_Implementation_30_Percent.docx`.
- The new document scopes Chapter 5 to 30% total implementation: 20% frontend completion and 10% backend prototype progress only.
- Backend content in the new document is intentionally limited to implemented MobileGaze ONNX integration, calibration/mapping, diagnostics, cursor smoothing/safety, and current limitations.

Known issues:

- The frontend was inspected statically; a production build was not run because the frontend project is outside the backend writable workspace and the README notes that `next/font` production builds may require network access.
- The backend remains in development and is not complete.

Next step:

- Use the 30% scoped document for the implementation chapter when the report should not claim full backend completion.

### 2026-05-25 - MobileGaze Demo-Stable Cursor Tuning Reverted

What changed:

- Reverted the median target stabilization experiment because the user reported it worsened the previously better cursor behavior.
- Removed `adaptive_target_median_window` from the smoother and CLI.
- Restored the prior default cursor/smoothing values: cursor speed cap `3000`, smoothing max jump `420`, adaptive fast alpha `0.38`, slow alpha `0.025`, fixation radius `90`, release radius `220`, and hold `180ms`.
- Kept the earlier working behavior where edge reach is enabled and edge boost is disabled by default.

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp src\visimove\tests\test_smoothing.py src\visimove\tests\test_run_tracking_startup.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
```

Result:

- Reverification after the revert is required.

Command to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --show-debug --cursor-backend win32 --no-edge-boost --vertical-offset 0
```

Known issues:

- Cursor is still shaky/jumpy and not landing exactly at the intended target.
- The rejected median-window approach should not be used for the presentation.

Next step:

- Return to the previously better command and avoid extra smoothing experiments until a new mapping/calibration-side fix is planned.

### 2026-05-25 - MobileGaze Unstable Gate Relaxed

What changed:

- Reviewed the user's latest MobileGaze debug output where the cursor stopped unexpectedly.
- Found that the live-quality monitor reports `unstable` after even a small number of domain/clipping events, for example `live_domain_violation_ratio=0.03`.
- The real-time pipeline was blocking cursor movement and feeding `None` into the smoother for both `unstable` and `unsafe`.
- Changed the runtime gate so only `unsafe` blocks cursor movement by default. `unstable` is still shown in debug but no longer freezes the cursor.
- Added a regression test confirming cursor movement is allowed when live quality is only `unstable`.

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp src\visimove\tests\test_realtime_live_safety.py src\visimove\tests\test_live_quality.py src\visimove\tests\test_smoothing.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

- Compile check passed.
- Focused safety/smoothing tests passed: `18 passed`.
- Full tests passed: `127 passed`.
- Tracking CLI help passed.

Command to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --show-debug --cursor-backend win32 --no-edge-boost --vertical-offset 0
```

Known issues:

- This fix addresses cursor freezing/stopping due to brief `unstable` live quality.
- It does not fully solve target accuracy or MobileGaze raw target noise.

Next step:

- Run a short cursor test and check whether `cursor_skip=live gaze outside calibrated domain` disappears unless `live_tracking_quality=unsafe`.

### 2026-05-25 - Adaptive Large-Jump Confirmation

What changed:

- Reviewed the user's latest report that the cursor no longer sticks, but still feels shaky/jumpy and does not land smoothly on the intended target.
- Kept the previous fix that allows `unstable` live quality to continue moving; only `unsafe` still blocks by default.
- Added large-jump confirmation to `AdaptiveSmoothingFilter`.
- A sudden large target change is now held as `smoothing_state=confirming` for one sample. If the next target remains near that new location, the smoother accepts it and moves normally.
- This avoids the rejected median-window approach and targets the current failure mode: isolated MobileGaze target spikes.
- Added regression tests for confirmed large movement and isolated spike rejection.

Files modified:

- `src/visimove/smoothing/adaptive_filter.py`
- `src/visimove/pipeline/realtime_pipeline.py`
- `src/visimove/tests/test_smoothing.py`
- `config/default.yaml`
- `docs/current_status.md`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp src\visimove\tests\test_smoothing.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp src\visimove\tests\test_realtime_live_safety.py src\visimove\tests\test_live_quality.py src\visimove\tests\test_run_tracking_startup.py src\visimove\tests\test_smoothing.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

- Compile check passed.
- Smoothing tests passed: `14 passed`.
- Focused safety/startup/smoothing tests passed: `37 passed`.
- Full test suite passed: `129 passed`.
- Tracking CLI help passed.

Command to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --show-debug --cursor-backend win32 --no-edge-boost --vertical-offset 0
```

Known issues:

- This should reduce one-frame jumps, but it cannot make noisy MobileGaze raw output perfectly pinpoint.
- Brief `smoothing_state=confirming` is expected when the target jumps suddenly.
- Real blink detection is still not integrated.

Next step:

- Run a short cursor test and report whether jumpiness is reduced, whether `smoothing_state=confirming` appears briefly, and whether the cursor still fails to reach specific screen regions.

### 2026-05-25 - Adaptive Edge Catch-Up

What changed:

- Reviewed the user's latest debug output after large-jump confirmation.
- Found that the mapper/edge-reach path can produce true corner targets such as `adjusted_after_gain=(0,0)`, but the final `smoothed` cursor can remain short of the corner, for example around `(135,99)`.
- Added an edge-aware catch-up path to `AdaptiveSmoothingFilter`.
- The smoother now recognizes physical screen-edge targets when the real-time pipeline supplies screen dimensions.
- Confirmed edge targets still use large-jump confirmation first, then move with a faster edge alpha and snap the final few pixels to the edge.
- Added tests for confirmed edge catch-up and near-edge snapping.

Files modified:

- `src/visimove/smoothing/adaptive_filter.py`
- `src/visimove/pipeline/realtime_pipeline.py`
- `src/visimove/tests/test_smoothing.py`
- `config/default.yaml`
- `docs/current_status.md`
- `docs/test_suite.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/PROJECT_STATE.json`

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp src\visimove\tests\test_smoothing.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp src\visimove\tests\test_realtime_live_safety.py src\visimove\tests\test_live_quality.py src\visimove\tests\test_run_tracking_startup.py src\visimove\tests\test_smoothing.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
```

Result:

- Compile check passed.
- Smoothing tests passed: `16 passed`.
- Focused safety/startup/smoothing tests passed: `39 passed`.
- Full test suite passed: `131 passed`.
- Tracking CLI help passed.

Command to run:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --show-debug --cursor-backend win32 --no-edge-boost --vertical-offset 0
```

Known issues:

- This should help top-left/top-right/bottom edges when `adjusted_after_gain` already reaches the edge.
- If the cursor is consistently offset beside/above/below non-edge targets, that is likely residual calibration bias or MobileGaze raw signal noise, not edge smoothing.
- Real blink detection is still not integrated.

Next step:

- Re-test the top-left corner and watch for `smoothing_state=edge` or `edge_hold`. If non-edge targets remain consistently offset, collect a short guided bias sample rather than adding hardcoded offsets.

### 2026-05-25 - Chapter 5 Backend Scope Inserted

What changed:

- Edited the user-provided Chapter 5 Word document from `C:\Users\MNA\Desktop\VisiMove_Chapter_5_Implementation.docx`.
- Created `C:\Users\MNA\Desktop\VisiMove_Chapter_5_Implementation_Backend_Updated.docx`.
- Preserved the existing frontend text and screenshots.
- Added backend implementation content only for the current 10% backend scope: Windows/Python development environment, OpenCV/webcam capture, face and eye detection, detector fallback design, and a backend screenshot note.
- Avoided adding later backend work such as gaze cursor control, calibration, blink clicking, smoothing, or full automation because the requested report scope is 30% total implementation: 20% frontend and 10% backend up to face/eye detection.

Known issues:

- The generated document was edited at the Word XML level because `python-docx` is not installed in the current virtual environment.
- The original Desktop document was left unchanged; the updated version is saved as a new file.

### 2026-05-25 - Chapter 5 Backend Scope Reformatted With Screenshot

What changed:

- Updated `VisiMove_Chapter_5_Implementation_Backend_Updated.docx` in the repo root.
- Preserved the existing frontend paragraphs.
- Reformatted backend content under the user's required headings:
  - `5.1. Development Environment`
  - `5.2. Technology Stack`
  - `5.3. Module Description`
  - `5.4. User Interface Screenshots`
- Limited backend description to the agreed 10% implementation scope: Python desktop prototype, webcam capture, OpenCV/MediaPipe-style detection, face detection, eye detection, and detection status output.
- Removed duplicate earlier backend paragraphs from the document.
- Added a generated backend visual figure at `docs/backend_face_eye_detection_preview.png` and embedded it into the Word document.
- Created `VisiMove_Chapter_5_Implementation_Backend_Updated.backup.docx` before editing.

Known issues:

- The backend figure is a report-ready visual representation of the face/eye detection preview stage, not a live camera capture from the user's webcam.
- Later backend work such as gaze estimation, calibration, cursor control, blink clicking, smoothing, and automation is intentionally excluded from this chapter scope.

### 2026-05-25 - Chapter 5 Updated Plain Scope Correction

What changed:

- Edited `VisiMove_Chapter_5_Implementation_Updated_Plain.docx` after the user reformatted it.
- Preserved the existing document structure and frontend content.
- Corrected backend wording so completed backend work is limited to webcam capture, face detection, eye detection, and detection-status preview.
- Kept gaze backend integration, calibration, runtime validation, smoothing, blink logic, cursor control, dwell click, diagnostics, and executable packaging in the document as planned/future backend stages.
- Updated backend environment, technology stack, module description, implementation status, processing pipeline, screenshots text, and summary language for consistency.
- Created `VisiMove_Chapter_5_Implementation_Updated_Plain.backup.docx` before editing.

Known issues:

- The Word document was edited directly at the DOCX XML level because `python-docx` is not installed in the current virtual environment.
- The report scope is intentionally conservative: 20% frontend plus 10% backend up to face/eye detection.

### 2026-05-26 - Chapter 5 Backend Expansion Suggestions

What changed:

- Inspected `C:\Users\MNA\Desktop\VisiMove_Chapter_5_Implementation_Backend_Updated.docx` without modifying it.
- Confirmed the backend section is currently scoped to webcam capture, face detection, eye detection, and detection-status preview, with later gaze/cursor features marked planned.
- Recommended safe additions that can make the backend section stronger without overstating progress:
  - a detailed backend workflow subsection for camera frame acquisition to detection preview,
  - a face/eye detection method subsection,
  - fallback/error handling details,
  - implementation status evidence,
  - testing/validation notes,
  - limitations and planned next stage,
  - screenshots of backend preview, face detection, eye-region detection, terminal startup/debug output, and project/module structure.

Known issues:

- The document itself was not modified in this step.

### 2026-05-26 - Expanded Backend Documentation Scope Agreed

What changed:

- User clarified that the documentation should include the backend work already completed/prototyped beyond face and eye detection.
- The backend documentation scope should now cover:
  - Python backend project structure,
  - OpenCV webcam capture,
  - `scripts/run_tracking.py` real-time pipeline,
  - startup summary and debug logging,
  - face detection and eye/face landmark detection,
  - detector/backend fallback behavior,
  - performance timing output,
  - EyeTrax, GazeFollower, and MobileGaze backend candidate integration/testing,
  - MobileGaze ONNX model wiring and adapter implementation,
  - MobileGaze raw gaze output verification and horizontal-axis fix,
  - native gaze debug output,
  - calibration script and JSON calibration profiles,
  - backend-aware calibration profiles,
  - calibration quality reports,
  - mapping diagnostics,
  - affine, linear, polynomial, IDW, and grid mapping approaches,
  - poor-calibration safety checks,
  - raw-domain validation.

Known issues:

- Cursor accuracy, cursor smoothing, blink click, dwell click, and final packaging should still not be presented as completed final features.

### 2026-05-26 - Chapter 5 Expanded Frontend and Backend Implementation

What changed:

- Updated `C:\Users\MNA\Desktop\VisiMove_Chapter_5_Implementation_Backend_Updated.docx`.
- Expanded the backend implementation content in the existing file to include the agreed completed/prototyped backend scope:
  - Python backend structure,
  - OpenCV webcam capture,
  - `scripts/run_tracking.py`,
  - startup/debug logging,
  - face/eye detection,
  - fallback behavior,
  - EyeTrax, GazeFollower, and MobileGaze backend candidate testing,
  - MobileGaze ONNX model and adapter,
  - calibration profile generation,
  - mapping diagnostics,
  - affine, linear, polynomial, IDW, and grid mapping approaches,
  - raw-domain safety validation.
- Added backend testing and validation content to Chapter 5 without adding individual test-file names there.
- Expanded the frontend content using the existing documented implementation:
  - Next.js/React/TypeScript app structure,
  - root layout and global styling,
  - navbar/footer,
  - hero/about/how-it-works/features/technology/download sections,
  - Three.js/React Three Fiber/Drei 3D eye model,
  - GSAP/Lenis motion,
  - Lucide icons,
  - reusable UI components,
  - frontend implementation status.
- Added screenshot guidance for frontend and backend evidence.
- Created backups on the Desktop:
  - `VisiMove_Chapter_5_Implementation_Backend_Updated.pre_expanded_backend_backup.docx`
  - `VisiMove_Chapter_5_Implementation_Backend_Updated.pre_frontend_expansion_backup.docx`

Verification:

- DOCX zip/XML validation passed after edits.

Known issues:

- The document was edited at the DOCX XML level because `python-docx` is not installed in the current virtual environment.
- Test file names are reserved for the next/testing chapter rather than Chapter 5.

### 2026-05-26 - Backend Screenshot Capture Plan

What changed:

- Prepared a one-by-one screenshot plan for Chapter 5 backend evidence.
- Required backend screenshots:
  1. `run_tracking.py` startup summary,
  2. webcam face/eye detection preview,
  3. MobileGaze debug output with `raw_gaze` and `native_gaze`,
  4. calibration collection screen,
  5. `diagnose_calibration_mapping.py` output,
  6. backend project folder/module structure.
- Recommended cropping each screenshot to show the meaningful evidence only, avoiding excessive terminal history.

Known issues:

- Screenshots still need to be captured manually from the user's machine.

### 2026-05-30 - Chapter 6 Quality Tables Alignment

What changed:

- Reviewed the user's Chapter 6 quality-testing tables 6.30 to 6.37.
- Identified wording that must be updated to match the current documented implementation scope:
  - frontend is implemented/running,
  - backend prototype is implemented for webcam, detection, MobileGaze adapter, calibration/mapping diagnostics, and safety validation,
  - cursor movement is prototype/partial only and not final,
  - blink/dwell and packaging remain pending,
  - FPS testing can be reported as performed at prototype level using backend FPS output rather than pending.

Known issues:

- The actual Chapter 6 document/table file has not been edited yet.

### 2026-05-31 - Detailed Project Implementation Document Created

What changed:

- Created `docs/VisiMove_Detailed_Project_Implementation.docx` as a separate submission document, not Chapter 5.
- Document includes:
  - complete frontend implementation details,
  - official backend scope only: Python backend structure, OpenCV webcam capture, real-time tracking pipeline, startup/debug logging, face/eye detection, fallback behavior, EyeTrax/GazeFollower/MobileGaze testing, MobileGaze ONNX model and adapter, calibration profiles, calibration quality reports, mapping diagnostics, affine/linear/polynomial/IDW/grid mapping approaches, raw-domain safety validation, and performance timing output.
- Document explicitly excludes final production claims for stable cursor control, blink click, dwell click, and executable packaging.
- Formatting applied through DOCX XML:
  - headings are bold and 14pt,
  - body text is 12pt,
  - body text is justified,
  - no heading colors were added.

Verification:

- DOCX zip/XML validation passed.

Known issues:

- `python-docx` is not installed, so the file was generated directly as a DOCX XML package.

### 2026-05-31 - Detailed Implementation Document Stack Added

What changed:

- Updated `C:\Users\MNA\Desktop\FYP-backend\VisiMove_Detailed_Project_Implementation.docx`.
- Added a dedicated `2.1 Technology Stack` section.
- Added `2.1.1 Frontend Technology Stack` covering TypeScript, JavaScript, Next.js, React, Tailwind CSS, Three.js, React Three Fiber, Drei, GSAP, Lenis, Lucide React, and Vercel.
- Added `2.1.2 Backend Technology Stack` covering Python, OpenCV, MediaPipe-style FaceLandmarker support, ONNX Runtime, MobileGaze, EyeTrax, GazeFollower, NumPy, YAML, JSON, pytest, and the mapping approaches.
- Preserved the existing document formatting approach: headings remain bold 14pt, body text remains 12pt and justified.
- Created backup `C:\Users\MNA\Desktop\FYP-backend\VisiMove_Detailed_Project_Implementation.pre_stack_backup.docx`.

Verification:

- DOCX zip/XML validation passed after adding the stack section.

## 2026-09-03 Code Health Recheck

- Rechecked the VisiMove backend project after the user's request to inspect new skills, inconsistencies, and redundancies.
- Checked current OpenAI skill/plugin guidance from official sources; no extra plugin or skill installation was needed for this local code-health pass.
- Found and fixed a stale compatibility module at `src/visimove/gaze/adapters.py`. It previously contained placeholder adapter shells that raised `NotImplementedError`, while the real EyeTrax, GazeFollower, and MobileGaze adapters live in backend-specific modules.
- `visimove.gaze.adapters` now re-exports the real adapter classes and config/status dataclasses, so older imports cannot accidentally route to obsolete placeholder behavior.
- Added a regression test proving the legacy `visimove.gaze.adapters` module points to the same real adapter classes exported by `visimove.gaze`.
- Verification passed: `python -m compileall -q src scripts` and `python -m pytest -q` completed with `132 passed`.

## 2026-09-03 Demo Cursor Tuning Profile

What changed:

- Planned and implemented the next cursor-stabilization step as an opt-in guided demo tuning profile instead of changing the current working cursor defaults.
- Added `src/visimove/calibration/demo_tuning.py` with JSON save/load support for demo tuning profiles.
- Added `scripts/tune_cursor_demo.py` to collect guided live samples with cursor movement disabled. It measures observed adjusted cursor targets against requested screen targets and saves a tuning profile.
- Added `--tuning-profile` to `scripts/run_tracking.py`. A generated profile can now apply measured calibration offsets, adaptive smoothing overrides, and cursor movement settings for a repeatable demo run.
- Explicit CLI options still override values loaded from the tuning profile, so manual test commands remain in control.
- Added tests for profile generation, JSON round-trip, config application, and run-tracking load order.

Commands:

```powershell
.\.venv\Scripts\python.exe scripts\tune_cursor_demo.py --gaze-backend mobilegaze --seconds-per-target 1.6
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --show-debug --tuning-profile data\calibration\demo_tuning_mobilegaze.json --no-edge-boost
```

Verification performed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q src\visimove\tests\test_demo_tuning.py src\visimove\tests\test_run_tracking_startup.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\run_tracking.py --help
.\.venv\Scripts\python.exe scripts\tune_cursor_demo.py --help
```

Result:

- Focused tests passed: `20 passed`.
- Full test suite passed: `135 passed`.
- CLI help checks passed for both tracking and tuning scripts.

Known issues:

- The tuning profile improves repeatability and measured center bias, but it still depends on live camera posture and MobileGaze signal stability.
- Real blink detection remains not integrated.
- Cursor accuracy still needs live validation after generating a real tuning profile on the user's machine.
## 2026-09-05 Demo Tuning Safety And Edge Preservation

- Live validation of the first generated demo tuning profile exposed an unsafe capture: center samples had 76% edge saturation and 973px robust jitter.
- The unsafe profile had learned horizontal offset -44px and vertical offset +180px; applying these after edge reach pulled the exact top-right target (2559,0) inward to (2515,180).
- Demo tuning profiles now receive a quality result. A center capture requires at least 8 samples, no more than 35% edge saturation, and robust jitter no greater than 22% of the smaller screen dimension (minimum 220px).
- Unsafe captures retain zero new offsets, are saved only for diagnosis, and cannot be loaded by scripts/run_tracking.py.
- Existing version-1 profiles without stored quality metadata are evaluated from their samples when loaded, so the current unsafe profile is blocked automatically.
- Axis adjustment now preserves exact physical screen-edge coordinates when edge reach is active, even if center-bias offsets are configured.
- Verification: Python compilation passed, focused tuning/axis/startup tests passed (29 passed), and the complete test suite passed (137 passed).
- The current data/calibration/demo_tuning_mobilegaze.json is intentionally rejected because its center capture is unsafe. It must be regenerated from a stable center capture before use.
## 2026-09-25 Ten-Day Completion Track

- The project has entered a maximum ten-day delivery window.
- Scope is frozen around the existing Next.js frontend and the MobileGaze Windows backend.
- No further EyeTrax, GazeFollower, mapping-model, or broad smoothing experiments are allowed unless the selected MobileGaze path becomes unusable.
- Delivery-critical backend work is: obtain one repeatable good calibration/tuning profile, validate cursor coverage and stability, implement one reliable click path, package a runnable Windows build, and execute/document acceptance tests.
- Blink click is desirable, but dwell click is the fallback required for project completion if reliable natural blink detection cannot be completed quickly.
- Each remaining change must have a focused test and must preserve the existing full test suite.
- Days 1-2: establish and freeze the cursor baseline using calibration, diagnostics, and the validated demo tuner.
- Days 3-4: complete clicking, prioritizing dwell and then natural blink if feasible.
- Days 5-6: create the runnable application entry point and Windows executable/package.
- Days 7-8: run functional, safety, performance, and repeatability tests; fix only release-blocking defects.
- Day 9: capture final screenshots, logs, test evidence, and presentation demonstration flow.
- Day 10: contingency, final clean-machine rehearsal, and submission freeze.
## 2026-09-25 Guided Tuning Removed From Release Path

- A second guided MobileGaze tuning attempt was rejected with center jitter of 1249.4px.
- Inspection found 47 center samples spanning the full screen: X ranged from 0 to 2559 and Y ranged from 0 to 1439, with 10 center samples pinned to an edge.
- This is continuous live target instability rather than a short transition immediately after pressing Enter.
- The saved MobileGaze grid calibration remains the selected release calibration and reports good calibration quality; the optional guided offset tuner is not trustworthy enough to remain on the ten-day critical path.
- The release candidate therefore uses the existing grid calibration, edge reach enabled, edge boost disabled, and zero horizontal/vertical offsets, without a demo tuning profile.
- Further tuning-profile work is frozen. The next delivery task is to validate the baseline briefly and proceed to click implementation.
## 2026-09-25 Cursor Fail-Safe And Edge Release Fix

- Live zero-offset baseline testing reached exact screen corner coordinates and triggered PyAutoGUI FailSafeException, terminating tracking.
- Root cause: edge reach used a zero-pixel margin and the adaptive smoother used a fast 0.65 edge catch-up rate, allowing noisy saturated gaze to move rapidly into PyAutoGUI emergency corners.
- PyAutoGUI fail-safe remains enabled. The controller now catches its exception, pauses cursor control, prints recovery guidance, and keeps the tracking application alive.
- Default edge reach margin is now 8px. This keeps close buttons and the taskbar reachable while avoiding exact emergency corner pixels.
- Adaptive edge catch-up alpha is now 0.45 instead of 0.65. Normal center movement settings and the 3000px/s cursor speed cap were not changed.
- Added a PyAutoGUI fail-safe regression test and a release-default configuration regression test.
- Verification: Python compilation passed, focused cursor/mapping/smoothing tests passed (44 passed), and the complete suite passed (139 passed).
- Live accuracy and perceived shake still require one short validation run; this change specifically fixes the crash and reduces edge aggressiveness.

## Update: Bounded Velocity Cursor And Dwell Integration (2026-09-26)

Latest live evidence showed that the calibrated MobileGaze target itself can jump across large screen distances while live_tracking_quality=stable. Multiple absolute smoothing adjustments did not remove the underlying target instability. The release path therefore now defaults to bounded velocity control rather than direct absolute placement.

VelocityCursorFilter converts mapped gaze displacement from screen center into incremental X/Y velocity. Defaults are a normalized 0.20 center deadzone, 900 px/s maximum speed, nonlinear response exponent 1.6, 0.10 s maximum integration interval, and 8 px screen margin. A single noisy frame can create only a bounded step. --cursor-mode absolute preserves the previous behavior for comparison.

The active realtime pipeline now uses DwellSelector. In velocity mode, dwell accumulates only during velocity_hold, fires one left-click after 1200 ms, and cannot repeat until the cursor leaves the dwell anchor. Tracking loss, safety blocking, or pause resets dwell. Resume re-synchronizes velocity state to the physical cursor.

Live validation remains required. Correct operation is: press p, steer by looking in the desired direction, return gaze near screen center to stop, then hold for the dwell click.

## Update: Velocity Debug Crash Fix (2026-09-26)

The first live velocity-mode run reached full MobileGaze, calibration, cursor, and dwell initialization, then crashed only when debug output was printed. RealtimeVisiMovePipeline._print_debug still contained one obsolete assignment that referenced click_event after the method signature had been changed to accept click_name. The stale assignment was removed. A regression test now executes _print_debug with click_name=dwell_left_click and verifies both the click name and cursor_mode=velocity output. The full automated suite passes with 152 tests.

## Update: Neutral-Calibrated Velocity Stabilization (2026-09-26)

Live velocity logs showed continuous `velocity_moving` even near the user's intended resting gaze. The root cause was that velocity direction was measured from the geometric screen center, while the user's session-specific mapped neutral gaze was substantially offset. Edge-reach expansion was also still being passed into relative control even though it is only useful for absolute placement.

Velocity mode now learns a robust neutral gaze for one second whenever cursor control is resumed. The neutral X/Y values are medians, and a five-sample rolling median rejects isolated live gaze spikes before velocity is calculated. Direction is normalized independently from the learned neutral toward each screen boundary, so an off-center neutral does not reduce directional range. Velocity mode now consumes the bounded calibrated screen point directly and bypasses edge reach, edge boost, gains, and offsets; those adjustments remain available to absolute mode.

While the cursor is paused, velocity calibration and integration do not run. Pressing `p` resumes and starts neutral acquisition; the user must look at the screen center for one second. Pressing `c` repeats neutral acquisition without restarting tracking. Dwell remains disabled during calibration and movement and is armed only in `velocity_hold`.

This design follows established gaze-pointer practice: robust central offset estimation and outlier rejection address systematic bias and spikes, while bounded relative motion prevents a noisy absolute sample from teleporting the Windows cursor. Verification completed with Python compilation and 154 passing tests. Live webcam validation of perceived smoothness and dwell remains required.

## Update: Velocity Vertical Plateau Fix (2026-09-26)

The first neutral-calibrated live test left the cursor stuck near the bottom. The velocity filter itself supported negative/upward movement, but its input still came from the calibrated grid mapper. Grid mapping is appropriate for absolute placement, but its bounded row interpolation can flatten a broad range of lower gaze samples to the same bottom calibration row, removing the continuous vertical gradient required by relative steering.

Velocity mode now derives direction directly from MobileGaze's continuous normalized `raw_x/raw_y` output, scaled into screen-sized direction space before neutral calibration and median rejection. Calibration mapping, edge reach, gains, and offsets remain available for absolute mode and continue to run for diagnostics and live-domain safety monitoring, but they no longer distort relative velocity direction. Verification completed with compilation and 155 passing tests. Live upward/downward steering validation remains required.

## Update: Asymmetric Velocity Sensitivity (2026-09-28)

Live continuous-raw velocity testing was smoother, but the cursor still moved predominantly downward and was slow. Debug showed a session neutral near `raw_y=0.63`; the shared normalized dead zone of `0.20` required upward gaze below roughly `raw_y=0.50` before any upward velocity could begin, while downward samples around `raw_y=0.77` activated more readily.

Velocity control now uses axis-specific dead zones: `0.10` horizontal and `0.06` vertical. The bounded maximum velocity is increased from 900 to 1400 px/s and the response exponent is reduced from 1.6 to 1.35 for earlier, faster response. The one-second neutral acquisition, five-sample median spike rejection, 0.10-second integration cap, and 8px edge margin remain unchanged. Verification completed with compilation and 156 passing tests. Live bidirectional movement and perceived speed still require validation.

## Update: Velocity Experiment Rejected; Absolute Release Mode Restored (2026-09-28)

Live testing confirmed that neutral-calibrated velocity control produced smoother motion but did not follow the looked-at screen location. Debug showed `raw_x` increasing from 0.442 to 0.548 while the cursor remained near the left edge because both samples were interpreted relative to a learned neutral. This is correct directional-joystick behavior but incompatible with VisiMove's requirement that gaze correspond to an intended screen target. Recentring cannot solve that semantic mismatch.

Calibrated absolute control is restored as the release default. It uses the selected MobileGaze grid mapping, safe edge reach, zero edge boost, adaptive stabilization, invalid-target hold, 8px edge safety margin, and the existing cursor speed limiter. Velocity mode remains available only as an explicit experimental option and is no longer recommended for the release demonstration. Absolute mode does not use or require the `c` recenter action. Verification completed with compilation, 54 focused tests, and all 156 tests passing.
## Update: One Euro Absolute Gaze Stabilization (2026-09-28)

Latest absolute-mode logs showed that calibrated and edge-reached targets were generally plausible, but the adaptive smoother repeatedly remained in `confirming`. Consecutive MobileGaze targets often differed by more than the confirmation radius, so the cursor lagged behind valid gaze changes and then caught up unevenly. Velocity mode had already been rejected because smooth relative steering does not place the cursor at the looked-at screen location.

A two-dimensional One Euro filter is now implemented for absolute gaze coordinates and is the release smoothing default. It smooths low-speed fixation noise more strongly and reduces smoothing during faster intentional movement, avoiding the adaptive filter's binary jump-confirmation gate. Missing gaze holds the last filtered position. The original adaptive, EMA, deadzone, fixation, and Kalman filters remain available for diagnostics, while velocity mode remains optional and experimental.

Release defaults are `one_euro_min_cutoff=0.35`, `one_euro_beta=0.0001`, `one_euro_derivative_cutoff=1.0`, and `one_euro_max_dt_seconds=0.20`. Runtime debug should now show `smoothing_state=one_euro_tracking` in absolute mode. Compilation and the complete automated suite pass with 160 tests. Live webcam validation is still required; smoothing can reduce jitter and lag but cannot remove systematic calibration bias or guarantee pixel-exact gaze placement.
## Update: Bounded Absolute Motion After Live One Euro Test (2026-09-28)

The first live One Euro test was smoother at low speed but made large movement more drastic. Live evidence showed the mapped target repeatedly saturating from one physical edge to another, for example from `(2551,1431)` to `(8,1431)`. The original One Euro `beta=0.0004` interpreted these screen-space spikes as intentional high-speed movement and reduced filtering, while the cursor safety limit still allowed `3000px/s`.

The absolute One Euro path now includes a motion planner with a `1600px/s` maximum output speed and `5000px/s^2` maximum acceleration. It uses stopping-distance-aware braking, prevents immediate high-speed direction reversals, and still converges to a stationary target. One Euro high-speed sensitivity is reduced to `beta=0.0001`. Missing gaze clears motion velocity, and resuming cursor control synchronizes the smoother with the physical cursor position so stale paused motion cannot be released suddenly. Debug can show `smoothing_state=one_euro_bounded` while speed or acceleration limiting is active.

Focused smoothing/startup verification passes with 43 tests. Compilation and the complete automated suite pass with 164 tests. Live webcam validation remains required. The change controls motion severity; it does not correct systematic calibration error or the grid mapper's edge saturation.

## Update: Settled Five-Point Accuracy Tuning (2026-09-29)

Live bounded One Euro testing confirms that motion is much smoother, but the final cursor can remain hundreds of pixels from the intended point. Runtime debug exposes the mapped and filtered coordinates but does not contain the user's ground-truth target, so a reliable spatial correction cannot be inferred from debug logs or a guessed global offset.

The guided demo tuner now measures the same filtered One Euro output used during normal absolute cursor tracking. Each target capture resets the smoother, allows a configurable stabilization interval, then records settled samples. Target summaries use medians rather than means. Five-point captures fit conservative per-axis gain and offset corrections from target-labelled medians; accepted gains are limited to 0.70 through 1.30 and offsets to the configured maximum. Unsafe or insufficient captures retain the existing quality rejection path and are not applied.

Generated tuning profiles preserve the active smoothing filter instead of forcing the older adaptive filter. The suggested launch command applies the profile once and no longer repeats hard-coded edge or vertical-offset overrides. Verification completed with focused tests, syntax/CLI checks, compilation, and all 165 automated tests passing. A new live five-point capture is required before exactness improvement can be evaluated.
## Update: Responsive Full-Screen Accuracy Tuner (2026-09-29)

The first live accuracy-tuner attempt made the OpenCV window appear stalled because the script called console input() between target captures. While input() blocked the main thread, cv2.waitKey() could not service the Windows message queue, so Windows marked the preview unresponsive.

Preview mode no longer uses console input. The tuner now runs a continuous OpenCV event loop, displays each requested target as a full-screen dot at its real screen coordinate, starts capture with Space or Enter, and cancels with Escape or Q. Stabilization and collection progress, face/eye availability, and sample count are displayed in the same window. Console input remains only for the explicit --no-preview mode. The full automated suite passes with 167 tests; the new Windows preview flow still requires one live user run.
## Update: Unsafe Five-Point Profile Rejected (2026-09-29)

The first responsive five-point capture produced a profile marked good with horizontal_offset=180 and vertical_offset=-180, both exactly at the configured safety limit. Per-target inspection showed the capture was not spatially coherent: center median was approximately (1083,1431), top-left (28,1411), top-right (2160,1142), bottom-left (994,1409), and bottom-right (871,1431). The bottom row was horizontally crossed, the left column had effectively no vertical separation, several targets had 90th-percentile jitter above 316.8px, and center was saturated at the bottom edge. No global gain/offset can repair this capture.

Multi-target validation now checks minimum samples, center placement, row/column directional separation, per-target jitter, and whether a bounded gain/offset fit succeeds. Failed multi-target fits and center-only corrections that reach the offset safety cap are marked needs_review and retain baseline gain/offset values. The existing bad tuning profile was quarantined as needs_review and its offsets reset to zero; run_tracking now rejects it before camera startup.

The tuning display is distraction-free during stabilization and collection, showing only the target dot, and defaults now allow 1.0 second stabilization plus 2.0 seconds collection. Verification completed with profile rejection, compilation, and all 168 automated tests passing. Baseline tracking without the tuning profile remains the usable path until a directionally valid capture is obtained.
## Update: MobileGaze Raw-Domain Temporal Stabilization (2026-09-29)

The latest live absolute-mode trace confirmed that bounded One Euro smoothing controls cursor speed, but the MobileGaze raw estimate still changes substantially during intended fixation. The nonlinear grid mapper magnifies those raw changes and repeatedly saturates the mapped X coordinate at the right calibration boundary, so screen-space smoothing alone cannot provide a stationary target.

MobileGaze now applies a configurable five-sample temporal median to normalized raw X/Y coordinates before calibration mapping. This rejects isolated inference spikes at their source while adding only a short majority-window delay. The unfiltered normalized sample remains available as unfiltered_raw_gaze in debug output beside the filtered raw_gaze, and native yaw/pitch remains unchanged. Filter history is cleared whenever face detection, face cropping, or inference is unavailable so stale gaze cannot affect reacquisition.

The release configuration uses mobilegaze.temporal_median_window=5. Focused adapter tests pass with 20 tests, Python compilation passes, and the complete automated suite passes with 170 tests. Live webcam validation is still required. This change targets fixation drift caused by transient raw estimates; systematic point-placement error still depends on calibration quality and cannot be claimed fixed without target-labelled live measurements.
## Update: MobileGaze Calibration Preprocessing Compatibility Guard (2026-09-29)

Live validation of the raw median filter used a calibration profile created on 2026-05-18 before raw-domain temporal filtering existed. The trace showed filtered raw Y values repeatedly mapping to the old grid's top-row knot, while filtered raw X changes were still magnified across a large horizontal range. A calibration model is valid only for the same gaze preprocessing used during collection, so this was a profile/runtime compatibility error rather than evidence that another screen-space smoother was needed.

New MobileGaze calibrations now store preprocessing version raw_median_v1 and the effective temporal median window in backend metadata. Tracking compares those fields with the active MobileGaze configuration. Profiles with missing, different, or invalid preprocessing metadata are reported as mismatched and cannot enable real cursor movement. The existing May profile is intentionally incompatible and must be replaced by a fresh calibration collected with the five-sample raw filter.

Focused startup and adapter tests pass with 40 tests, compilation passes, and the full automated suite passes with 171 tests. No cursor movement, mapping, edge-reach, or One Euro parameters were changed in this update.
## Update: Worst-Point Calibration Quality Guard (2026-09-29)

The first compatible raw-median MobileGaze calibration improved average mapping metrics and eliminated clipping, but it still contained severe local errors that the previous diagnostics missed. Point 9 was 421px wrong on X and point 1 was 306px wrong on Y. The point means also overlap or cross in several areas, so no existing mapper can recover pixel-accurate targets from this capture.

Held-out evaluation confirmed grid remains the best existing mapper: approximately 165.6px X and 118.4px Y MAE, compared with 300.0/142.4 for IDW and larger errors for linear, ridge, polynomial, and affine models. Robust median or trimmed-mean fitting only traded horizontal error for vertical error and did not justify changing release mapping behavior.

Mapping diagnostics now warn when any individual calibration point exceeds 15% of the calibrated target range on either axis. Runtime recomputation therefore downgrades the current profile from good to needs_review with warnings for 421px X and 306px Y worst-point error. Cursor movement is blocked for this profile unless the existing explicit low-quality override is used. Focused tests pass with 37 tests, compilation passes, and the full suite passes with 172 tests.

## Update: Claude Development Handoff (2026-10-02)

The latest compatible nine-point MobileGaze calibration remains unsuitable for cursor control. Its mean absolute mapping error is approximately 170.9px X and 112.2px Y, raw-sample error is approximately 287.7px X and 209.3px Y, and worst-point errors are 457px X and 676px Y. The profile is correctly classified as needs_review, so runtime forces cursor_enabled=no. This is expected safety behavior.

Root-level `CLAUDE.md` and `docs/CLAUDE_HANDOFF.md` now provide a low-token transfer path for development on another PC. They document the architecture, runtime and calibration flows, safety invariants, Git-ignored external assets, exact current blocker, rejected approaches, focused file-reading order, setup commands, and the next implementation slice.

The next task is calibration acquisition quality: configurable per-target robust outlier rejection, face/head stability gating, cross-target spatial consistency checks, and candidate-profile preservation so rejected calibrations do not replace a prior acceptable profile. Do not spend another iteration on mapper switching, smoothing, offsets, edge boost, or velocity mode before this work. The full automated suite passes with 172 tests.
