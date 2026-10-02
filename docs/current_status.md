# Current Status

VisiMove is currently in the `real_gaze_backend_integration` phase.

The main persistent project memory file is:

- `docs/PROJECT_CONTEXT.md`

The machine-readable project state file is:

- `docs/PROJECT_STATE.json`

Future coding tasks must read both files before making changes and update both files after completing changes.

## Summary

- `scripts/run_tracking.py` works.
- `scripts/run_tracking.py --enable-cursor` works.
- Cursor can move with real gaze in earlier EyeTrax tests, but EyeTrax is not recommended for the current setup because Y-axis output can saturate.
- Tracking startup prints selected detector, gaze, blink, calibration, cursor, camera, frame, and smoothing settings.
- Dummy gaze cursor movement is blocked unless `--enable-cursor --allow-dummy-cursor` are both passed.
- Tracking loads calibration profiles and reports calibration metadata/mismatch warnings.
- `--show-debug` prints throttled raw, mapped, and smoothed gaze coordinates with cursor skip reasons.
- External repository setup is prepared under `external/` for EyeTrax, GazeFollower, MobileGaze, and OCEC.
- EyeTrax backend can be selected for preview/debug mode and falls back safely if required assets are missing.
- EyeTrax FaceLandmarker task and `gaze_model.pkl` now exist locally.
- EyeTrax preview smoke test showed `fallback=no` with cursor disabled, but later guided tracing showed unreliable/saturated Y output.
- GazeFollower has been inspected as the next real gaze backend candidate.
- GazeFollower bundled model files were found at `external/gazefollower/gazefollower/res/model_weights/base.mnn` and `external/gazefollower/gazefollower/res/model_weights/blaze_face.mnn`.
- GazeFollower adapter setup validation and safe dummy fallback are implemented.
- GazeFollower dependencies `MNN 3.5.0`, `pygame 2.6.1`, and `pandas 3.0.3` are installed in the selected virtual environment.
- GazeFollower adapter preflight now passes.
- The previous GazeFollower runtime error `No module named 'pandas'` is fixed.
- The GazeFollower BlazeFace path now avoids the upstream MediaPipe face-alignment initializer, fixing `module 'mediapipe' has no attribute 'solutions'`.
- Tracking debug now prints `native_gaze` and `native_units` when a backend provides native model output.
- GazeFollower preview reaches `fallback=no` and `face_found=yes`.
- GazeFollower native output has been confirmed from the external repo as uncalibrated MNN model coordinates, not Windows screen pixels.
- GazeFollower adapter now labels native units as `gazefollower_model_coordinates` and uses a centered `tanh` transform for VisiMove 0..1 preview raw gaze.
- GazeFollower guided diagnostics showed weak left/center/right separation, so it is not recommended for cursor control in the current setup.
- MobileGaze ONNX weight exists at `external/mobilegaze/weights/mobileone_s0_gaze.onnx`.
- MobileGaze ONNX adapter is wired for preview using VisiMove's detector face crop.
- MobileGaze outputs yaw/pitch radians; VisiMove maps those angles into normalized raw gaze for preview and later calibration.
- MobileGaze live diagnostics showed usable horizontal and vertical separation, but the first pass had the horizontal axis inverted; the adapter has been corrected so left is lower `raw_x` and right is higher `raw_x`.
- MobileGaze calibration is now wired to collect real ONNX gaze output and save `data/calibration/user_profile_mobilegaze.json`.
- MobileGaze calibration has been run and tracking now loads a matching profile with `calibration_quality=good`.
- MobileGaze preview now maps gaze to varied screen coordinates across the display, while cursor remains disabled.
- Latest MobileGaze preview reached `live_tracking_quality=unsafe` because mapped output frequently clipped at screen edges, especially right and bottom.
- MobileGaze calibration mapping diagnostics are healthy: no saved-profile clipping warnings and mapped X/Y ranges cover the calibration target range.
- `scripts/diagnose_live_gaze.py` was fixed to auto-load `data/calibration/user_profile_mobilegaze.json`; the previous guided output with mapped `(1283,721)` for all targets used the wrong/default profile.
- Corrected MobileGaze guided diagnostics show usable separation: left/center/right mapped X means around `88/1146/2373`, and top/bottom mapped Y means around `149/1215`.
- Runtime mapping now clamps input to the affine mapper's fitted input domain, reducing off-screen extrapolation from values that were inside the broad raw sample domain but outside the fitted point-mean domain.
- Post-clamp preview confirms mapped predictions are bounded to the calibrated target area instead of jumping far beyond the screen.
- First MobileGaze cursor-enabled debug run now shows `cursor_skip=none` after pressing `p`, so the gaze pipeline is issuing cursor movement requests.
- A cursor backend selector and smoke-test script are available to compare PyAutoGUI and Win32 cursor movement on Windows.
- Latest MobileGaze cursor run follows the user's gaze, but it is not pinpoint accurate and still jumps. This is now a smoothing, speed-limit, and calibration-quality tuning problem rather than a backend-selection failure.
- Safe edge-reach mode is now available. It expands the inset calibrated target area to the full screen after safe raw-domain mapping, so the cursor is no longer intentionally bounded to the inner calibration rectangle.
- Adaptive stabilization is now available. It uses faster smoothing for intentional movement and a hold/slow mode for fixation jitter.
- Adaptive stabilization now releases based on distance from the current smoothed cursor to the mapped target. This fixes cases where `adjusted_after_gain` was near the intended location but `smoothed` lagged far behind while `smoothing_state=slow`.
- Latest debug confirms adaptive release is working because `smoothing_state=moving` appears when the cursor is far from the mapped target.
- The current live issue is target amplification: `edge_boost=yes` and `--vertical-offset -220` can push the adjusted target to screen extremes and make the cursor chase a shaky target.
- The saved MobileGaze calibration profile is internally consistent. Its center raw mean is about `(0.428, 0.665)`, while latest live center-looking samples are closer to left/bottom raw values such as `raw_x=0.22-0.33` and `raw_y=0.70-0.90`.
- This means the current live session has drifted away from the saved calibration posture/camera setup. Recalibration is the next clean step; more gain/offset would only mask the problem.
- Edge boost is now available. It helps near-edge gaze reach screen extremes without requiring the user to look outside the screen.
- Latest MobileGaze recalibration produced `calibration_quality=good`, but preview still showed normal in-domain samples being forced to top/edge targets because runtime mapping was clamping to the fitted point-mean domain.
- Runtime mapping input domain clamping is now disabled by default. Raw calibration-domain clamping stays active, but in-domain MobileGaze noise is no longer forced to the nearest point-mean boundary.
- Calibration quality per-point diagnostics now preserve collection order instead of sorting by `(x,y)`.
- Follow-up preview showed that MobileGaze's min/max affine mapper still extrapolates too aggressively from noisy samples, even after the clamp fix.
- MobileGaze calibration now uses a backend-specific mapping default: `auto` model selection trained on point means, while the general default remains affine over point means.
- A later MobileGaze recalibration selected `calibration model type: polynomial`; its point-mean error looked acceptable, but live preview and sample-level diagnostics showed severe instability.
- Current diagnostics now report raw-sample mapping stability. The latest polynomial MobileGaze profile maps `46%` of raw calibration samples outside the screen, with raw-sample mean absolute error around `564px` X and `408px` Y.
- Startup now re-checks saved profile mapping diagnostics, including raw-sample instability, and downgrades unsafe profiles to `needs_review` even when their point-mean diagnostics look good.
- MobileGaze `auto` mapping now includes a bounded `idw` point-mean interpolation candidate. On the current calibration samples it selects `idw`, avoids off-screen raw-sample predictions, and prevents affine/polynomial extrapolation from snapping preview to screen corners.
- The latest MobileGaze recalibration saved `mapping model type: idw` with `calibration_quality=good`. Diagnostics show point-mean error `0px`, raw-sample clipped prediction ratio `0%`, no warnings, and raw-sample MAE around `317px` X and `158px` Y.
- Preview with cursor disabled, edge reach disabled, edge boost disabled, and vertical offset `0` showed `live_tracking_quality=stable`, no raw-domain violations, and mapped output moving without corner snapping.
- First cursor-enabled IDW baseline showed `cursor_skip=none`, `live_tracking_quality=stable`, and real cursor movement, but the cursor is still shaky and does not reach the physical screen edges or exact intended target.
- Not reaching edges is expected with `--no-edge-reach`: bounded IDW maps to the calibration target area (`x=307..2252`, `y=173..1266`), not the full `2560x1440` screen.
- The current shake is target noise, not cursor backend failure. Debug still shows `smoothing_state=moving`, so adaptive smoothing is chasing changing MobileGaze targets instead of entering a steadier fixation state.
- A new bounded `grid` mapping candidate is implemented. On the current MobileGaze samples it beats IDW in dry-run scoring: raw-sample MAE is about `262px` X and `121px` Y with `0%` raw-sample clipping.
- The latest grid/edge-reach cursor test is much better: the cursor reaches close to the intended target and edge reach can produce full-screen targets.
- The top-right miss is now mainly smoothing lag, not mapping failure. Debug shows examples where `mapped_after_clamp=(2252,173)` becomes `edge_reach_after=(2559,0)` and `adjusted_after_gain=(2559,0)`, but the final `smoothed` cursor remains behind.
- Remaining shake is still MobileGaze target noise plus adaptive smoothing staying in `moving` state.
- The latest user observation is positive overall: cursor movement is much better, with remaining issues of slow response, occasional eye-tracking loss, and mild shaking.
- Cursor tuning has been updated for smoother default behavior: edge reach is now enabled by default, PyAutoGUI movement duration is `0`, cursor speed cap is higher, adaptive smoothing uses a wider fixation lock, and the smoother holds its last position when face/eyes or live gaze quality are not valid.
- A revised Chapter 5 implementation document has been generated at `docs/VisiMove_Chapter_5_Implementation_Revised.docx`.
- The separate frontend project at `C:\Users\MNA\Desktop\FYP\visimove` has been inspected and can be treated as the completed 20% frontend implementation for the report scope.
- A new 30% scoped implementation document has been generated at `docs/VisiMove_Chapter_5_Implementation_30_Percent.docx`; it covers 20% completed frontend plus only 10% backend prototype work.
- The median-window/demo-stable cursor tuning attempt was reverted because it made the previous cursor behavior worse.
- Cursor/smoothing no longer freeze on `live_tracking_quality=unstable`; only `unsafe` blocks by default. This should reduce the cursor stopping on its own during brief low-ratio live-quality blips.
- Adaptive smoothing now confirms large target jumps before moving. A single noisy MobileGaze spike is held as `smoothing_state=confirming`; if the new target repeats, the cursor releases normally. This is intended to reduce jumpiness without using the rejected median-window approach.
- Adaptive smoothing is now edge-aware. When `adjusted_after_gain` reaches a physical screen edge or corner, the smoother uses faster edge catch-up and snaps the last few pixels to the edge instead of holding short of the corner.
- Edge boost should stay disabled for now; edge reach alone is the safer full-screen reach mode.
- Real blink detection is not integrated yet. The current dummy blink backend only supports keyboard-triggered fake blink testing.
- EyeTrax VisiMove calibration is wired to collect real EyeTrax samples and save `data/calibration/user_profile_eyetrax.json`.
- Calibration profiles now include quality diagnostics and warnings for pinned/low-range gaze.
- Calibration diagnostics can inspect mapped predictions before and after screen clamping.
- EyeTrax screen calibration now defaults to affine mapping for safer edge behavior.
- Calibration profiles store the calibrated raw gaze domain.
- Runtime tracking clamps mapping input to the actual calibrated raw min/max for safer preview mapping; margin is only violation-detection tolerance.
- Live tracking quality reports `stable`, `unstable`, or `unsafe` based on raw-domain violations and screen clipping.
- Tracking blocks real cursor movement when calibration quality is `poor` or `needs_review` unless a careful testing override is passed.
- Tracking blocks real cursor movement when live tracking quality is `unsafe`; `unstable` is reported in debug but does not freeze the cursor by default.
- Cursor is disabled by default.
- Calibration saves JSON.
- Tests pass: `131 passed in 4.67s`.
- OpenCV detector fallback works.
- MediaPipe has compatibility issues unless the correct Tasks `.task` model is configured.

## Immediate Next Step

For the report implementation chapter, use:

```text
docs/VisiMove_Chapter_5_Implementation_30_Percent.docx
```

It intentionally presents frontend as complete and backend as still in development.

Keep the current good MobileGaze calibration and use the restored previous smoother defaults:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --show-debug --cursor-backend win32 --no-edge-boost --vertical-offset 0
```

Watch `adjusted_after_gain`, `smoothed`, `smoothing_state`, and `cursor_skip`. Brief `smoothing_state=confirming` is expected when the target jumps suddenly. At physical screen edges, `smoothing_state=edge` or `edge_hold` means the edge catch-up path is active. `cursor_skip=live gaze outside calibrated domain` should now appear only when `live_tracking_quality=unsafe`, not when it is merely `unstable`.

## 2026-09-26 Cursor Architecture Update

- Repeated live logs proved that MobileGaze mapped targets can change by hundreds or thousands of pixels between frames even while domain quality is reported as stable. This made absolute pointer placement inherently shaky.
- A bounded velocity cursor mode is implemented and is now the release default. Gaze direction controls incremental movement; a center deadzone stops the cursor.
- Velocity movement is capped at 900 pixels/second with a 0.10-second frame-gap cap and an 8-pixel screen margin, so one noisy sample cannot teleport the cursor across the display or enter a PyAutoGUI fail-safe corner.
- The active realtime pipeline now integrates dwell clicking. Dwell runs only while velocity state is velocity_hold, fires once after 1200 ms, and requires movement away from the anchor before another click.
- Pause/resume re-synchronizes velocity state with the physical cursor position for PyAutoGUI and Win32 controllers.
- Automated tests cover bounded motion, frame-gap protection, edge clamping, missing gaze, pause synchronization, one-shot dwell, dwell reset, dwell movement gating, configuration selection, and pipeline integration.
- Live webcam validation of velocity steering and dwell clicking is still required.
