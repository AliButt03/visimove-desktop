# Calibration

Calibration maps raw gaze output into Windows screen coordinates. VisiMove needs it because every gaze backend reports values in its own coordinate space, and each webcam, monitor size, seating position, and user setup changes that relationship.

The tracking pipeline can run without calibration in preview/debug mode, but real cursor control should use a valid profile created with the same gaze backend, detector backend, camera, and screen size.

## Dummy Gaze

The dummy gaze backend generates synthetic movement for pipeline testing. It is useful for checking camera, preview, smoothing, performance logging, and cursor safety, but it is not real eye gaze.

A calibration profile created with dummy gaze is not valid for EyeTrax, GazeFollower, MobileGaze, or any other real gaze backend. Recalibrate after switching gaze backend, detector backend, camera index, monitor, resolution, or scaling.

## Profile Loading

Tracking chooses a calibration profile in this order:

1. `--calibration-profile path`
2. `config/default.yaml` `calibration.profile_path`
3. `data/calibration/user_profile.json` if it exists and no config path was set
4. no loaded calibration

At startup, `scripts/run_tracking.py` prints the profile path, loaded status, model type, saved gaze/detector backend, saved screen size, current screen size, and timestamp.

## Profile Fields

Newly saved profiles include:

- `timestamp`
- `gaze_backend`
- `detector_backend`
- `blink_backend`
- `camera_index`
- `screen_width`
- `screen_height`
- `calibration_point_layout`
- `calibration_points`
- `raw_gaze_samples`
- `target_screen_coordinates`
- `mapping_model_type`
- `mapping_parameters`
- `backend_metadata`
- `sample_count_per_point`
- `confidence_statistics`
- `calibration_quality`
- `calibration_warnings`
- `quality_metrics`

## Warnings

Tracking warns when:

- no calibration profile is loaded
- a profile was created with dummy gaze
- profile gaze backend differs from the current gaze backend
- profile detector backend differs from the current detector backend
- profile screen size differs from the current screen size

When cursor movement is requested with `--enable-cursor`, missing or mismatched calibration blocks real cursor movement unless `--allow-dummy-cursor` is used for controlled pipeline testing.

## Recommended Workflow

Before real backend integration:

```powershell
python scripts/run_tracking.py --gaze-backend dummy --show-debug
python scripts/run_calibration.py --gaze-backend dummy
python scripts/run_tracking.py --gaze-backend dummy --show-debug
```

After a real backend is configured:

```powershell
python scripts/run_tracking.py --gaze-backend eyetrax --show-debug
python scripts/run_calibration.py --gaze-backend eyetrax
python scripts/run_tracking.py --gaze-backend eyetrax --show-debug
python scripts/run_tracking.py --gaze-backend eyetrax --enable-cursor
```

Use preview/debug mode first. Enable cursor movement only after gaze values look stable, calibration metadata matches the active backend, and cursor safety is understood.

## EyeTrax Calibration Safety

EyeTrax needs a real saved gaze model, normally `models/gaze/eyetrax/gaze_model.pkl`, plus a local `models/detection/face_landmarker.task`.

EyeTrax model preparation is separate from VisiMove screen calibration. First create the EyeTrax per-user model:

```powershell
python scripts/prepare_eyetrax_model.py --check-only
python scripts/prepare_eyetrax_model.py --download-face-landmarker
python scripts/prepare_eyetrax_model.py --run
```

EyeTrax model preparation is not the same as VisiMove screen calibration:

- EyeTrax `gaze_model.pkl` is a per-user model that lets EyeTrax estimate gaze/screen output from webcam features.
- VisiMove calibration maps that EyeTrax output into the current Windows screen coordinate system and stores safety diagnostics.

VisiMove refuses `python scripts/run_calibration.py --gaze-backend eyetrax` if EyeTrax is not set up. It does not create an EyeTrax profile from dummy or mouse-position data. Use EyeTrax calibration tooling to create the EyeTrax model first, then verify VisiMove preview/debug output:

```powershell
python scripts/run_tracking.py --gaze-backend eyetrax --show-debug
```

When EyeTrax setup is ready, run:

```powershell
python scripts/run_calibration.py --gaze-backend eyetrax
```

This opens the full-screen VisiMove calibration UI, reads webcam frames, calls the EyeTrax adapter, stores real `raw_x/raw_y/confidence` samples, trains the VisiMove mapping model, and saves:

```text
data/calibration/user_profile_eyetrax.json
```

Recalibrate after selecting EyeTrax or changing the gaze backend, detector backend, camera, monitor, resolution, scaling, user, seating position, or lighting.

Do not use dummy calibration for EyeTrax. After `gaze_model.pkl` is prepared and preview shows `fallback=no`, run VisiMove calibration with the EyeTrax backend so the VisiMove screen mapper is matched to real EyeTrax output.

## Calibration Quality

After calibration, VisiMove prints and saves diagnostics:

- number of points with valid samples
- total and valid sample count
- valid samples per point
- raw gaze min/max/range for x and y
- confidence min/mean/max
- mapping model training success
- mapping prediction error and clipping warnings
- warnings for pinned values, low range, low confidence, missing points, low sample count, or broken mapped screen output

Quality values:

- `good`: mapping trained successfully, raw gaze ranges are healthy, mapped screen output covers the screen, and no serious clipping was detected.
- `acceptable`: mapping is usable for careful testing, usually with only one mildly weak calibration point and otherwise healthy output.
- `needs_review`: mapping may work in preview, but multiple weak points, borderline ranges, or clipping warnings need another calibration pass.
- `poor`: mapping failed, raw gaze is pinned, mapped screen output is stuck, predictions clip heavily, or training error is high.

Raw gaze range alone is not enough. The mapping model must also produce screen coordinates that move across both X and Y. If mapped Y sticks at `0`, predicted Y is probably negative before clamping to the top edge. Cursor movement must not be enabled in that state.

Saved calibration quality and live tracking quality are separate:

- `calibration_quality` describes the saved calibration samples and mapping fit.
- `live_tracking_quality` describes whether current live EyeTrax raw gaze is staying inside the raw gaze domain captured during calibration.

A profile can be `good` while live gaze is `unsafe` if the current raw gaze leaves the calibrated raw range. For example, if calibration captured `raw_y_min=0.306` but live EyeTrax reports `raw_y=0.000`, affine mapping extrapolates above the screen and Windows screen clamping forces mapped Y to `0`.

VisiMove stores the calibrated raw domain in the profile:

- `raw_x_min`
- `raw_x_max`
- `raw_y_min`
- `raw_y_max`
- `raw_x_range`
- `raw_y_range`

At runtime, VisiMove checks each live raw gaze sample against that domain. `calibration.raw_domain_margin` is used only as violation-detection tolerance. With `calibration.clamp_raw_input_to_calibration_domain: true`, the mapping input is clamped to the actual calibrated raw min/max before mapping. The original raw gaze is still printed in debug output, along with `mapped_input_after_domain_clamp`.

Live cursor safety:

- `stable`: recent live samples are inside the calibrated raw domain.
- `unstable`: some recent samples are outside the domain.
- `unsafe`: too many recent samples are outside the domain or screen output is frequently clipped.

Cursor movement is blocked when live tracking quality is `unstable` or `unsafe` unless `--allow-unstable-live-gaze` is passed. That flag is only for careful testing, not normal use.

If many samples are pinned near `raw_x=1.0` or `raw_y=0.0`, or if raw or mapped gaze range is too small, the profile is marked `needs_review` or `poor`. Tracking still allows preview/debug mode with a low-quality profile, but blocks real cursor movement by default. `acceptable` profiles are allowed by default for careful cursor testing through `calibration.allow_acceptable_calibration_for_cursor: true`. Use `--allow-low-quality-calibration` only when deliberately testing a `needs_review` or `poor` profile.

To diagnose a saved profile:

```powershell
python scripts/diagnose_calibration_mapping.py --profile data/calibration/user_profile_eyetrax.json
```

The diagnostic prints raw ranges, target ranges, per-point mapped predictions before and after clamping, prediction errors, RMSE, clipping ratio, negative-Y ratio, raw-sample mapping error, raw-sample clipping ratio, and whether mapped Y is stuck at the top edge. The raw-sample metrics matter because a model can fit the nine averaged calibration dots while still sending noisy live samples far outside the screen.

To diagnose live left/right bias after a good calibration:

```powershell
python scripts/diagnose_live_gaze.py --gaze-backend eyetrax --guided
```

Guided live diagnosis asks you to look left, center, right, top, and bottom. For each direction it reports raw gaze range, mapped screen range, smoothed range, confidence, live-domain violations, and raw X saturation. It warns when left gaze separation is weak, the right side is much stronger than the left, horizontal range is too small, or raw X is often saturated near `1.0`.

For root-cause tracing of the EyeTrax path, use:

```powershell
python scripts/trace_eyetrax_pipeline.py --guided
```

This script does not apply gain or offset correction. It traces:

- VisiMove detector face status
- EyeTrax FaceLandmarker face status and landmark count
- EyeTrax feature vector count and a short feature preview
- EyeTrax model prediction before adapter conversion
- adapter `raw_x/raw_y` after normalizing the EyeTrax prediction
- VisiMove mapped screen coordinate
- confidence, blink status, and invalid reason

EyeTrax trains its `gaze_model.pkl` against absolute screen pixel targets, so the raw EyeTrax model prediction should already look like screen coordinates. The VisiMove adapter normalizes those screen pixels into `raw_x/raw_y` so the standard calibration mapper can consume them. If EyeTrax pixel predictions cannot separate far-left, center, and far-right, the issue is in the EyeTrax model/calibration for the current camera/user setup. If EyeTrax prediction separates correctly but VisiMove mapped X does not, inspect VisiMove calibration sample-target pairing and mapping parameters.

The expected VisiMove 9-point order is row-major:

```text
top-left -> top-center -> top-right ->
middle-left -> center -> middle-right ->
bottom-left -> bottom-center -> bottom-right
```

The saved profile should show target coordinates in that order. Sample means are grouped by their exact target coordinate before fitting, so raw samples remain paired with the intended screen target.

## Axis Gain Tuning

Some EyeTrax sessions produce stable gaze but a compressed or biased horizontal range. In that case, calibration can be good while live cursor movement still stays near the center/right side. Do not use gain/offset as a first fix; run `scripts/trace_eyetrax_pipeline.py --guided` first to determine whether the model, adapter, or mapping is the source of the bias. VisiMove supports a small post-mapping adjustment after calibration and before smoothing:

```text
adjusted_x = screen_center_x + (mapped_x - screen_center_x) * horizontal_gain + horizontal_offset
adjusted_y = screen_center_y + (mapped_y - screen_center_y) * vertical_gain + vertical_offset
```

Config keys:

- `calibration.horizontal_gain`
- `calibration.vertical_gain`
- `calibration.horizontal_offset`
- `calibration.vertical_offset`
- `calibration.center_bias_correction`
- `calibration.allow_axis_gain_tuning`

Use preview first:

```powershell
python scripts/run_tracking.py --gaze-backend eyetrax --show-debug --horizontal-gain 1.2
python scripts/run_tracking.py --gaze-backend eyetrax --show-debug --horizontal-gain 1.3
python scripts/run_tracking.py --gaze-backend eyetrax --show-debug --horizontal-gain 1.4
```

Debug output includes `mapped_after_clamp`, `adjusted_after_gain`, `smoothed`, gain values, and offsets. Cursor remains disabled unless `--enable-cursor` is passed. Values above `2.0` can amplify noise and should be avoided except for brief controlled testing.

Recommended safe flow:

1. Run the guided live gaze diagnostic.
2. Try preview with `--horizontal-gain 1.2`, then `1.3`, then `1.4` if needed.
3. Enable cursor only after preview shows left, center, and right coverage and `live_tracking_quality=stable`.

Tracking debug prints the live domain state:

```text
raw_gaze=(1.000,0.000)
calibration_raw_domain=x[0.005,1.000],y[0.306,1.000]
raw_domain_status=outside
raw_domain_violation=raw_y_below_min
mapped_input_after_domain_clamp=(1.000,0.306)
mapped_raw_before_screen_clamp=(2252.0,173.0)
mapped_after_clamp=(2252,173)
live_tracking_quality=unsafe
```

Safe EyeTrax flow:

```powershell
python scripts/prepare_eyetrax_model.py --check-only
python scripts/run_tracking.py --gaze-backend eyetrax --show-debug
python scripts/run_calibration.py --gaze-backend eyetrax
python scripts/run_tracking.py --gaze-backend eyetrax --show-debug
python scripts/trace_eyetrax_pipeline.py --guided
python scripts/diagnose_live_gaze.py --gaze-backend eyetrax --guided
python scripts/run_tracking.py --gaze-backend eyetrax --enable-cursor
```

Only run the final cursor command when `fallback=no`, calibration metadata matches the active backend, and `calibration_quality` is `good`.

For EyeTrax, the default VisiMove screen mapping is `affine`, which maps calibration-point raw min/max values to screen min/max values. This is intentionally simpler and more stable than a ridge or polynomial fit when EyeTrax output is already screen-like but may be nonlinear near edges. Ridge and polynomial remain available for experiments, but they should be validated with `diagnose_calibration_mapping.py` before cursor testing.

## GazeFollower Calibration Readiness

GazeFollower is the next backend to inspect after EyeTrax showed unreliable Y-axis saturation on this setup. The adapter validates `external/gazefollower/`, the bundled MNN weights, and optional dependencies before it attempts inference:

```text
external/gazefollower/gazefollower/res/model_weights/base.mnn
external/gazefollower/gazefollower/res/model_weights/blaze_face.mnn
```

Preview first:

```powershell
python scripts/run_tracking.py --gaze-backend gazefollower --show-debug
```

If the startup summary reports missing `MNN` or `pygame`, install them in the selected virtual environment before expecting real GazeFollower output. Until GazeFollower produces real preview output and a backend-specific VisiMove calibration is created, cursor movement must remain disabled.

Current GazeFollower preview status:

```text
gaze_backend=gazefollower
fallback=no
face_found=yes
native_gaze=(0.367,-9.740)
raw_gaze=(0.000,0.000)
gaze_conf=0.80
```

The external code confirms that `raw_gaze_coordinates` is `res[:2]` from the MNN model output. GazeFollower's own screen coordinate path uses SVR calibration over the full `features` vector, then converts the calibrated prediction to pixels. That means the native values are uncalibrated model coordinates, not Windows screen pixels.

VisiMove now labels the native units as `gazefollower_model_coordinates` and maps them into 0..1 preview space with a centered `tanh` transform:

```yaml
gazefollower:
  native_output_mode: model_coordinates
  native_coordinate_scale_x: 10.0
  native_coordinate_scale_y: 10.0
```

This prevents small/negative native values from being incorrectly divided by the screen size and pinned to `(0,0)`. It does not make GazeFollower safe for cursor control by itself. Use preview first and only create a GazeFollower calibration profile if `raw_gaze` changes meaningfully as you look left/right/up/down.

## MobileGaze Calibration Readiness

MobileGaze is now wired for preview through ONNX Runtime. It uses VisiMove's detected face box as the model input crop and expects:

```text
external/mobilegaze/weights/mobileone_s0_gaze.onnx
```

MobileGaze outputs gaze angles:

- `yaw`: horizontal gaze angle in radians
- `pitch`: vertical gaze angle in radians

These are not screen coordinates. The adapter maps yaw/pitch into normalized raw gaze using configured angle ranges before VisiMove screen calibration:

```yaml
mobilegaze:
  yaw_range_deg: 45.0
  pitch_range_deg: 35.0
```

Live preview showed MobileGaze had useful directional separation, but the first adapter pass mapped the horizontal axis backwards: left gaze produced a larger `raw_x` than right gaze. The adapter now uses MobileGaze's yaw sign so left maps toward `raw_x=0`, center toward `0.5`, and right toward `raw_x=1`.

Recommended safe workflow:

```powershell
python scripts/run_tracking.py --gaze-backend mobilegaze --show-debug
python scripts/diagnose_live_gaze.py --gaze-backend mobilegaze --guided
python scripts/run_calibration.py --gaze-backend mobilegaze --verbose-quality
python scripts/run_tracking.py --gaze-backend mobilegaze --show-debug
```

MobileGaze calibration saves to `data/calibration/user_profile_mobilegaze.json` by default. Do not enable cursor until that matching MobileGaze calibration profile has acceptable or good quality and live tracking is stable.

MobileGaze uses backend-specific mapping defaults:

```yaml
calibration:
  mobilegaze_mapping_model: auto
  mobilegaze_mapping_fit_strategy: point_means
```

`auto` trains candidate mapping models and saves the one with the best stability score. It now includes bounded `grid` and `idw` point-mean interpolation models before polynomial, linear, and affine candidates. The score still uses calibration point means, but it also penalizes candidates whose individual raw calibration samples map outside the screen. This is important because MobileGaze yaw/pitch samples are noisy: polynomial, linear, or affine fits can look good on the nine averaged dots but still produce huge off-screen predictions between those means.

`grid` maps the calibrated left/center/right columns and top/middle/bottom rows independently. This is useful for MobileGaze because it can reach the calibrated row/column edges when live yaw/pitch moves beyond the point means, while remaining bounded by the calibration target rectangle.

Current MobileGaze status:

- A MobileGaze profile exists, but the current saved profile should be treated as `needs_review` until it is recalibrated with the bounded `idw` auto candidate.
- The latest saved affine profile still maps about `36%` of raw calibration samples outside the screen, with raw-sample mean absolute error around `549px` X and `250px` Y.
- A dry-run of the new bounded `idw` model on the current calibration samples keeps raw-sample clipped prediction ratio at `0%`, but raw-sample error is still high. Recalibrate and preview before any cursor-enabled run.
- After IDW cursor testing, a dry-run of the new `grid` candidate on the same samples reduced raw-sample MAE to about `262px` X and `121px` Y, still with `0%` raw-sample clipping. Recalibrate again so `auto` can save `grid` if it remains the best candidate.
- Startup now recomputes mapping diagnostics and downgrades unstable profiles to `needs_review` even if the saved JSON says `good`.
- The live guided diagnostic previously loaded the wrong default calibration profile; rerun it after the profile-selection fix before making cursor decisions.
- Corrected guided diagnostics show useful directional separation: left/center/right mapped X means around `88/1146/2373`, and top/bottom mapped Y means around `149/1215`.
- Runtime mapping input-domain clamping is disabled by default for MobileGaze noise, while raw calibration-domain clamping remains enabled.

Safe next flow:

```powershell
python scripts\diagnose_calibration_mapping.py --profile data\calibration\user_profile_mobilegaze.json
python scripts\run_calibration.py --gaze-backend mobilegaze --verbose-quality
python scripts\diagnose_calibration_mapping.py --profile data\calibration\user_profile_mobilegaze.json
python scripts\diagnose_live_gaze.py --gaze-backend mobilegaze --guided
python scripts\run_tracking.py --gaze-backend mobilegaze --show-debug --no-edge-reach --no-edge-boost --vertical-offset 0
```

Ignore older guided diagnostic output if every target reported mapped values stuck around `(1283,721)`. That indicated the diagnostic was using the old default profile, not `user_profile_mobilegaze.json`. Do not run the cursor command while calibration diagnostics warn about high point-mean error or raw-sample instability, while corrected debug output frequently shows `mapped_after_clamp` at `x=0`, `x=2559`, or `y=1439`, or while `live_tracking_quality=unsafe`. If final preview is mostly stable, the cursor can be tested carefully with:

```powershell
python scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor
```

If cursor movement is blocked only by occasional `live_tracking_quality=unstable`, use `--allow-unstable-live-gaze` only for a short controlled test with `p` and `Ctrl+C` ready.

If the cursor still does not visibly move, rerun with debug:

```powershell
python scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --allow-unstable-live-gaze --show-debug
```

Press `p` once to resume and inspect `cursor_skip`. If `cursor_skip=none` but the pointer still does not move, test the cursor backend directly:

```powershell
python scripts\test_cursor_backend.py --backend pyautogui
python scripts\test_cursor_backend.py --backend win32
```

If Win32 works better on Windows, run tracking with:

```powershell
python scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --allow-unstable-live-gaze --show-debug --cursor-backend win32
```

If the cursor follows gaze but jumps too much, tune smoothing before recalibrating. Start with a lower EMA alpha and lower cursor speed:

```powershell
python scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --allow-unstable-live-gaze --show-debug --cursor-backend win32 --smoothing-filter ema --ema-alpha 0.18 --max-speed-px-per-sec 700
```

Lower `--ema-alpha` values reduce jitter but add lag. Lower `--max-speed-px-per-sec` values reduce jumps but make the pointer slower. Do not tune for clicks until the cursor movement is stable enough for preview.

## Safe Edge Reach

VisiMove calibration points are intentionally inset from the physical screen edges. On a `2560x1440` display, a 9-point profile may learn a target area such as `x[307,2252], y[173,1266]`. That protects calibration quality, but it can make the cursor feel like it hits an invisible boundary before the extreme left, right, top, or bottom.

Safe edge-reach mode keeps the raw gaze and mapping input clamps in place, then expands the calibrated target rectangle to the full screen after mapping. This is safer than letting the mapper extrapolate raw gaze outside the calibrated domain.

Default config:

```yaml
calibration:
  edge_reach_enabled: false
  edge_margin_px: 0
```

Recommended MobileGaze baseline test:

```powershell
python scripts\run_tracking.py --gaze-backend mobilegaze --show-debug --no-edge-reach --no-edge-boost --vertical-offset 0
```

Use `--edge-reach --edge-margin-px 0` only after the baseline is stable and the cursor clearly needs more reach toward taskbar/corner controls. Use `--edge-margin-px 20` if hitting screen corners triggers PyAutoGUI failsafe or feels too aggressive.

## Adaptive Stabilization

Fixed smoothing can make the cursor feel worse: a very low EMA alpha reduces shake, but it also creates lag and can leave the pointer far from the intended target. Adaptive stabilization uses a faster response for clear intentional movement and holds/slowly filters small fixation jitter.

Recommended MobileGaze command:

```powershell
python scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --allow-unstable-live-gaze --show-debug --cursor-backend win32 --smoothing-filter adaptive --max-speed-px-per-sec 850 --no-edge-reach --no-edge-boost --vertical-offset 0
```

Watch `smoothing_state` in debug output:

- `moving`: the cursor is responding to a clear gaze move.
- `slow`: movement is being damped.
- `settling`: the cursor is close to fixation but not held yet.
- `hold`: fixation jitter is being suppressed.

If debug shows `adjusted_after_gain` close to the desired target but `smoothed` far away, the issue is smoothing lag rather than calibration. The adaptive filter now releases into `moving` when the smoothed cursor is far from the current mapped target, even if the fixation anchor is already near that target. For live testing, prefer a faster release before adding more gain or offset:

```powershell
python scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --allow-unstable-live-gaze --show-debug --cursor-backend win32 --smoothing-filter adaptive --max-speed-px-per-sec 850 --no-edge-reach --no-edge-boost --vertical-offset 0
```

When the target is far from the cursor, `smoothing_state` should switch to `moving`. When your eyes are still, it should settle toward `hold` instead of continuously wandering.

## Edge Boost

If the cursor only reaches the screen edges when you look outside the screen, enable edge boost. Edge boost is applied after safe mapping/edge reach and makes near-edge gaze move farther toward the actual screen boundary without changing the raw gaze model.

Recommended command:

```powershell
python scripts\run_tracking.py --gaze-backend mobilegaze --enable-cursor --allow-unstable-live-gaze --show-debug --cursor-backend win32 --smoothing-filter adaptive --adaptive-fast-alpha 0.34 --adaptive-slow-alpha 0.06 --adaptive-fixation-radius 42 --adaptive-release-radius 110 --adaptive-hold-ms 90 --max-speed-px-per-sec 850 --edge-reach --edge-margin-px 0 --edge-boost --edge-boost-gamma 0.78
```

Lower gamma means stronger edge boost. Try `0.82` if it becomes too aggressive, or `0.72` if top-right/taskbar controls still require looking outside the screen.

Natural blink detection is not available yet; the dummy blink backend only supports controlled fake blink testing from the preview loop.
