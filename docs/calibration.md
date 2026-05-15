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

The diagnostic prints raw ranges, target ranges, per-point mapped predictions before and after clamping, prediction errors, RMSE, clipping ratio, negative-Y ratio, and whether mapped Y is stuck at the top edge.

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
