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
- EyeTrax VisiMove calibration is wired to collect real EyeTrax samples and save `data/calibration/user_profile_eyetrax.json`.
- Calibration profiles now include quality diagnostics and warnings for pinned/low-range gaze.
- Calibration diagnostics can inspect mapped predictions before and after screen clamping.
- EyeTrax screen calibration now defaults to affine mapping for safer edge behavior.
- Calibration profiles store the calibrated raw gaze domain.
- Runtime tracking clamps mapping input to the actual calibrated raw min/max for safer preview mapping; margin is only violation-detection tolerance.
- Live tracking quality reports `stable`, `unstable`, or `unsafe` based on raw-domain violations and screen clipping.
- Tracking blocks real cursor movement when calibration quality is `poor` or `needs_review` unless a careful testing override is passed.
- Tracking also blocks real cursor movement when live tracking quality is unstable/unsafe unless `--allow-unstable-live-gaze` is explicitly passed.
- Cursor is disabled by default.
- Calibration saves JSON.
- Tests pass: `89 passed in 12.49s`.
- OpenCV detector fallback works.
- MediaPipe has compatibility issues unless the correct Tasks `.task` model is configured.

## Immediate Next Step

Run GazeFollower tracking preview with cursor disabled:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking.py --gaze-backend gazefollower --show-debug
```

Keep your face clearly visible in the preview. The current saved default calibration profile is old/degenerate and is not valid for GazeFollower cursor control.

In the debug lines, inspect:

```text
native_gaze=(...)
native_units=...
raw_gaze=(...)
gaze_conf=...
```
