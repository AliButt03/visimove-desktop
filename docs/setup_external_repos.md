# External Repository Setup

VisiMove can use selected open-source repositories as isolated real gaze and blink backends. External code lives only in `external/`; first-party app code stays in `src/visimove/`.

The app must continue to run with dummy backends when external repos, dependencies, or model weights are absent.

## Quick Setup

Prepare folders and print instructions:

```powershell
python scripts\setup_external_repos.py
```

Show script options:

```powershell
python scripts\setup_external_repos.py --help
```

Optionally clone missing repos:

```powershell
python scripts\setup_external_repos.py --clone
```

The script does not download model weights, datasets, checkpoints, or heavy dependencies.

## Expected Folder Structure

```text
external/
|-- README.md
|-- eyetrax/
|-- gazefollower/
|-- mobilegaze/
|-- ocec/
```

Empty folders may contain `.gitkeep` placeholders. Once a real external repository is cloned, its own files remain isolated under that folder.

## Repository Plan

### EyeTrax / EyePy

Purpose: first practical real gaze backend for prototype testing.

Expected path:

```text
external/eyetrax/
```

Placeholder clone command:

```powershell
git clone --depth 1 https://github.com/ck-zhang/eyetrax.git external/eyetrax
```

Dependency notes:

- Inspect `external/eyetrax/pyproject.toml` before installing.
- Prefer a separate research environment until adapter dependencies are confirmed.
- Do not install conflicting GUI/camera dependencies into `.venv` blindly.
- EyeTrax requires MediaPipe Tasks and a local `face_landmarker.task`.
- The adapter will not let EyeTrax auto-download model files during tracking.

Actual repo findings:

- No pretrained `gaze_model.pkl`, `.pt`, `.pth`, `.onnx`, `.task`, `.joblib`, or `.pickle` model files are included in `external/eyetrax`.
- EyeTrax is not plug-and-play. It expects a per-user calibration/training step.
- `external/eyetrax/src/eyetrax/gaze.py` defines `GazeEstimator`.
- `GazeEstimator.extract_features(frame)` extracts eye/face features.
- `GazeEstimator.train(X, y)` trains the regression model.
- `GazeEstimator.save_model(path)` writes the calibrated `.pkl`.
- `GazeEstimator.load_model(path)` loads the calibrated `.pkl`.
- `external/eyetrax/src/eyetrax/app/build_model.py` is the repo's model-building script.
- `pyproject.toml` exposes that script as `eyetrax-build-model` after installing EyeTrax.
- Calibration routines live under `external/eyetrax/src/eyetrax/calibration/`.

Expected model weight location:

```text
models/gaze/eyetrax/gaze_model.pkl
models/detection/face_landmarker.task
```

Generate `gaze_model.pkl`:

First place a local MediaPipe Tasks FaceLandmarker model here:

```text
models/detection/face_landmarker.task
```

Official MediaPipe model URL:

```text
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

Check readiness:

```powershell
python scripts\prepare_eyetrax_model.py --check-only
```

Download the FaceLandmarker task file with the helper:

```powershell
python scripts\prepare_eyetrax_model.py --download-face-landmarker
```

Use `--force` only if you need to replace an existing task file:

```powershell
python scripts\prepare_eyetrax_model.py --download-face-landmarker --force
```

Inspect the EyeTrax build command without launching calibration:

```powershell
python scripts\prepare_eyetrax_model.py
```

Launch EyeTrax full-screen calibration:

```powershell
python scripts\prepare_eyetrax_model.py --run
```

Optional arguments:

```powershell
python scripts\prepare_eyetrax_model.py --camera 0 --output models/gaze/eyetrax/gaze_model.pkl --random 60 --retrain-every 10 --run
```

The wrapper sets `PYTHONPATH` to `external/eyetrax/src` and sets `EYETRAX_FACE_LANDMARKER_MODEL` to the local task model so EyeTrax does not need to auto-download that file during calibration.

EyeTrax calibration is interactive and user-specific:

- a full-screen calibration window opens;
- look directly at the calibration points;
- keep the camera, monitor, seating position, and lighting stable;
- regenerate `gaze_model.pkl` for another user, device, camera, monitor, resolution, or seating setup.

Equivalent direct EyeTrax command:

```powershell
$env:PYTHONPATH="external/eyetrax/src"
$env:EYETRAX_FACE_LANDMARKER_MODEL="models/detection/face_landmarker.task"
python -m eyetrax.app.build_model --camera 0 --outfile models/gaze/eyetrax/gaze_model.pkl
```

After `gaze_model.pkl` exists, verify preview mode:

```powershell
python scripts\run_tracking.py --gaze-backend eyetrax --show-debug
```

Config keys:

```yaml
gaze:
  gaze_backend: eyetrax
  backend: eyetrax
  fallback_to_dummy: true

eyetrax:
  repo_path: external/eyetrax
  model_path: models/gaze/eyetrax
  face_landmarker_model_path: models/detection/face_landmarker.task
  use_gpu: false
  input_size: null
```

Preview mode:

```powershell
python scripts/run_tracking.py --gaze-backend eyetrax --show-debug
```

Startup prints whether EyeTrax setup checks passed, whether fallback to dummy occurred, and which model paths are expected. Debug output includes raw gaze, mapped coordinates, smoothed coordinates, confidence, fallback status, and the unavailable reason when EyeTrax cannot provide gaze.

Calibration note:

```powershell
python scripts/run_calibration.py --gaze-backend eyetrax
```

VisiMove refuses to create an EyeTrax calibration profile unless EyeTrax is properly set up, and the current VisiMove calibration UI is not yet wired to collect EyeTrax features. Use EyeTrax calibration tooling to produce a real `gaze_model.pkl`, then run VisiMove tracking preview. Do not create a misleading EyeTrax profile using dummy gaze.

Recommendation:

- Continue with EyeTrax if the goal is a practical prototype and you can run per-user full-screen calibration.
- Switch to GazeFollower or MobileGaze next if you need a pretrained or model-driven gaze estimator that is closer to plug-and-play.

### GazeFollower

Purpose: next real gaze backend candidate after EyeTrax showed unreliable Y-axis behavior on the current setup.

Expected path:

```text
external/gazefollower/
```

Placeholder clone command:

```powershell
git clone --depth 1 https://github.com/GanchengZhu/GazeFollower.git external/gazefollower
```

Dependency notes:

- The inspected repo includes `LICENSE-CC-BY-NC-SA`; treat it as non-commercial/research unless your use is cleared.
- The repo depends on `MNN`, `pygame`, `screeninfo`, `opencv-python`, `numpy`, `pandas`, and a compatible `mediapipe`.
- The current VisiMove adapter prefers GazeFollower's `BlazeFaceAlignment` so it can avoid the classic `mediapipe.solutions.face_mesh` API path that is unstable in this environment.
- Do not install or upgrade heavy/conflicting dependencies until you are ready to test GazeFollower in preview mode.

Inspected model weight locations:

```text
external/gazefollower/gazefollower/res/model_weights/base.mnn
external/gazefollower/gazefollower/res/model_weights/blaze_face.mnn
```

These bundled `.mnn` files are used by default. Do not move them into `src/visimove/`.

Config keys:

```yaml
gaze:
  gaze_backend: gazefollower
  backend: gazefollower
  fallback_to_dummy: true

gazefollower:
  repo_path: external/gazefollower
  model_path: external/gazefollower/gazefollower/res/model_weights/base.mnn
  face_model_path: external/gazefollower/gazefollower/res/model_weights/blaze_face.mnn
  face_alignment_backend: blazeface
```

Preview command:

```powershell
python scripts/run_tracking.py --gaze-backend gazefollower --show-debug
```

If dependencies or model files are missing, VisiMove prints a clear setup error and falls back to dummy gaze when `fallback_to_dummy: true`. Cursor movement is still blocked unless a real backend is active and calibrated.

### MobileGaze

Purpose: PyTorch/ONNX gaze backend and fine-tuning experiments.

Expected path:

```text
external/mobilegaze/
```

Placeholder clone command:

```powershell
git clone --depth 1 https://github.com/yakhyo/gaze-estimation.git external/mobilegaze
```

Dependency notes:

- Keep training and PyTorch-heavy dependencies outside the desktop runtime by default.
- Prefer ONNX exports for real-time VisiMove inference.
- The VisiMove preview adapter only requires `onnxruntime`, `opencv-python`, and `numpy`.
- Use cloud notebooks for training or fine-tuning work.

Expected model weight locations:

```text
external/mobilegaze/weights/mobileone_s0_gaze.onnx
models/gaze/mobilegaze/*.onnx
models/gaze/mobilegaze/*.pt
```

Config keys:

```yaml
gaze:
  gaze_backend: mobilegaze
  backend: mobilegaze

mobilegaze:
  repo_path: external/mobilegaze
  model_path: external/mobilegaze/weights/mobileone_s0_gaze.onnx
  yaw_range_deg: 45.0
  pitch_range_deg: 35.0
  face_crop_scale: 1.15
```

Preview command:

```powershell
python scripts/run_tracking.py --gaze-backend mobilegaze --show-debug
```

### OCEC

Purpose: open/closed eye classification for blink-click support.

Expected path:

```text
external/ocec/
```

Placeholder clone command:

```powershell
git clone --depth 1 https://github.com/PINTO0309/OCEC.git external/ocec
```

Dependency notes:

- Inspect `external/ocec/pyproject.toml` before installing.
- Prefer ONNX Runtime for blink inference in the desktop pipeline.
- Do not commit generated blink datasets, checkpoints, or exported weights.

Expected model weight location:

```text
models/blink/ocec/ocec_l.onnx
```

Config keys:

```yaml
blink:
  backend: ocec
  model_path: models/blink/ocec/ocec_l.onnx
```

## License And Research-Use Warning

Each external repository has its own license, citation, and usage restrictions. Review upstream licenses before using models in demos, reports, publications, redistribution, or commercial settings.

Do not upload restricted raw datasets or restricted derived weights publicly unless the license explicitly allows it.

## Troubleshooting

Missing repository:

- Run `python scripts\setup_external_repos.py` to recreate folders and print clone commands.
- Confirm the expected folder exists under `external/`.
- If an adapter says a repo is missing, keep using `gaze_backend: dummy` or `blink.backend: dummy` until setup is complete.

Missing model weights:

- Confirm the configured `model_path` exists.
- Download weights manually according to the upstream license and documentation.
- Keep weights under `models/gaze/...` or `models/blink/...`, not inside source code.
- If weights are absent, adapters must fail gracefully and VisiMove must continue with dummy backends.

Dependency problems:

- Inspect upstream requirements before installing.
- Prefer a separate virtual environment for research and training.
- Keep the VisiMove `.venv` focused on runtime dependencies.

Fallback behavior:

- Gaze backend missing: fall back to dummy gaze.
- Blink backend missing: fall back to dummy blink.
- Detection backend missing: fall back to MediaPipe/OpenCV or dummy according to configuration.

## Isolation Reminder

Do not copy external code into `src/visimove/`. Build adapter wrappers in first-party code and keep each external repository isolated and updateable.
