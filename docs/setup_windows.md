# Windows Setup

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Keep cursor movement disabled until calibration has been tested with `--dry-run`.

## Optional External Gaze Backends

Clone optional external repositories:

```powershell
python scripts\setup_external_repos.py --clone
```

Manual clone commands:

```powershell
git clone --depth 1 https://github.com/GanchengZhu/GazeFollower.git external/gazefollower
git clone --depth 1 https://github.com/ck-zhang/eyetrax.git external/eyetrax
git clone --depth 1 https://github.com/yakhyo/gaze-estimation.git external/mobilegaze
```

Place model weights under:

```text
models/gaze/gazefollower/
models/gaze/eyetrax/gaze_model.pkl
models/gaze/mobilegaze/
```

Select a backend in `config/default.yaml`:

```yaml
gaze:
  gaze_backend: dummy
```

```yaml
gaze:
  gaze_backend: gazefollower
  model_path: models/gaze/gazefollower/model.pth
```

```yaml
gaze:
  gaze_backend: eyetrax
  model_path: models/gaze/eyetrax/gaze_model.pkl
```

```yaml
gaze:
  gaze_backend: mobilegaze
  model_path: models/gaze/mobilegaze/model.onnx
```

## Troubleshooting External Backends

If you see a missing repository message, run:

```powershell
python scripts\setup_external_repos.py --clone
```

If you see a missing model path message, place the required weights under `models/gaze/...` and update `model_path`.

If you see a dependency import error, install that external repo's dependencies only after checking compatibility. The main VisiMove app still runs with the dummy gaze backend when an external backend is unavailable.
