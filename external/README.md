# External Repositories

This folder is reserved for optional third-party backends used by VisiMove. These repositories are not part of the first-party product code.

First-party VisiMove code must stay in `src/visimove/`. External repositories must stay isolated here and must be accessed only through adapter classes such as the gaze and blink adapters under `src/visimove/`.

## Rules

- Do not copy external repository code into `src/visimove/`.
- Do not import external code globally from unrelated modules.
- Do not edit external repo code unless absolutely necessary for local experiments.
- Keep external repos updateable with their own Git history.
- Keep large model weights, datasets, checkpoints, logs, runs, and generated files out of Git.
- The main app must continue to run with dummy backends when these repos are missing.

## Expected Layout

```text
external/
|-- README.md
|-- eyetrax/
|-- gazefollower/
|-- mobilegaze/
|-- ocec/
```

## Planned Backends

| Folder | Purpose | VisiMove adapter |
| --- | --- | --- |
| `external/eyetrax/` | First practical real gaze backend for prototype testing. | `src/visimove/gaze/eyetrax_adapter.py` |
| `external/gazefollower/` | Alternate high-accuracy gaze backend. | `src/visimove/gaze/gazefollower_adapter.py` |
| `external/mobilegaze/` | PyTorch/ONNX gaze backend and fine-tuning experiments. | `src/visimove/gaze/mobilegaze_adapter.py` |
| `external/ocec/` | Open/closed eye classification for blink-click support. | `src/visimove/blink/ocec_adapter.py` |

## Setup

Prepare folders and print manual setup steps:

```powershell
python scripts\setup_external_repos.py
```

Optionally clone missing repos:

```powershell
python scripts\setup_external_repos.py --clone
```

The setup script does not download model weights, datasets, checkpoints, or heavy dependencies automatically.
