# VisiMove Claude Guide

Read this file first. Then read `docs/CLAUDE_HANDOFF.md`. Open other files only when the handoff points to them.

## Repository Scope

This repository is the Windows desktop/backend eye-control application. The separately developed website frontend is not stored here.

## Current Verified Baseline

- Full automated suite: `172 passed` on 2026-10-02.
- Primary real gaze backend: MobileGaze ONNX.
- Primary cursor mode: absolute.
- Primary smoothing filter: bounded One Euro.
- Current MobileGaze calibration profile: `needs_review`.
- Cursor movement is intentionally disabled for that profile by the startup safety gate.
- Current blocker: calibration targets produce overlapping or contradictory raw gaze values. This happens before smoothing and cursor control.

## Mandatory Working Rules

1. Preserve cursor safety defaults. Never silently enable the cursor, disable PyAutoGUI fail-safe, or bypass low-quality/live-domain checks.
2. Keep model paths, thresholds, gains, timing, and quality limits config-driven.
3. Keep external repositories under `external/`; access them only through adapters in `src/visimove/`.
4. Do not commit model weights, calibration profiles, datasets, logs, virtual environments, or external repositories.
5. Do not treat dummy gaze or dummy blink as real functionality.
6. Add focused tests for behavioral changes, then run the complete test suite.
7. Update `docs/CLAUDE_HANDOFF.md`, `docs/PROJECT_CONTEXT.md`, and `docs/PROJECT_STATE.json` after substantial work.
8. Do not rewrite or revert unrelated dirty files. This handoff is being created over a large intentional working tree.

## Fast Read Order

For calibration work:

1. `docs/CLAUDE_HANDOFF.md`
2. `scripts/run_calibration.py`
3. `src/visimove/calibration/calibration_ui.py`
4. `src/visimove/calibration/calibration_quality.py`
5. `src/visimove/calibration/mapping_diagnostics.py`
6. `src/visimove/calibration/mapping_model.py`
7. Relevant tests in `src/visimove/tests/`

For runtime cursor work:

1. `scripts/run_tracking.py`
2. `src/visimove/pipeline/realtime_pipeline.py`
3. `src/visimove/calibration/mapper.py`
4. `src/visimove/smoothing/one_euro_filter.py`
5. `src/visimove/cursor/cursor_safety.py`
6. Cursor controller selected under `src/visimove/cursor/`

## Immediate Next Task

Improve calibration acquisition, not cursor smoothing:

1. Reject target-local outlier samples before fitting.
2. Reject or restart a target when face/head position changes excessively.
3. Validate spatial ordering and separation across target rows and columns.
4. Do not replace the active profile when the new profile is poor or `needs_review`; preserve it as a rejected diagnostic profile.
5. Keep the existing post-fit worst-point checks.
6. Start with synthetic unit tests, then perform a live 9-point calibration.

Do not switch away from grid mapping without new held-out evidence. Grid was the best existing model in prior held-out comparison, but no mapper can recover screen targets whose raw gaze measurements overlap.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src scripts
git diff --check
```

