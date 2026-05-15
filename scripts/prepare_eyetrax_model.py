from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EYETRAX_ROOT = PROJECT_ROOT / "external" / "eyetrax"
EYETRAX_SRC = EYETRAX_ROOT / "src"
EYETRAX_BUILD_MODEL = EYETRAX_SRC / "eyetrax" / "app" / "build_model.py"
EYETRAX_GAZE = EYETRAX_SRC / "eyetrax" / "gaze.py"
DEFAULT_FACE_MODEL = PROJECT_ROOT / "models" / "detection" / "face_landmarker.task"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "gaze" / "eyetrax" / "gaze_model.pkl"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


@dataclass(frozen=True)
class ReadinessStatus:
    face_landmarker_present: bool
    eyetrax_repo_present: bool
    build_model_present: bool
    gaze_py_present: bool
    output_model_exists: bool
    dependencies_available: bool
    dependency_error: str | None
    face_model_path: Path
    output_path: Path

    @property
    def ready_to_build(self) -> bool:
        return (
            self.face_landmarker_present
            and self.eyetrax_repo_present
            and self.build_model_present
            and self.gaze_py_present
            and self.dependencies_available
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a real EyeTrax calibrated gaze_model.pkl for VisiMove."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--output",
        "--outfile",
        dest="output",
        default=str(DEFAULT_OUTPUT),
        help="Destination calibrated EyeTrax .pkl model.",
    )
    parser.add_argument("--face-model", default=str(DEFAULT_FACE_MODEL))
    parser.add_argument("--model", default="ridge", help="EyeTrax regression model name.")
    parser.add_argument("--random", type=int, default=60, help="Adaptive random calibration points.")
    parser.add_argument("--retrain-every", type=int, default=10)
    parser.add_argument("--no-show-pred", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Only print readiness checks.")
    parser.add_argument(
        "--download-face-landmarker",
        action="store_true",
        help="Download the official MediaPipe FaceLandmarker task file only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing face_landmarker.task when downloading.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually launch full-screen EyeTrax calibration. Without this, only print the command.",
    )
    args = parser.parse_args()

    face_model = resolve_project_path(args.face_model)
    output = resolve_project_path(args.output)

    if args.download_face_landmarker:
        download_face_landmarker(face_model, force=args.force)

    status = check_readiness(face_model=face_model, output_path=output)
    print_readiness(status)

    if args.check_only:
        return

    if not status.ready_to_build:
        print_missing_setup_instructions(status)
        raise SystemExit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_eyetrax_command(
        camera=args.camera,
        output=output,
        model=args.model,
        random_points=args.random,
        retrain_every=args.retrain_every,
        show_predictions=not args.no_show_pred,
    )
    env = build_eyetrax_env(face_model)

    print_interactive_calibration_notice(output)
    print("Command:")
    print(" ".join(command))

    if not args.run:
        print("\nDry run only. Add --run to launch the full-screen EyeTrax calibration.")
        return

    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    if output.exists():
        print(f"EyeTrax model saved: {output}")
    else:
        raise SystemExit(f"EyeTrax command finished, but output model was not found: {output}")


def check_readiness(face_model: Path = DEFAULT_FACE_MODEL, output_path: Path = DEFAULT_OUTPUT) -> ReadinessStatus:
    dependency_error = validate_eyetrax_imports() if EYETRAX_SRC.exists() else "external/eyetrax/src missing"
    return ReadinessStatus(
        face_landmarker_present=face_model.exists(),
        eyetrax_repo_present=EYETRAX_ROOT.exists() and any(EYETRAX_ROOT.iterdir()),
        build_model_present=EYETRAX_BUILD_MODEL.exists(),
        gaze_py_present=EYETRAX_GAZE.exists(),
        output_model_exists=output_path.exists(),
        dependencies_available=dependency_error is None,
        dependency_error=dependency_error,
        face_model_path=face_model,
        output_path=output_path,
    )


def print_readiness(status: ReadinessStatus) -> None:
    print("EyeTrax preparation readiness")
    print(f"  FaceLandmarker present: {yes_no(status.face_landmarker_present)}")
    print(f"  FaceLandmarker path: {status.face_model_path}")
    print(f"  EyeTrax repo present: {yes_no(status.eyetrax_repo_present)}")
    print(f"  EyeTrax build_model.py present: {yes_no(status.build_model_present)}")
    print(f"  EyeTrax gaze.py present: {yes_no(status.gaze_py_present)}")
    print(f"  Output model exists: {yes_no(status.output_model_exists)}")
    print(f"  Output model path: {status.output_path}")
    print(f"  Dependencies available: {yes_no(status.dependencies_available)}")
    if status.dependency_error:
        print(f"  Dependency error: {status.dependency_error}")
    print(f"  Next action: {next_action(status)}")


def next_action(status: ReadinessStatus) -> str:
    if not status.face_landmarker_present:
        return "download or place models/detection/face_landmarker.task"
    if not status.eyetrax_repo_present:
        return "run scripts/setup_external_repos.py or clone EyeTrax into external/eyetrax"
    if not status.build_model_present or not status.gaze_py_present:
        return "repair external/eyetrax checkout; required EyeTrax files are missing"
    if not status.dependencies_available:
        return "install or fix EyeTrax runtime dependencies"
    if not status.output_model_exists:
        return "run this script with --run to create the user-specific gaze_model.pkl"
    return "run python scripts/run_tracking.py --gaze-backend eyetrax --show-debug"


def print_missing_setup_instructions(status: ReadinessStatus) -> None:
    if not status.face_landmarker_present:
        print("\nMissing FaceLandmarker task model.")
        print("Download the MediaPipe Face Landmarker task file and place it here:")
        print(f"  {DEFAULT_FACE_MODEL}")
        print("Expected URL:")
        print(f"  {FACE_LANDMARKER_URL}")
        print("You can also try:")
        print("  python scripts/prepare_eyetrax_model.py --download-face-landmarker")
    if not status.dependencies_available:
        print("\nEyeTrax dependencies are not ready.")
        print("Inspect external/eyetrax/pyproject.toml before installing packages into .venv.")
    if not status.build_model_present or not status.gaze_py_present:
        print("\nEyeTrax checkout is incomplete. Expected:")
        print(f"  {EYETRAX_BUILD_MODEL}")
        print(f"  {EYETRAX_GAZE}")


def print_interactive_calibration_notice(output: Path) -> None:
    print("\nEyeTrax calibration is interactive and user-specific.")
    print("- A full-screen calibration window will open.")
    print("- Look at each calibration point until the flow completes.")
    print("- Keep your head/camera/screen setup stable.")
    print("- Regenerate this model for a different user, camera, monitor, resolution, or seating setup.")
    print(f"- Output will be saved to: {output}")


def download_face_landmarker(destination: Path, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"FaceLandmarker already exists, not downloading: {destination}")
        print("Use --force to overwrite it.")
        return

    tmp = destination.with_suffix(destination.suffix + ".tmp")
    print("Downloading MediaPipe FaceLandmarker task file")
    print(f"Source URL: {FACE_LANDMARKER_URL}")
    print(f"Destination: {destination}")
    try:
        request = urllib.request.Request(FACE_LANDMARKER_URL, headers={"User-Agent": "visimove"})
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
        tmp.replace(destination)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"FaceLandmarker download failed: {exc}") from exc
    print(f"Downloaded FaceLandmarker task model: {destination}")


def validate_eyetrax_imports() -> str | None:
    added = False
    src = str(EYETRAX_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
        added = True
    try:
        importlib.import_module("eyetrax")
        importlib.import_module("eyetrax.gaze")
        importlib.import_module("eyetrax.app.build_model")
    except Exception as exc:
        return str(exc)
    finally:
        if added:
            try:
                sys.path.remove(src)
            except ValueError:
                pass
    return None


def build_eyetrax_command(
    camera: int,
    output: Path,
    model: str,
    random_points: int,
    retrain_every: int,
    show_predictions: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "eyetrax.app.build_model",
        "--camera",
        str(camera),
        "--outfile",
        str(output),
        "--model",
        model,
        "--random",
        str(random_points),
        "--retrain-every",
        str(retrain_every),
    ]
    if not show_predictions:
        command.append("--no-show-pred")
    return command


def build_eyetrax_env(face_model: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(EYETRAX_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["EYETRAX_FACE_LANDMARKER_MODEL"] = str(face_model)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    main()
