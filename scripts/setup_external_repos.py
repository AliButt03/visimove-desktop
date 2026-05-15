from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external"


@dataclass(frozen=True)
class ExternalRepo:
    name: str
    url: str
    purpose: str
    model_locations: tuple[str, ...]
    dependency_notes: tuple[str, ...]

    @property
    def path(self) -> Path:
        return EXTERNAL_ROOT / self.name


REPOS = (
    ExternalRepo(
        name="eyetrax",
        url="https://github.com/ck-zhang/eyetrax.git",
        purpose="First practical real gaze backend for prototype testing.",
        model_locations=("models/gaze/eyetrax/gaze_model.pkl",),
        dependency_notes=(
            "Inspect pyproject.toml before installing.",
            "Keep experimental dependencies out of the main runtime until the adapter needs them.",
        ),
    ),
    ExternalRepo(
        name="gazefollower",
        url="https://github.com/GanchengZhu/GazeFollower.git",
        purpose="Alternate high-accuracy webcam gaze backend.",
        model_locations=("models/gaze/gazefollower/model.pth",),
        dependency_notes=(
            "Review upstream research/non-commercial license terms before use.",
            "Inspect requirements.txt in an isolated environment first.",
        ),
    ),
    ExternalRepo(
        name="mobilegaze",
        url="https://github.com/yakhyo/gaze-estimation.git",
        purpose="PyTorch/ONNX gaze backend and later fine-tuning experiments.",
        model_locations=("models/gaze/mobilegaze/*.onnx", "models/gaze/mobilegaze/*.pt"),
        dependency_notes=(
            "Do not install PyTorch-heavy training dependencies into the desktop runtime by default.",
            "Prefer ONNX exports for real-time VisiMove inference.",
        ),
    ),
    ExternalRepo(
        name="ocec",
        url="https://github.com/PINTO0309/OCEC.git",
        purpose="Open/closed eye classification backend for blink-click support.",
        model_locations=("models/blink/ocec/*.onnx",),
        dependency_notes=(
            "Inspect pyproject.toml before installing.",
            "Use exported ONNX weights for the runtime adapter when possible.",
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare optional external VisiMove repositories.")
    parser.add_argument("--clone", action="store_true", help="Clone missing repositories with Git.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing external repo directory before cloning. Use carefully.",
    )
    parser.add_argument("--depth", type=int, default=1, help="Git clone depth when --clone is used.")
    args = parser.parse_args()

    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    print("VisiMove external repository setup")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"External root: {EXTERNAL_ROOT}")
    print("No model weights, datasets, checkpoints, or heavy dependencies are downloaded.")

    git_path = shutil.which("git")
    if args.clone and git_path is None:
        print("Warning: Git was not found. Folders will be prepared, but clone commands will be skipped.")

    for repo in REPOS:
        prepare_repo(repo, clone=args.clone and git_path is not None, force=args.force, depth=args.depth)

    print_model_weight_warning()
    print_folder_structure()


def prepare_repo(repo: ExternalRepo, clone: bool, force: bool, depth: int) -> None:
    print(f"\n[{repo.name}]")
    print(f"Purpose: {repo.purpose}")
    print(f"Path: {repo.path}")
    print(f"Clone command: git clone --depth {depth} {repo.url} {repo.path}")
    print("Expected model locations:")
    for location in repo.model_locations:
        print(f"  - {location}")
    print("Dependency notes:")
    for note in repo.dependency_notes:
        print(f"  - {note}")

    if repo.path.exists() and any(repo.path.iterdir()) and not force:
        print("Status: folder already has content. Leaving it unchanged.")
        return

    if force and repo.path.exists():
        print("Status: removing existing folder before clone.")
        shutil.rmtree(repo.path)

    repo.path.mkdir(parents=True, exist_ok=True)
    if not clone:
        (repo.path / ".gitkeep").touch(exist_ok=True)
        print("Status: folder prepared. Clone manually later if needed.")
        return

    command = ["git", "clone", "--depth", str(depth), repo.url, str(repo.path)]
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    except OSError as exc:
        print(f"Warning: Git clone could not start: {exc}")
        print("Status: folder remains prepared for manual setup.")
        return

    if completed.returncode == 0:
        print("Status: cloned.")
    else:
        print(f"Warning: Git clone failed with exit code {completed.returncode}.")
        print("Status: folder remains prepared for manual setup.")
        (repo.path / ".gitkeep").touch(exist_ok=True)


def print_model_weight_warning() -> None:
    print("\nModel weights and datasets")
    print("- Download model weights manually only after checking upstream licenses.")
    print("- Do not commit weights, datasets, checkpoints, logs, or generated training outputs.")
    print("- The app must continue to run with dummy backends when external repos or weights are missing.")


def print_folder_structure() -> None:
    print("\nExpected external folder structure")
    print("external/")
    print("|-- README.md")
    for repo in REPOS:
        status = "present" if repo.path.exists() else "missing"
        print(f"|-- {repo.name}/ ({status})")


if __name__ == "__main__":
    main()
