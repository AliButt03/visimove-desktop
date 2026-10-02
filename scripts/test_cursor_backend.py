from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import sleep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visimove.cursor import CursorSafety, CursorSafetyConfig, PyAutoGuiCursorController, Win32CursorController
from visimove.utils.screen import get_screen_bounds


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely smoke-test VisiMove cursor movement backends.")
    parser.add_argument("--backend", choices=("pyautogui", "win32"), default="pyautogui")
    parser.add_argument("--x", type=int, help="Target X coordinate. Defaults to screen center.")
    parser.add_argument("--y", type=int, help="Target Y coordinate. Defaults to screen center.")
    parser.add_argument("--countdown", type=float, default=2.0, help="Seconds to wait before moving.")
    args = parser.parse_args()

    bounds = get_screen_bounds()
    target_x = args.x if args.x is not None else bounds.width // 2
    target_y = args.y if args.y is not None else bounds.height // 2
    target_x, target_y = bounds.clamp(target_x, target_y)

    print(f"Testing cursor backend: {args.backend}")
    print(f"Screen size: {bounds.width}x{bounds.height}")
    print(f"Target: ({target_x}, {target_y})")
    print(f"Moving in {args.countdown:.1f}s. Press Ctrl+C to cancel.")
    sleep(max(0.0, args.countdown))

    safety = CursorSafety(
        bounds=bounds,
        config=CursorSafetyConfig(min_gaze_confidence=0.0, max_speed_px_per_sec=100_000.0),
        paused=False,
    )
    if args.backend == "win32":
        controller = Win32CursorController(safety=safety)
    else:
        controller = PyAutoGuiCursorController(safety=safety, movement_duration_sec=0.05)

    controller.move_to(target_x, target_y, gaze_confidence=1.0, target_visible=True)
    print("Move command sent.")


if __name__ == "__main__":
    main()
