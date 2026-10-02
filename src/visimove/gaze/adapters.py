"""Compatibility exports for optional gaze backend adapters.

The real adapter implementations live in backend-specific modules. This module
keeps older imports such as ``visimove.gaze.adapters.MobileGazeAdapter`` working
without routing callers to stale placeholder classes.
"""

from __future__ import annotations

from visimove.gaze.eyetrax_adapter import EyeTraxAdapter, EyeTraxAdapterConfig, EyeTraxSetupStatus
from visimove.gaze.gazefollower_adapter import (
    GazeFollowerAdapter,
    GazeFollowerAdapterConfig,
    GazeFollowerSetupStatus,
)
from visimove.gaze.mobilegaze_adapter import MobileGazeAdapter, MobileGazeAdapterConfig, MobileGazeSetupStatus

__all__ = [
    "EyeTraxAdapter",
    "EyeTraxAdapterConfig",
    "EyeTraxSetupStatus",
    "GazeFollowerAdapter",
    "GazeFollowerAdapterConfig",
    "GazeFollowerSetupStatus",
    "MobileGazeAdapter",
    "MobileGazeAdapterConfig",
    "MobileGazeSetupStatus",
]
