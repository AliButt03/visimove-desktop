from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from visimove.calibration.mapper import MappingDebugInfo


@dataclass
class LiveTrackingQualityMonitor:
    window_size: int = 30
    unsafe_ratio_threshold: float = 0.30
    samples: deque[tuple[bool, bool]] = field(default_factory=deque)

    def update(self, mapping_debug: MappingDebugInfo) -> "LiveTrackingQualityState":
        domain_violation = mapping_debug.raw_domain_status == "outside"
        clipped_output = mapping_debug.clipped_x or mapping_debug.clipped_y
        self.samples.append((domain_violation, clipped_output))
        while len(self.samples) > self.window_size:
            self.samples.popleft()
        return self.state

    @property
    def state(self) -> "LiveTrackingQualityState":
        if not self.samples:
            return LiveTrackingQualityState(
                quality="stable",
                domain_violation_ratio=0.0,
                clipped_output_ratio=0.0,
                sample_count=0,
            )
        domain_ratio = sum(1 for domain, _clip in self.samples if domain) / len(self.samples)
        clipped_ratio = sum(1 for _domain, clip in self.samples if clip) / len(self.samples)
        if domain_ratio >= self.unsafe_ratio_threshold or clipped_ratio >= self.unsafe_ratio_threshold:
            quality = "unsafe"
        elif domain_ratio > 0.0 or clipped_ratio > 0.0:
            quality = "unstable"
        else:
            quality = "stable"
        return LiveTrackingQualityState(
            quality=quality,
            domain_violation_ratio=domain_ratio,
            clipped_output_ratio=clipped_ratio,
            sample_count=len(self.samples),
        )


@dataclass(frozen=True)
class LiveTrackingQualityState:
    quality: str
    domain_violation_ratio: float
    clipped_output_ratio: float
    sample_count: int
