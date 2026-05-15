from __future__ import annotations

from dataclasses import dataclass

from visimove.calibration.calibration_store import CalibrationProfile
from visimove.calibration.mapping_model import RegressionMappingModel
from visimove.types import GazeEstimate, ScreenPoint


@dataclass(frozen=True)
class RawCalibrationDomain:
    raw_x_min: float
    raw_x_max: float
    raw_y_min: float
    raw_y_max: float

    @property
    def raw_x_range(self) -> float:
        return self.raw_x_max - self.raw_x_min

    @property
    def raw_y_range(self) -> float:
        return self.raw_y_max - self.raw_y_min


@dataclass(frozen=True)
class RawDomainCheck:
    status: str
    violations: tuple[str, ...]
    original_raw_x: float
    original_raw_y: float
    mapped_input_x: float
    mapped_input_y: float
    domain: RawCalibrationDomain | None

    @property
    def is_outside(self) -> bool:
        return self.status == "outside"


@dataclass(frozen=True)
class MappingDebugInfo:
    raw_x: float
    raw_y: float
    mapped_input_x: float
    mapped_input_y: float
    before_clamp_x: float
    before_clamp_y: float
    after_clamp_x: int
    after_clamp_y: int
    clipped_x: bool
    clipped_y: bool
    raw_domain_status: str = "unavailable"
    raw_domain_violations: tuple[str, ...] = ()
    raw_domain: RawCalibrationDomain | None = None


@dataclass
class CalibrationMapper:
    screen_width: int = 1920
    screen_height: int = 1080
    mapping_model: RegressionMappingModel | None = None
    raw_domain: RawCalibrationDomain | None = None
    clamp_raw_input_to_calibration_domain: bool = False
    raw_domain_margin: float = 0.05

    def map_to_screen(self, gaze: GazeEstimate) -> ScreenPoint:
        return self.map_with_debug(gaze)[0]

    def map_with_debug(self, gaze: GazeEstimate) -> tuple[ScreenPoint, MappingDebugInfo]:
        domain_check = self.check_raw_domain(gaze.point.x, gaze.point.y)
        mapped_input = (domain_check.mapped_input_x, domain_check.mapped_input_y)
        if self.mapping_model is not None:
            prediction = self.mapping_model.predict(mapped_input)
            before_x = prediction.x
            before_y = prediction.y
        else:
            before_x = max(0.0, min(1.0, mapped_input[0])) * (self.screen_width - 1)
            before_y = max(0.0, min(1.0, mapped_input[1])) * (self.screen_height - 1)
        rounded_x = round(before_x)
        rounded_y = round(before_y)
        x = max(0, min(self.screen_width - 1, rounded_x))
        y = max(0, min(self.screen_height - 1, rounded_y))
        debug = MappingDebugInfo(
            raw_x=gaze.point.x,
            raw_y=gaze.point.y,
            mapped_input_x=mapped_input[0],
            mapped_input_y=mapped_input[1],
            before_clamp_x=before_x,
            before_clamp_y=before_y,
            after_clamp_x=x,
            after_clamp_y=y,
            clipped_x=x != rounded_x,
            clipped_y=y != rounded_y,
            raw_domain_status=domain_check.status,
            raw_domain_violations=domain_check.violations,
            raw_domain=domain_check.domain,
        )
        return ScreenPoint(x=x, y=y), debug

    def check_raw_domain(self, raw_x: float, raw_y: float) -> RawDomainCheck:
        if self.raw_domain is None:
            return RawDomainCheck(
                status="unavailable",
                violations=(),
                original_raw_x=raw_x,
                original_raw_y=raw_y,
                mapped_input_x=raw_x,
                mapped_input_y=raw_y,
                domain=None,
            )

        violations: list[str] = []
        violation_x_min = self.raw_domain.raw_x_min - self.raw_domain_margin
        violation_x_max = self.raw_domain.raw_x_max + self.raw_domain_margin
        violation_y_min = self.raw_domain.raw_y_min - self.raw_domain_margin
        violation_y_max = self.raw_domain.raw_y_max + self.raw_domain_margin

        if raw_x < violation_x_min:
            violations.append("raw_x_below_min")
        if raw_x > violation_x_max:
            violations.append("raw_x_above_max")
        if raw_y < violation_y_min:
            violations.append("raw_y_below_min")
        if raw_y > violation_y_max:
            violations.append("raw_y_above_max")

        mapped_x = raw_x
        mapped_y = raw_y
        if self.clamp_raw_input_to_calibration_domain:
            mapped_x = _clamp(
                raw_x,
                self.raw_domain.raw_x_min,
                self.raw_domain.raw_x_max,
            )
            mapped_y = _clamp(
                raw_y,
                self.raw_domain.raw_y_min,
                self.raw_domain.raw_y_max,
            )
        return RawDomainCheck(
            status="outside" if violations else "inside",
            violations=tuple(violations),
            original_raw_x=raw_x,
            original_raw_y=raw_y,
            mapped_input_x=mapped_x,
            mapped_input_y=mapped_y,
            domain=self.raw_domain,
        )

    @classmethod
    def from_profile(cls, profile: CalibrationProfile) -> "CalibrationMapper":
        mapping_model = RegressionMappingModel.from_parameters(profile.mapping_parameters)
        raw_domain = None
        if None not in (profile.raw_x_min, profile.raw_x_max, profile.raw_y_min, profile.raw_y_max):
            raw_domain = RawCalibrationDomain(
                raw_x_min=float(profile.raw_x_min),
                raw_x_max=float(profile.raw_x_max),
                raw_y_min=float(profile.raw_y_min),
                raw_y_max=float(profile.raw_y_max),
            )
        return cls(
            screen_width=profile.screen_width,
            screen_height=profile.screen_height,
            mapping_model=mapping_model,
            raw_domain=raw_domain,
        )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
