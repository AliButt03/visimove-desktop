from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class MappingModelType(str, Enum):
    AFFINE = "affine"
    GRID = "grid"
    IDW = "idw"
    LINEAR = "linear"
    RIDGE = "ridge"
    POLYNOMIAL = "polynomial"


@dataclass(frozen=True)
class MappingPrediction:
    x: float
    y: float


class MappingModel:
    def fit(self, raw_gaze: list[tuple[float, float]], targets: list[tuple[float, float]]) -> None:
        raise NotImplementedError

    def predict(self, raw_gaze: tuple[float, float]) -> MappingPrediction:
        raise NotImplementedError

    def to_parameters(self) -> dict[str, Any]:
        raise NotImplementedError


class RegressionMappingModel(MappingModel):
    def __init__(
        self,
        model_type: MappingModelType = MappingModelType.LINEAR,
        polynomial_degree: int = 2,
        ridge_alpha: float = 1.0,
        idw_power: float = 2.0,
        idw_epsilon: float = 1e-6,
    ) -> None:
        self.model_type = model_type
        self.polynomial_degree = polynomial_degree
        self.ridge_alpha = ridge_alpha
        self.idw_power = idw_power
        self.idw_epsilon = idw_epsilon
        self.coefficients: np.ndarray | None = None
        self.control_raw: np.ndarray | None = None
        self.control_targets: np.ndarray | None = None
        self.raw_x_knots: np.ndarray | None = None
        self.target_x_knots: np.ndarray | None = None
        self.raw_y_knots: np.ndarray | None = None
        self.target_y_knots: np.ndarray | None = None
        self.input_min: np.ndarray | None = None
        self.input_max: np.ndarray | None = None

    def fit(self, raw_gaze: list[tuple[float, float]], targets: list[tuple[float, float]]) -> None:
        if len(raw_gaze) != len(targets):
            raise ValueError("raw_gaze and targets must contain the same number of samples.")
        if len(raw_gaze) < 2:
            raise ValueError("At least two samples are required to fit a mapping model.")

        raw_array = np.asarray(raw_gaze, dtype=float)
        target_array = np.asarray(targets, dtype=float)
        self.input_min = raw_array.min(axis=0)
        self.input_max = raw_array.max(axis=0)

        if self.model_type is MappingModelType.GRID:
            self._fit_grid(raw_array, target_array)
            self.coefficients = None
            return

        if self.model_type is MappingModelType.IDW:
            self.control_raw = raw_array
            self.control_targets = target_array
            self.coefficients = None
            return

        if self.model_type is MappingModelType.AFFINE:
            self.coefficients = self._fit_affine(raw_array, target_array)
            return

        x_matrix = self._features(raw_array)
        y_matrix = target_array

        if self.model_type is MappingModelType.RIDGE:
            regularizer = self.ridge_alpha * np.eye(x_matrix.shape[1])
            regularizer[0, 0] = 0.0
            self.coefficients = np.linalg.solve(
                x_matrix.T @ x_matrix + regularizer,
                x_matrix.T @ y_matrix,
            )
        else:
            self.coefficients, *_ = np.linalg.lstsq(x_matrix, y_matrix, rcond=None)

    def predict(self, raw_gaze: tuple[float, float]) -> MappingPrediction:
        if self.model_type is MappingModelType.GRID:
            if (
                self.raw_x_knots is None
                or self.target_x_knots is None
                or self.raw_y_knots is None
                or self.target_y_knots is None
            ):
                raise RuntimeError("Mapping model must be fitted before predict().")
            x = float(np.interp(raw_gaze[0], self.raw_x_knots, self.target_x_knots))
            y = float(np.interp(raw_gaze[1], self.raw_y_knots, self.target_y_knots))
            return MappingPrediction(x=x, y=y)

        if self.model_type is MappingModelType.IDW:
            if self.control_raw is None or self.control_targets is None:
                raise RuntimeError("Mapping model must be fitted before predict().")
            raw_array = np.asarray(raw_gaze, dtype=float)
            distances = np.linalg.norm(self.control_raw - raw_array, axis=1)
            nearest_index = int(np.argmin(distances))
            if distances[nearest_index] <= self.idw_epsilon:
                prediction = self.control_targets[nearest_index]
            else:
                weights = 1.0 / np.maximum(distances, self.idw_epsilon) ** self.idw_power
                weights = weights / weights.sum()
                prediction = weights @ self.control_targets
            return MappingPrediction(x=float(prediction[0]), y=float(prediction[1]))

        if self.coefficients is None:
            raise RuntimeError("Mapping model must be fitted before predict().")
        features = self._features(np.asarray([raw_gaze], dtype=float))
        prediction = features @ self.coefficients
        return MappingPrediction(x=float(prediction[0, 0]), y=float(prediction[0, 1]))

    def to_parameters(self) -> dict[str, Any]:
        if self.model_type is MappingModelType.GRID:
            if (
                self.raw_x_knots is None
                or self.target_x_knots is None
                or self.raw_y_knots is None
                or self.target_y_knots is None
            ):
                raise RuntimeError("Mapping model must be fitted before serialization.")
            return {
                "model_type": self.model_type.value,
                "raw_x_knots": self.raw_x_knots.tolist(),
                "target_x_knots": self.target_x_knots.tolist(),
                "raw_y_knots": self.raw_y_knots.tolist(),
                "target_y_knots": self.target_y_knots.tolist(),
                "input_min": None if self.input_min is None else self.input_min.tolist(),
                "input_max": None if self.input_max is None else self.input_max.tolist(),
            }

        if self.model_type is MappingModelType.IDW:
            if self.control_raw is None or self.control_targets is None:
                raise RuntimeError("Mapping model must be fitted before serialization.")
            return {
                "model_type": self.model_type.value,
                "idw_power": self.idw_power,
                "idw_epsilon": self.idw_epsilon,
                "control_raw": self.control_raw.tolist(),
                "control_targets": self.control_targets.tolist(),
                "input_min": None if self.input_min is None else self.input_min.tolist(),
                "input_max": None if self.input_max is None else self.input_max.tolist(),
            }

        if self.coefficients is None:
            raise RuntimeError("Mapping model must be fitted before serialization.")
        return {
            "model_type": self.model_type.value,
            "polynomial_degree": self.polynomial_degree,
            "ridge_alpha": self.ridge_alpha,
            "coefficients": self.coefficients.tolist(),
            "input_min": None if self.input_min is None else self.input_min.tolist(),
            "input_max": None if self.input_max is None else self.input_max.tolist(),
        }

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any]) -> "RegressionMappingModel":
        model = cls(
            model_type=MappingModelType(parameters["model_type"]),
            polynomial_degree=int(parameters.get("polynomial_degree", 2)),
            ridge_alpha=float(parameters.get("ridge_alpha", 1.0)),
            idw_power=float(parameters.get("idw_power", 2.0)),
            idw_epsilon=float(parameters.get("idw_epsilon", 1e-6)),
        )
        if model.model_type is MappingModelType.GRID:
            required = ("raw_x_knots", "target_x_knots", "raw_y_knots", "target_y_knots")
            if any(parameters.get(key) is None for key in required):
                raise ValueError("Grid mapping parameters are missing calibration knots.")
            model.raw_x_knots = np.asarray(parameters["raw_x_knots"], dtype=float)
            model.target_x_knots = np.asarray(parameters["target_x_knots"], dtype=float)
            model.raw_y_knots = np.asarray(parameters["raw_y_knots"], dtype=float)
            model.target_y_knots = np.asarray(parameters["target_y_knots"], dtype=float)
            if parameters.get("input_min") is not None:
                model.input_min = np.asarray(parameters["input_min"], dtype=float)
            if parameters.get("input_max") is not None:
                model.input_max = np.asarray(parameters["input_max"], dtype=float)
            return model

        if model.model_type is MappingModelType.IDW:
            if parameters.get("control_raw") is None or parameters.get("control_targets") is None:
                raise ValueError("IDW mapping parameters are missing control points.")
            model.control_raw = np.asarray(parameters["control_raw"], dtype=float)
            model.control_targets = np.asarray(parameters["control_targets"], dtype=float)
            if parameters.get("input_min") is not None:
                model.input_min = np.asarray(parameters["input_min"], dtype=float)
            if parameters.get("input_max") is not None:
                model.input_max = np.asarray(parameters["input_max"], dtype=float)
            return model

        model.coefficients = np.asarray(parameters["coefficients"], dtype=float)
        if parameters.get("input_min") is not None:
            model.input_min = np.asarray(parameters["input_min"], dtype=float)
        if parameters.get("input_max") is not None:
            model.input_max = np.asarray(parameters["input_max"], dtype=float)
        return model

    def _features(self, raw_gaze: np.ndarray) -> np.ndarray:
        x = raw_gaze[:, 0]
        y = raw_gaze[:, 1]
        if self.model_type is MappingModelType.POLYNOMIAL:
            if self.polynomial_degree != 2:
                raise ValueError("Only polynomial_degree=2 is currently supported.")
            return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
        return np.column_stack([np.ones_like(x), x, y])

    @staticmethod
    def _fit_affine(raw_gaze: np.ndarray, targets: np.ndarray) -> np.ndarray:
        raw_min = raw_gaze.min(axis=0)
        raw_max = raw_gaze.max(axis=0)
        target_min = targets.min(axis=0)
        target_max = targets.max(axis=0)
        raw_range = np.maximum(raw_max - raw_min, 1e-6)
        scale = (target_max - target_min) / raw_range
        intercept = target_min - (raw_min * scale)
        return np.asarray(
            [
                [intercept[0], intercept[1]],
                [scale[0], 0.0],
                [0.0, scale[1]],
            ],
            dtype=float,
        )

    def _fit_grid(self, raw_gaze: np.ndarray, targets: np.ndarray) -> None:
        target_x_knots = np.asarray(sorted(set(float(value) for value in targets[:, 0])), dtype=float)
        target_y_knots = np.asarray(sorted(set(float(value) for value in targets[:, 1])), dtype=float)
        if len(target_x_knots) < 2 or len(target_y_knots) < 2:
            raise ValueError("Grid mapping requires at least two target columns and rows.")

        raw_x_knots = np.asarray(
            [float(np.mean(raw_gaze[targets[:, 0] == target_x, 0])) for target_x in target_x_knots],
            dtype=float,
        )
        raw_y_knots = np.asarray(
            [float(np.mean(raw_gaze[targets[:, 1] == target_y, 1])) for target_y in target_y_knots],
            dtype=float,
        )
        if np.any(np.diff(raw_x_knots) <= 1e-6) or np.any(np.diff(raw_y_knots) <= 1e-6):
            raise ValueError("Grid mapping requires monotonic raw calibration columns and rows.")

        self.raw_x_knots = raw_x_knots
        self.target_x_knots = target_x_knots
        self.raw_y_knots = raw_y_knots
        self.target_y_knots = target_y_knots


class LinearRegressionMappingModel(RegressionMappingModel):
    def __init__(self) -> None:
        super().__init__(model_type=MappingModelType.LINEAR)


class AffineMappingModel(RegressionMappingModel):
    def __init__(self) -> None:
        super().__init__(model_type=MappingModelType.AFFINE)


class GridMappingModel(RegressionMappingModel):
    def __init__(self) -> None:
        super().__init__(model_type=MappingModelType.GRID)


class RidgeRegressionMappingModel(RegressionMappingModel):
    def __init__(self, ridge_alpha: float = 1.0) -> None:
        super().__init__(model_type=MappingModelType.RIDGE, ridge_alpha=ridge_alpha)


class PolynomialRegressionMappingModel(RegressionMappingModel):
    def __init__(self, polynomial_degree: int = 2) -> None:
        super().__init__(
            model_type=MappingModelType.POLYNOMIAL,
            polynomial_degree=polynomial_degree,
        )


class InverseDistanceMappingModel(RegressionMappingModel):
    def __init__(self, idw_power: float = 2.0) -> None:
        super().__init__(
            model_type=MappingModelType.IDW,
            idw_power=idw_power,
        )


def create_mapping_model(
    model_type: MappingModelType | str,
    polynomial_degree: int = 2,
    ridge_alpha: float = 1.0,
    idw_power: float = 2.0,
) -> RegressionMappingModel:
    resolved_type = model_type if isinstance(model_type, MappingModelType) else MappingModelType(str(model_type))
    return RegressionMappingModel(
        model_type=resolved_type,
        polynomial_degree=polynomial_degree,
        ridge_alpha=ridge_alpha,
        idw_power=idw_power,
    )
