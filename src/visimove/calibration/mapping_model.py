from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class MappingModelType(str, Enum):
    AFFINE = "affine"
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
    ) -> None:
        self.model_type = model_type
        self.polynomial_degree = polynomial_degree
        self.ridge_alpha = ridge_alpha
        self.coefficients: np.ndarray | None = None

    def fit(self, raw_gaze: list[tuple[float, float]], targets: list[tuple[float, float]]) -> None:
        if len(raw_gaze) != len(targets):
            raise ValueError("raw_gaze and targets must contain the same number of samples.")
        if len(raw_gaze) < 2:
            raise ValueError("At least two samples are required to fit a mapping model.")

        raw_array = np.asarray(raw_gaze, dtype=float)
        target_array = np.asarray(targets, dtype=float)

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
        if self.coefficients is None:
            raise RuntimeError("Mapping model must be fitted before predict().")
        features = self._features(np.asarray([raw_gaze], dtype=float))
        prediction = features @ self.coefficients
        return MappingPrediction(x=float(prediction[0, 0]), y=float(prediction[0, 1]))

    def to_parameters(self) -> dict[str, Any]:
        if self.coefficients is None:
            raise RuntimeError("Mapping model must be fitted before serialization.")
        return {
            "model_type": self.model_type.value,
            "polynomial_degree": self.polynomial_degree,
            "ridge_alpha": self.ridge_alpha,
            "coefficients": self.coefficients.tolist(),
        }

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any]) -> "RegressionMappingModel":
        model = cls(
            model_type=MappingModelType(parameters["model_type"]),
            polynomial_degree=int(parameters.get("polynomial_degree", 2)),
            ridge_alpha=float(parameters.get("ridge_alpha", 1.0)),
        )
        model.coefficients = np.asarray(parameters["coefficients"], dtype=float)
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


class LinearRegressionMappingModel(RegressionMappingModel):
    def __init__(self) -> None:
        super().__init__(model_type=MappingModelType.LINEAR)


class AffineMappingModel(RegressionMappingModel):
    def __init__(self) -> None:
        super().__init__(model_type=MappingModelType.AFFINE)


class RidgeRegressionMappingModel(RegressionMappingModel):
    def __init__(self, ridge_alpha: float = 1.0) -> None:
        super().__init__(model_type=MappingModelType.RIDGE, ridge_alpha=ridge_alpha)


class PolynomialRegressionMappingModel(RegressionMappingModel):
    def __init__(self, polynomial_degree: int = 2) -> None:
        super().__init__(
            model_type=MappingModelType.POLYNOMIAL,
            polynomial_degree=polynomial_degree,
        )


def create_mapping_model(
    model_type: MappingModelType | str,
    polynomial_degree: int = 2,
    ridge_alpha: float = 1.0,
) -> RegressionMappingModel:
    resolved_type = model_type if isinstance(model_type, MappingModelType) else MappingModelType(str(model_type))
    return RegressionMappingModel(
        model_type=resolved_type,
        polynomial_degree=polynomial_degree,
        ridge_alpha=ridge_alpha,
    )
