"""Simulation-driven design optimization with explicit constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

from .parameters import get_parameter, replace_parameters

Direction = Literal["minimize", "maximize"]
Method = Literal["slsqp", "differential_evolution"]


@dataclass(frozen=True)
class DesignVariable:
    """A bounded scalar variable located in a nested parameter tree."""

    path: str
    lower: float
    upper: float
    logarithmic: bool = False

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("path cannot be empty")
        if not np.isfinite((self.lower, self.upper)).all() or self.upper <= self.lower:
            raise ValueError("design-variable bounds must be finite and increasing")
        if self.logarithmic and self.lower <= 0.0:
            raise ValueError("logarithmic variables require positive bounds")


@dataclass(frozen=True)
class Objective:
    metric: str
    direction: Direction = "minimize"
    weight: float = 1.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("direction must be 'minimize' or 'maximize'")
        if (
            not np.isfinite((self.weight, self.scale)).all()
            or self.weight < 0.0
            or self.scale <= 0.0
        ):
            raise ValueError("objective weight must be non-negative and scale positive")


@dataclass(frozen=True)
class MetricConstraint:
    metric: str
    lower: float | None = None
    upper: float | None = None
    penalty: float = 1.0e4

    def __post_init__(self) -> None:
        if self.lower is None and self.upper is None:
            raise ValueError("constraint requires at least one bound")
        if (
            self.lower is not None
            and self.upper is not None
            and self.upper < self.lower
        ):
            raise ValueError("constraint upper bound must be >= lower bound")
        supplied = tuple(
            value for value in (self.lower, self.upper) if value is not None
        )
        if not np.isfinite(supplied).all():
            raise ValueError("constraint bounds must be finite")
        if not np.isfinite(self.penalty) or self.penalty <= 0.0:
            raise ValueError("constraint penalty must be positive")


@dataclass(frozen=True)
class DesignRecord:
    variables: Mapping[str, float]
    metrics: Mapping[str, float]
    objective: float
    feasible: bool


@dataclass(frozen=True)
class DesignResult:
    parameters: object
    variables: Mapping[str, float]
    metrics: Mapping[str, float]
    objective: float
    feasible: bool
    success: bool
    message: str
    evaluations: tuple[DesignRecord, ...]


class DesignOptimizer:
    """Optimize any deterministic simulator returning scalar metrics."""

    def __init__(
        self,
        base_parameters: object,
        variables: Sequence[DesignVariable],
        objectives: Sequence[Objective],
        evaluator: Callable[[object], Mapping[str, float]],
        constraints: Sequence[MetricConstraint] = (),
    ) -> None:
        if not variables:
            raise ValueError("at least one design variable is required")
        if not objectives:
            raise ValueError("at least one objective is required")
        self.base_parameters = base_parameters
        self.variables = tuple(variables)
        self.objectives = tuple(objectives)
        self.evaluator = evaluator
        self.constraints = tuple(constraints)
        self._records: list[DesignRecord] = []

    def optimize(
        self,
        *,
        method: Method = "slsqp",
        maximum_iterations: int = 100,
        population_size: int = 15,
        tolerance: float = 1.0e-7,
        seed: int | None = None,
    ) -> DesignResult:
        """Run a bounded local or global optimization."""

        integer_values = (maximum_iterations, population_size)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            for value in integer_values
        ):
            raise TypeError("iteration and population counts must be integers")
        if maximum_iterations <= 0 or population_size <= 0 or tolerance <= 0.0:
            raise ValueError("iteration, population, and tolerance values must be positive")
        self._records = []
        bounds = [self._internal_bounds(variable) for variable in self.variables]
        initial = np.asarray(
            [
                self._to_internal(
                    variable,
                    float(get_parameter(self.base_parameters, variable.path)),
                )
                for variable in self.variables
            ],
            dtype=float,
        )
        initial = np.clip(initial, [item[0] for item in bounds], [item[1] for item in bounds])

        if method == "slsqp":
            try:
                from scipy.optimize import minimize
            except ImportError as exc:
                raise ImportError("design optimization requires SciPy") from exc
            raw = minimize(
                self._evaluate_vector,
                initial,
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": maximum_iterations, "ftol": tolerance},
            )
        elif method == "differential_evolution":
            try:
                from scipy.optimize import differential_evolution
            except ImportError as exc:
                raise ImportError("design optimization requires SciPy") from exc
            raw = differential_evolution(
                self._evaluate_vector,
                bounds,
                maxiter=maximum_iterations,
                popsize=population_size,
                tol=tolerance,
                seed=seed,
                polish=True,
            )
        else:
            raise ValueError(f"unsupported optimization method: {method}")

        variables = self._decode(np.asarray(raw.x, dtype=float))
        parameters = replace_parameters(self.base_parameters, variables)
        metrics = _finite_metrics(self.evaluator(parameters))
        objective, feasible = self._score(metrics)
        return DesignResult(
            parameters=parameters,
            variables=variables,
            metrics=metrics,
            objective=objective,
            feasible=feasible,
            success=bool(raw.success),
            message=str(raw.message),
            evaluations=tuple(self._records),
        )

    def _evaluate_vector(self, vector: np.ndarray) -> float:
        variables = self._decode(vector)
        parameters = replace_parameters(self.base_parameters, variables)
        metrics = _finite_metrics(self.evaluator(parameters))
        score, feasible = self._score(metrics)
        self._records.append(
            DesignRecord(
                variables=dict(variables),
                metrics=dict(metrics),
                objective=score,
                feasible=feasible,
            )
        )
        return score

    def _score(self, metrics: Mapping[str, float]) -> tuple[float, bool]:
        missing = {
            item.metric for item in (*self.objectives, *self.constraints)
        } - set(metrics)
        if missing:
            raise ValueError(f"evaluator omitted required metrics: {sorted(missing)}")
        score = 0.0
        for objective in self.objectives:
            sign = 1.0 if objective.direction == "minimize" else -1.0
            score += (
                objective.weight * sign * metrics[objective.metric] / objective.scale
            )
        feasible = True
        for constraint in self.constraints:
            value = metrics[constraint.metric]
            if constraint.lower is not None and value < constraint.lower:
                violation = constraint.lower - value
                score += constraint.penalty * violation * violation
                feasible = False
            if constraint.upper is not None and value > constraint.upper:
                violation = value - constraint.upper
                score += constraint.penalty * violation * violation
                feasible = False
        return float(score), feasible

    def _decode(self, vector: np.ndarray) -> dict[str, float]:
        if vector.shape != (len(self.variables),):
            raise ValueError("optimizer vector has the wrong shape")
        return {
            variable.path: self._from_internal(variable, float(value))
            for variable, value in zip(self.variables, vector, strict=True)
        }

    @staticmethod
    def _internal_bounds(variable: DesignVariable) -> tuple[float, float]:
        if variable.logarithmic:
            return float(np.log(variable.lower)), float(np.log(variable.upper))
        return variable.lower, variable.upper

    @staticmethod
    def _to_internal(variable: DesignVariable, value: float) -> float:
        if variable.logarithmic:
            if value <= 0.0:
                raise ValueError(f"{variable.path} must be positive")
            return float(np.log(value))
        return value

    @staticmethod
    def _from_internal(variable: DesignVariable, value: float) -> float:
        return float(np.exp(value)) if variable.logarithmic else value


def pareto_front(
    records: Sequence[DesignRecord],
    objectives: Sequence[Objective],
) -> tuple[DesignRecord, ...]:
    """Return feasible non-dominated records for the requested objectives."""

    if not objectives:
        raise ValueError("at least one objective is required")
    feasible = [record for record in records if record.feasible]
    front: list[DesignRecord] = []
    for candidate in feasible:
        dominated = False
        candidate_values = _directed_values(candidate, objectives)
        for other in feasible:
            if other is candidate:
                continue
            other_values = _directed_values(other, objectives)
            no_worse = all(
                left <= right for left, right in zip(other_values, candidate_values, strict=True)
            )
            strictly_better = any(
                left < right for left, right in zip(other_values, candidate_values, strict=True)
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return tuple(front)


def _directed_values(
    record: DesignRecord, objectives: Sequence[Objective]
) -> tuple[float, ...]:
    return tuple(
        record.metrics[objective.metric]
        * (1.0 if objective.direction == "minimize" else -1.0)
        for objective in objectives
    )


def _finite_metrics(values: Mapping[str, float]) -> dict[str, float]:
    result = {str(name): float(value) for name, value in values.items()}
    if not result or not np.isfinite(tuple(result.values())).all():
        raise ValueError("evaluator must return finite scalar metrics")
    return result
