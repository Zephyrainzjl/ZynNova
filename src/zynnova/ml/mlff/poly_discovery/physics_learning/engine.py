from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import Observation
from .backends import SymbolicProblem, create_backend
from .config import PhysicsLearningConfig
from .dimensions import (
    VariableSpec,
    buckingham_pi_groups,
    resolve_variable_specs,
)
from .interaction import HessianInteractionDecomposer, QuadraticInteractionOracle
from .neural import PhysicsKANOracle
from .schema import BackendStatus, PhysicsEquation, PhysicsLearningReport


class PhysicsLearningEngine:
    """Discover compact, dimension-aware laws from polymer observations.

    The engine separates three jobs: a predictive oracle estimates nonlinear
    interactions, symbolic backends distil candidate equations, and a held-out
    environment or fold ranks those candidates. Optional heavyweight backends
    are isolated behind lazy adapters so importing this module only needs NumPy.
    """

    def __init__(self, config: PhysicsLearningConfig | None = None) -> None:
        self.config = config or PhysicsLearningConfig(enabled=True)
        self.config.__post_init__()

    def discover(
        self,
        observations: Sequence[Observation],
        target: str,
        *,
        feature_names: Sequence[str] | None = None,
        variable_specs: Mapping[str, VariableSpec] | None = None,
    ) -> PhysicsLearningReport:
        problem, preparation = self._prepare_problem(
            observations,
            target,
            feature_names=feature_names,
            variable_specs=variable_specs,
        )
        warnings: list[str] = []
        interaction, oracle_diagnostics = self._discover_interactions(
            problem,
            warnings,
        )
        equations, statuses = self._run_symbolic_backends(problem, warnings)
        unit_rejected = tuple(
            equation.equation_id
            for equation in equations
            if self.config.reject_unit_inconsistent
            and equation.unit_consistent is False
        )
        ranked = self._rank_equations(equations)
        if equations and not ranked:
            warnings.append(
                "All symbolic equations were rejected by the configured "
                "physical-unit consistency policy."
            )
        groups = buckingham_pi_groups(
            tuple(
                spec
                for spec in (*problem.feature_specs, problem.target_spec)
                if spec.dimension_known
            )
        )
        pareto = _pareto_equation_ids(ranked)
        return PhysicsLearningReport(
            target=target,
            sample_count=len(problem.target_values),
            feature_names=problem.feature_names,
            feature_units={
                spec.name: (
                    spec.unit if spec.dimension_known else "unknown"
                )
                for spec in problem.feature_specs
            },
            target_unit=(
                problem.target_spec.unit
                if problem.target_spec.dimension_known
                else "unknown"
            ),
            equations=ranked,
            best_equation_id=ranked[0].equation_id if ranked else None,
            interaction_decomposition=interaction,
            dimensionless_groups=groups,
            backend_status=statuses,
            diagnostics={
                **preparation,
                "oracle": oracle_diagnostics,
                "pareto_equation_ids": list(pareto),
                "ranking": {
                    "validation_r2": 1.0,
                    "complexity_penalty": self.config.complexity_penalty,
                    "unit_consistency_bonus": self.config.unit_consistency_bonus,
                    "reject_unit_inconsistent": (
                        self.config.reject_unit_inconsistent
                    ),
                    "unit_rejected_equation_ids": list(unit_rejected),
                    "environment_consistency_weight": (
                        self.config.environment_consistency_weight
                    ),
                },
                "causal_status": (
                    "predictive physical-law candidates; interventions or "
                    "independent simulations are required for causal claims"
                ),
            },
            warnings=tuple(warnings),
            schema_version="1.0",
        )

    def _prepare_problem(
        self,
        observations: Sequence[Observation],
        target: str,
        *,
        feature_names: Sequence[str] | None,
        variable_specs: Mapping[str, VariableSpec] | None,
    ) -> tuple[SymbolicProblem, dict[str, Any]]:
        rows = tuple(
            observation
            for observation in observations
            if target in observation.targets
            and np.isfinite(float(observation.targets[target]))
        )
        if len(rows) < self.config.min_samples:
            raise ValueError(
                f"target {target!r} has {len(rows)} finite rows; "
                f"at least {self.config.min_samples} are required"
            )
        target_values = np.asarray(
            [observation.targets[target] for observation in rows],
            dtype=float,
        )
        if float(np.std(target_values)) < 1.0e-12:
            raise ValueError(f"target {target!r} is constant")

        candidates = (
            sorted(
                {
                    name
                    for observation in rows
                    for name, value in observation.explanatory_values.items()
                    if np.isfinite(float(value))
                }
            )
            if feature_names is None
            else [str(name) for name in feature_names]
        )
        required_support = max(
            8,
            int(
                math.ceil(
                    self.config.minimum_feature_support_fraction * len(rows)
                )
            ),
        )
        columns: list[np.ndarray] = []
        names: list[str] = []
        supports: list[int] = []
        correlations: list[float] = []
        imputed_counts: list[int] = []
        for name in candidates:
            column = np.asarray(
                [
                    observation.explanatory_values.get(name, float("nan"))
                    for observation in rows
                ],
                dtype=float,
            )
            finite = np.isfinite(column)
            support = int(np.count_nonzero(finite))
            if support < required_support:
                continue
            median = float(np.median(column[finite]))
            completed = np.where(finite, column, median)
            if float(np.std(completed)) < 1.0e-12:
                continue
            columns.append(completed)
            names.append(name)
            supports.append(support)
            imputed_counts.append(len(rows) - support)
            correlations.append(abs(_correlation(completed, target_values)))
        if not columns:
            raise ValueError("no supported non-constant explanatory feature")

        order = sorted(
            range(len(names)),
            key=lambda index: (-correlations[index], names[index]),
        )[: self.config.max_features]
        retained_names = tuple(names[index] for index in order)
        matrix = np.column_stack([columns[index] for index in order])
        retained_support = [supports[index] for index in order]
        retained_imputed = [imputed_counts[index] for index in order]
        environments = tuple(observation.environment for observation in rows)
        train, validation, split_method = _validation_split(
            environments,
            len(rows),
            validation_fraction=self.config.validation_fraction,
            feature_count=len(retained_names),
            random_seed=self.config.random_seed,
        )
        weights, uncertainty_support = _sample_weights(rows, target)
        feature_specs = resolve_variable_specs(
            retained_names,
            overrides=variable_specs,
        )
        target_spec = resolve_variable_specs(
            (target,),
            overrides=variable_specs,
        )[0]
        workspace = self._workspace(target)
        problem = SymbolicProblem(
            observations=rows,
            target=target,
            feature_names=retained_names,
            matrix=matrix,
            target_values=target_values,
            environments=environments,
            feature_specs=feature_specs,
            target_spec=target_spec,
            train_indices=train,
            validation_indices=validation,
            sample_weights=weights,
            workspace=workspace,
        )
        preparation = {
            "validation_strategy": split_method,
            "train_samples": int(len(train)),
            "validation_samples": int(len(validation)),
            "validation_environments": sorted(
                {environments[index] for index in validation}
            ),
            "feature_support": dict(
                zip(retained_names, retained_support, strict=True)
            ),
            "imputed_feature_values": dict(
                zip(retained_names, retained_imputed, strict=True)
            ),
            "target_uncertainty_support": uncertainty_support,
            "workspace": str(workspace),
        }
        return problem, preparation

    def _discover_interactions(
        self,
        problem: SymbolicProblem,
        warnings: list[str],
    ):
        if self.config.oracle_backend == "none":
            return None, {"backend": "none", "executed": False}

        train = problem.train_indices
        center = np.mean(problem.matrix[train], axis=0)
        scale = np.std(problem.matrix[train], axis=0)
        scale[scale < 1.0e-12] = 1.0
        standardized = (problem.matrix - center) / scale
        oracle_name = self.config.oracle_backend
        diagnostics: dict[str, Any]
        predictor = None
        hessian_provider = None
        validation_r2 = float("nan")

        use_kan = self.config.oracle_backend in {"auto", "kan"}
        if use_kan and PhysicsKANOracle.available():
            monotonic = {
                problem.feature_names.index(name): int(direction)
                for name, direction in self.config.monotonic_constraints.items()
                if name in problem.feature_names
            }
            oracle = PhysicsKANOracle(
                hidden_width=self.config.kan_hidden_width,
                grid_size=self.config.kan_grid_size,
                layers=self.config.kan_layers,
                epochs=self.config.kan_epochs,
                learning_rate=self.config.kan_learning_rate,
                weight_decay=self.config.kan_weight_decay,
                sparsity_weight=self.config.kan_sparsity_weight,
                patience=self.config.kan_patience,
                device=self.config.kan_device,
                dtype=self.config.kan_dtype,
                random_seed=self.config.random_seed,
                monotonic_constraints=monotonic,
                monotonicity_weight=self.config.monotonicity_weight,
            )
            try:
                oracle.fit(
                    problem.matrix,
                    problem.target_values,
                    train_indices=train,
                    validation_indices=problem.validation_indices,
                )
                standardized = oracle.standardize(problem.matrix)
                predictor = oracle.predict_standardized
                hessian_provider = oracle.mixed_hessian_standardized
                oracle_name = "sparse-rbf-kan"
                validation_r2 = (
                    float("nan")
                    if oracle.diagnostics_ is None
                    else oracle.diagnostics_.validation_r2
                )
                diagnostics = {
                    "backend": oracle_name,
                    "executed": True,
                    "derivatives": "exact PyTorch autograd Hessian",
                    **(
                        {}
                        if oracle.diagnostics_ is None
                        else asdict(oracle.diagnostics_)
                    ),
                }
            except Exception as exc:
                if self.config.oracle_backend == "kan":
                    raise
                warnings.append(
                    "KAN interaction oracle failed; using the quadratic "
                    f"fallback: {type(exc).__name__}: {exc}"
                )
                predictor = None
                hessian_provider = None
        elif self.config.oracle_backend == "kan":
            raise ImportError(
                "oracle_backend='kan' requires PyTorch; install "
                "zynnova[physics-discovery]"
            )

        if predictor is None:
            oracle = QuadraticInteractionOracle().fit(
                standardized,
                problem.target_values,
                train_indices=train,
                validation_indices=problem.validation_indices,
            )
            predictor = oracle.predict
            oracle_name = "quadratic-hessian-fallback"
            validation_r2 = oracle.validation_r2_
            diagnostics = {
                "backend": oracle_name,
                "executed": True,
                "derivatives": "central finite-difference Hessian",
                "validation_r2": validation_r2,
                "feature_center": center.tolist(),
                "feature_scale": scale.tolist(),
            }
            if self.config.oracle_backend == "auto" and not PhysicsKANOracle.available():
                warnings.append(
                    "PyTorch is unavailable; nonlinear interactions use the "
                    "auditable quadratic Hessian fallback."
                )

        decomposition = HessianInteractionDecomposer(
            step=self.config.interaction_step,
            sample_count=self.config.interaction_sample_count,
            relative_threshold=self.config.interaction_relative_threshold,
            score_quantile=self.config.interaction_quantile,
            random_seed=self.config.random_seed,
        ).decompose(
            standardized,
            problem.feature_names,
            predictor,
            oracle_name=oracle_name,
            oracle_validation_r2=validation_r2,
            hessian_provider=hessian_provider,
        )
        return decomposition, diagnostics

    def _run_symbolic_backends(
        self,
        problem: SymbolicProblem,
        warnings: list[str],
    ) -> tuple[tuple[PhysicsEquation, ...], tuple[BackendStatus, ...]]:
        equations: list[PhysicsEquation] = []
        statuses: list[BackendStatus] = []
        for backend_name in self.config.symbolic_backends:
            backend = create_backend(backend_name)
            status = backend.status(self.config)
            if not status.available:
                statuses.append(status)
                warnings.append(f"{backend_name} was skipped: {status.detail}")
                if self.config.strict_backend_failures:
                    raise RuntimeError(status.detail)
                continue
            try:
                fitted = backend.fit(problem, self.config)
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                statuses.append(
                    replace(
                        status,
                        executed=True,
                        detail=f"execution failed: {detail}",
                    )
                )
                warnings.append(f"{backend_name} failed: {detail}")
                if self.config.strict_backend_failures:
                    raise
            else:
                returned = (
                    (fitted,)
                    if isinstance(fitted, PhysicsEquation)
                    else tuple(fitted)
                )
                if not returned:
                    detail = "backend returned no equations"
                    statuses.append(
                        replace(
                            status,
                            executed=True,
                            detail=detail,
                        )
                    )
                    warnings.append(f"{backend_name} failed: {detail}")
                    if self.config.strict_backend_failures:
                        raise RuntimeError(detail)
                    continue
                equations.extend(returned)
                best_validation = max(
                    (
                        equation.validation_r2
                        for equation in returned
                        if np.isfinite(equation.validation_r2)
                    ),
                    default=float("nan"),
                )
                statuses.append(
                    replace(
                        status,
                        executed=True,
                        detail=(
                            f"completed; {len(returned)} equation(s); "
                            f"best validation R2={best_validation:.4g}"
                        ),
                    )
                )
        return tuple(equations), tuple(statuses)

    def _rank_equations(
        self,
        equations: Sequence[PhysicsEquation],
    ) -> tuple[PhysicsEquation, ...]:
        ranked = []
        seen: set[str] = set()
        for equation in equations:
            if (
                self.config.reject_unit_inconsistent
                and equation.unit_consistent is False
            ):
                continue
            canonical = re.sub(r"\s+", "", equation.expression).lower()
            if canonical in seen:
                continue
            seen.add(canonical)
            validation = (
                float(equation.validation_r2)
                if np.isfinite(equation.validation_r2)
                else -1.0
            )
            score = (
                validation
                - self.config.complexity_penalty
                * math.log1p(max(equation.complexity, 0))
                + self.config.environment_consistency_weight
                * float(np.clip(equation.environment_consistency, 0.0, 1.0))
                + (
                    self.config.unit_consistency_bonus
                    if equation.unit_consistent is True
                    else 0.0
                )
            )
            ranked.append(replace(equation, ranking_score=float(score)))
        ranked.sort(
            key=lambda item: (
                -item.ranking_score,
                item.complexity,
                item.backend,
                item.expression,
            )
        )
        return tuple(ranked)

    def _workspace(self, target: str) -> Path:
        root = (
            Path(self.config.workspace_root).expanduser()
            if self.config.workspace_root is not None
            else Path.cwd() / "zynnova-physics-learning"
        )
        safe_target = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(target)).strip("-")
        destination = root.resolve() / f"{safe_target or 'target'}-seed{self.config.random_seed}"
        destination.mkdir(parents=True, exist_ok=True)
        return destination


def discover_physical_laws(
    observations: Sequence[Observation],
    target: str,
    *,
    feature_names: Sequence[str] | None = None,
    variable_specs: Mapping[str, VariableSpec] | None = None,
    config: PhysicsLearningConfig | None = None,
) -> PhysicsLearningReport:
    """Functional entry point for neural-symbolic physical-law discovery."""

    resolved = config or PhysicsLearningConfig(enabled=True)
    return PhysicsLearningEngine(resolved).discover(
        observations,
        target,
        feature_names=feature_names,
        variable_specs=variable_specs,
    )


def _validation_split(
    environments: Sequence[str],
    sample_count: int,
    *,
    validation_fraction: float,
    feature_count: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    environment_array = np.asarray(tuple(environments), dtype=object)
    desired = max(3, int(round(validation_fraction * sample_count)))
    minimum_train = max(12, feature_count + 3)
    candidates = []
    for environment in sorted(set(environments)):
        valid = np.flatnonzero(environment_array == environment)
        train = np.flatnonzero(environment_array != environment)
        if len(valid) >= 3 and len(train) >= minimum_train:
            candidates.append(
                (abs(len(valid) - desired), str(environment), train, valid)
            )
    if candidates:
        _distance, environment, train, valid = min(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        return train, valid, f"held-out environment: {environment}"

    rng = np.random.default_rng(random_seed)
    permutation = rng.permutation(sample_count)
    valid_count = min(
        max(desired, 3),
        sample_count - minimum_train,
    )
    if valid_count < 3:
        raise ValueError("too few samples for a non-empty validation split")
    valid = np.sort(permutation[:valid_count])
    train = np.sort(permutation[valid_count:])
    return train, valid, "seeded random holdout"


def _sample_weights(
    observations: Sequence[Observation],
    target: str,
) -> tuple[np.ndarray, int]:
    uncertainty = np.asarray(
        [
            observation.uncertainty.get(target, float("nan"))
            for observation in observations
        ],
        dtype=float,
    )
    supported = np.isfinite(uncertainty) & (uncertainty > 0.0)
    support = int(np.count_nonzero(supported))
    if support < 3:
        return np.ones(len(observations), dtype=float), support
    fill = float(np.median(uncertainty[supported]))
    completed = np.where(supported, uncertainty, fill)
    weights = 1.0 / np.maximum(completed**2, 1.0e-24)
    low, high = np.quantile(weights, (0.05, 0.95))
    weights = np.clip(weights, low, high)
    return weights / np.mean(weights), support


def _pareto_equation_ids(
    equations: Sequence[PhysicsEquation],
) -> tuple[str, ...]:
    pareto = []
    for candidate in equations:
        candidate_r2 = (
            candidate.validation_r2
            if np.isfinite(candidate.validation_r2)
            else -float("inf")
        )
        dominated = False
        for other in equations:
            if other is candidate:
                continue
            other_r2 = (
                other.validation_r2
                if np.isfinite(other.validation_r2)
                else -float("inf")
            )
            at_least_as_good = (
                other_r2 >= candidate_r2
                and other.complexity <= candidate.complexity
            )
            strictly_better = (
                other_r2 > candidate_r2
                or other.complexity < candidate.complexity
            )
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(candidate.equation_id)
    return tuple(pareto)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) < 1.0e-12 or float(np.std(right)) < 1.0e-12:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


__all__ = [
    "PhysicsLearningEngine",
    "discover_physical_laws",
]
