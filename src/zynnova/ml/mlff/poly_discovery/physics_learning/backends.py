from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import re
import sys
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..schema import Observation
from ..symbolic import SymbolicLawMiner
from .config import PhysicsLearningConfig
from .dimensions import DIMENSIONLESS, VariableSpec
from .schema import BackendStatus, PhysicsEquation


@dataclass(frozen=True, slots=True)
class SymbolicProblem:
    observations: tuple[Observation, ...]
    target: str
    feature_names: tuple[str, ...]
    matrix: np.ndarray
    target_values: np.ndarray
    environments: tuple[str, ...]
    feature_specs: tuple[VariableSpec, ...]
    target_spec: VariableSpec
    train_indices: np.ndarray
    validation_indices: np.ndarray
    sample_weights: np.ndarray
    workspace: Path


class SymbolicBackend(Protocol):
    name: str

    def status(self, config: PhysicsLearningConfig) -> BackendStatus: ...

    def fit(
        self,
        problem: SymbolicProblem,
        config: PhysicsLearningConfig,
    ) -> PhysicsEquation | Sequence[PhysicsEquation]: ...


class NativeSymbolicBackend:
    name = "native"

    def status(self, config: PhysicsLearningConfig) -> BackendStatus:
        return BackendStatus(
            name=self.name,
            available=True,
            executed=False,
            detail=(
                "NumPy auditable standardized symbolic search; always available"
            ),
            version="1",
        )

    def fit(
        self,
        problem: SymbolicProblem,
        config: PhysicsLearningConfig,
    ) -> PhysicsEquation:
        law = SymbolicLawMiner(
            max_terms=config.native_max_terms,
            max_base_features=config.native_max_base_features,
            maximum_pair_terms=config.native_pair_terms,
            bootstrap_repeats=config.bootstrap_repeats,
            random_seed=config.random_seed,
        ).discover(
            problem.observations,
            problem.target,
            feature_names=problem.feature_names,
        )
        stable = [
            low > 0.0 or high < 0.0
            for low, high in law.coefficient_ci
        ]
        stability = float(np.mean(stable)) if stable else 0.0
        train_rmse = _rmse_from_r2(law.train_r2)
        validation_rmse = _rmse_from_r2(law.validation_r2)
        return PhysicsEquation(
            equation_id="native-0",
            target=problem.target,
            expression=law.expression,
            backend=self.name,
            feature_names=problem.feature_names,
            train_r2=float(law.train_r2),
            validation_r2=float(law.validation_r2),
            train_rmse=train_rmse,
            validation_rmse=validation_rmse,
            complexity=len(law.terms) + 1,
            unit_consistent=True,
            normalized=True,
            stability=stability,
            environment_consistency=_bounded_score(law.validation_r2),
            metadata={
                "terms": list(law.terms),
                "coefficients": list(law.coefficients),
                "coefficient_ci": [list(interval) for interval in law.coefficient_ci],
                "bic": law.bic,
                "normalization": "z-score variables and z-score target",
            },
            caveats=(
                "The equation is dimensionally valid only in its explicitly "
                "reported dimensionless z-score coordinates.",
            ),
        )


class PSEBackend:
    """Adapter for the official 2026 PSE/PSRN parallel enumerator."""

    name = "pse"

    def status(self, config: PhysicsLearningConfig) -> BackendStatus:
        available = (
            importlib.util.find_spec("psrn") is not None
            and importlib.util.find_spec("torch") is not None
        )
        return BackendStatus(
            name=self.name,
            available=available,
            executed=False,
            detail=(
                "GPU/CPU parallel symbolic enumeration with common-subtree reuse"
                if available
                else (
                    "Install psrn>=0.1.2 on Python <=3.12 to enable the "
                    "official PSE/PSRN backend"
                )
            ),
            version=_version("psrn") if available else None,
        )

    def fit(
        self,
        problem: SymbolicProblem,
        config: PhysicsLearningConfig,
    ) -> Sequence[PhysicsEquation]:
        try:
            import torch
            from psrn import PSRN_Regressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("PSE/PSRN is not installed") from exc

        if config.pse_device == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = str(config.pse_device)
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"PSE requested {device_name!r}, but CUDA is unavailable"
            )
        device = torch.device(device_name)
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)

        safe_names = _safe_feature_names(problem.feature_names)
        operators = list(config.pse_operators)
        input_count = max(
            len(safe_names),
            (
                len(safe_names) + (2 if config.pse_use_constants else 0)
                if config.pse_inputs is None
                else int(config.pse_inputs)
            ),
        )
        work = problem.workspace / "pse"
        work.mkdir(parents=True, exist_ok=True)
        dr_mask_dir = (
            Path(config.pse_dr_mask_dir).expanduser().resolve()
            if config.pse_dr_mask_dir is not None
            else work / "dr_mask"
        )
        if config.pse_use_dr_mask and not dr_mask_dir.is_dir():
            raise FileNotFoundError(
                "pse_use_dr_mask=True requires an existing PSE mask directory; "
                f"not found: {dr_mask_dir}"
            )
        regressor = PSRN_Regressor(
            variables=list(safe_names),
            use_const=config.pse_use_constants,
            n_symbol_layers=config.pse_symbol_layers,
            device=device,
            use_dr_mask=config.pse_use_dr_mask,
            dr_mask_dir=str(dr_mask_dir),
            token_generator_config={
                "base": {
                    "has_const": config.pse_use_constants,
                    "tokens": operators,
                }
            },
            stage_config={
                "default": {
                    "operators": operators,
                    "time_limit": config.pse_time_limit_seconds,
                    "n_psrn_inputs": input_count,
                    "n_sample_variables": min(3, len(safe_names)),
                },
                "stages": [{}],
            },
        )
        train = problem.train_indices
        valid = problem.validation_indices
        start = time.perf_counter()
        with _pushd(work):
            fit_flag, _pareto = regressor.fit(
                problem.matrix[train],
                problem.target_values[train, None],
                n_down_sample=min(config.pse_downsample, len(train)),
                use_threshold=False,
                threshold=1.0e-20,
                probe=None,
                prun_const=True,
                prun_ndigit=8,
                real_time_display=False,
                top_k=config.pse_top_k,
            )
            table = list(regressor.display_expr_table(sort_by="mse"))
        runtime = time.perf_counter() - start
        positions = _pareto_positions(
            len(table),
            config.pse_pareto_candidates,
        )
        units_known = (
            all(spec.dimension_known for spec in problem.feature_specs)
            and problem.target_spec.dimension_known
        )
        result = []
        candidate_errors = []
        for position in positions:
            try:
                expression, reward, search_loss, complexity = _pse_row(
                    table[position]
                )
                symbolic = _normalize_symbolic_expression(expression)
                train_prediction = _sympy_predict(
                    symbolic,
                    safe_names,
                    problem.matrix[train],
                )
                valid_prediction = _sympy_predict(
                    symbolic,
                    safe_names,
                    problem.matrix[valid],
                )
            except Exception as exc:
                candidate_errors.append(
                    f"row {position}: {type(exc).__name__}: {exc}"
                )
                continue
            display = _restore_names(
                symbolic,
                safe_names,
                problem.feature_names,
            )
            train_r2 = _r2(
                problem.target_values[train],
                train_prediction,
            )
            validation_r2 = _r2(
                problem.target_values[valid],
                valid_prediction,
            )
            unit_consistent = (
                _sympy_dimension_consistent(
                    symbolic,
                    safe_names,
                    problem.feature_specs,
                    problem.target_spec,
                )
                if units_known
                else None
            )
            result.append(
                PhysicsEquation(
                    equation_id=f"pse-{position}",
                    target=problem.target,
                    expression=f"{problem.target} = {display}",
                    backend=self.name,
                    feature_names=problem.feature_names,
                    train_r2=train_r2,
                    validation_r2=validation_r2,
                    train_rmse=_rmse(
                        problem.target_values[train],
                        train_prediction,
                    ),
                    validation_rmse=_rmse(
                        problem.target_values[valid],
                        valid_prediction,
                    ),
                    complexity=int(complexity),
                    unit_consistent=unit_consistent,
                    normalized=False,
                    stability=_bounded_score(validation_r2),
                    environment_consistency=_environment_consistency(
                        problem.target_values[valid],
                        valid_prediction,
                        tuple(
                            problem.environments[index]
                            for index in valid
                        ),
                    ),
                    metadata={
                        "pareto_index": position,
                        "pse_reward": _optional_float(reward),
                        "pse_search_loss": _optional_float(search_loss),
                        "fit_flag": str(fit_flag),
                        "operators": operators,
                        "symbol_layers": config.pse_symbol_layers,
                        "input_slots": input_count,
                        "device": str(device),
                        "use_dr_mask": config.pse_use_dr_mask,
                        "dr_mask_dir": (
                            str(dr_mask_dir)
                            if config.pse_use_dr_mask
                            else None
                        ),
                        "runtime_seconds": runtime,
                        "workspace": str(work),
                        "official_package": "psrn",
                    },
                    caveats=(
                        "PSE does not enforce SI dimensions during enumeration; "
                        "unit consistency is checked post hoc when all dimensions "
                        "are known.",
                    ),
                )
            )
        if not result:
            detail = "; ".join(candidate_errors[:3])
            raise RuntimeError(
                "PSE returned no numerically evaluable expression"
                + (f": {detail}" if detail else "")
            )
        return tuple(result)


class PySRBackend:
    name = "pysr"

    def status(self, config: PhysicsLearningConfig) -> BackendStatus:
        available = importlib.util.find_spec("pysr") is not None
        return BackendStatus(
            name=self.name,
            available=available,
            executed=False,
            detail=(
                "High-performance evolutionary Pareto symbolic regression with "
                "DynamicQuantities unit constraints"
                if available
                else "Install zynnova[physics-symbolic] to enable PySR"
            ),
            version=_version("pysr") if available else None,
        )

    def fit(
        self,
        problem: SymbolicProblem,
        config: PhysicsLearningConfig,
    ) -> Sequence[PhysicsEquation]:
        try:
            from pysr import PySRRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("PySR is not installed") from exc

        safe_names = _safe_feature_names(problem.feature_names)
        kwargs: dict[str, Any] = {
            "model_selection": "best",
            "niterations": config.pysr_iterations,
            "populations": config.pysr_populations,
            "population_size": config.pysr_population_size,
            "maxsize": config.pysr_max_size,
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": ["square", "cube", "sqrt", "exp", "log"],
            "constraints": {
                "/": (-1, 10),
                "square": 10,
                "cube": 8,
                "sqrt": 8,
                "exp": 6,
                "log": 6,
            },
            "nested_constraints": {
                "exp": {"exp": 0, "log": 0},
                "log": {"exp": 0, "log": 0},
            },
            "dimensional_constraint_penalty": 1.0e5,
            "dimensionless_constants_only": (
                config.pysr_dimensionless_constants_only
            ),
            "complexity_of_constants": 2,
            "precision": 64,
            "progress": False,
            "verbosity": 0,
            "random_state": config.random_seed,
        }
        if config.pysr_timeout_seconds is not None:
            kwargs["timeout_in_seconds"] = config.pysr_timeout_seconds
        output_directory = problem.workspace / "pysr"
        output_directory.mkdir(parents=True, exist_ok=True)
        kwargs["output_directory"] = str(output_directory)
        kwargs["run_id"] = "zynnova-physics"
        model = PySRRegressor(**kwargs)
        train = problem.train_indices
        fit_kwargs: dict[str, Any] = {
            "weights": problem.sample_weights[train],
            "variable_names": list(safe_names),
        }
        units_known = (
            all(spec.dimension_known for spec in problem.feature_specs)
            and problem.target_spec.dimension_known
        )
        if units_known:
            fit_kwargs["X_units"] = [
                spec.dimension.pysr_unit()
                for spec in problem.feature_specs
            ]
            fit_kwargs["y_units"] = problem.target_spec.dimension.pysr_unit()
        model.fit(
            problem.matrix[train],
            problem.target_values[train],
            **fit_kwargs,
        )
        valid = problem.validation_indices
        equation_table = model.equations_
        positions = _pareto_positions(
            len(equation_table),
            config.pysr_pareto_candidates,
        )
        result = []
        for position in positions:
            symbolic = model.sympy(index=position)
            train_prediction = np.asarray(
                model.predict(problem.matrix[train], index=position),
                dtype=float,
            ).reshape(-1)
            valid_prediction = np.asarray(
                model.predict(problem.matrix[valid], index=position),
                dtype=float,
            ).reshape(-1)
            expression = _restore_names(
                str(symbolic),
                safe_names,
                problem.feature_names,
            )
            row = equation_table.iloc[position]
            complexity = int(
                _mapping_value(
                    row,
                    "complexity",
                    _expression_complexity(expression),
                )
            )
            train_r2 = _r2(
                problem.target_values[train],
                train_prediction,
            )
            validation_r2 = _r2(
                problem.target_values[valid],
                valid_prediction,
            )
            unit_consistent = (
                _sympy_dimension_consistent(
                    symbolic,
                    safe_names,
                    problem.feature_specs,
                    problem.target_spec,
                )
                if units_known
                and config.pysr_dimensionless_constants_only
                else None
            )
            result.append(
                PhysicsEquation(
                    equation_id=f"pysr-{position}",
                    target=problem.target,
                    expression=f"{problem.target} = {expression}",
                    backend=self.name,
                    feature_names=problem.feature_names,
                    train_r2=train_r2,
                    validation_r2=validation_r2,
                    train_rmse=_rmse(
                        problem.target_values[train],
                        train_prediction,
                    ),
                    validation_rmse=_rmse(
                        problem.target_values[valid],
                        valid_prediction,
                    ),
                    complexity=complexity,
                    unit_consistent=unit_consistent,
                    normalized=False,
                    stability=_bounded_score(validation_r2),
                    environment_consistency=_environment_consistency(
                        problem.target_values[valid],
                        valid_prediction,
                        tuple(
                            problem.environments[index]
                            for index in valid
                        ),
                    ),
                    metadata={
                        "pareto_index": position,
                        "search_loss": _optional_float(
                            _mapping_value(row, "loss", None)
                        ),
                        "search_score": _optional_float(
                            _mapping_value(row, "score", None)
                        ),
                        "unit_engine": (
                            "DynamicQuantities.jl"
                            if units_known
                            else "disabled: one or more dimensions are unknown"
                        ),
                        "output_directory": str(output_directory),
                    },
                    caveats=(
                        ()
                        if units_known
                        else (
                            "At least one variable dimension is unknown, so "
                            "this PySR search was not dimension constrained.",
                        )
                    ),
                )
            )
        if not result:
            raise RuntimeError("PySR returned an empty Pareto table")
        return tuple(result)


class PhySOBackend:
    name = "physo"

    def status(self, config: PhysicsLearningConfig) -> BackendStatus:
        available = importlib.util.find_spec("physo") is not None
        return BackendStatus(
            name=self.name,
            available=available,
            executed=False,
            detail=(
                "Unit-guided deep-reinforcement symbolic optimization"
                if available
                else "Install zynnova[physics-symbolic] to enable PhySO"
            ),
            version=_version("physo") if available else None,
        )

    def fit(
        self,
        problem: SymbolicProblem,
        config: PhysicsLearningConfig,
    ) -> PhysicsEquation:
        try:
            import physo
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("PhySO is not installed") from exc

        safe_names = _safe_feature_names(problem.feature_names)
        train = problem.train_indices
        valid = problem.validation_indices
        feature_scale = _reference_scales(
            problem.matrix[train],
            problem.feature_specs,
        )
        target_scale = (
            float(problem.target_spec.reference_scale)
            if problem.target_spec.reference_scale is not None
            else max(
                float(np.median(np.abs(problem.target_values[train]))),
                1.0e-12,
            )
        )
        scaled_matrix = problem.matrix / feature_scale
        scaled_target = problem.target_values / target_scale
        work = problem.workspace / "physo"
        work.mkdir(parents=True, exist_ok=True)
        run_config = _physo_run_config(physo)
        units_known = (
            all(spec.dimension_known for spec in problem.feature_specs)
            and problem.target_spec.dimension_known
        )
        unit_kwargs: dict[str, Any] = {}
        if units_known:
            unit_kwargs = {
                "X_units": [
                    spec.dimension.physo_vector()
                    for spec in problem.feature_specs
                ],
                "y_units": problem.target_spec.dimension.physo_vector(),
                "fixed_consts_units": [[0.0] * 7],
            }
        with _pushd(work):
            expression, _logs = physo.SR(
                scaled_matrix[train].T,
                scaled_target[train],
                y_weights=problem.sample_weights[train],
                X_names=list(safe_names),
                y_name=_safe_name(problem.target),
                fixed_consts=[1.0],
                op_names=[
                    "mul",
                    "add",
                    "sub",
                    "div",
                    "inv",
                    "n2",
                    "sqrt",
                    "neg",
                    "exp",
                    "log",
                ],
                run_config=run_config,
                parallel_mode=config.physo_parallel,
                epochs=config.physo_epochs,
                device=config.physo_device,
                **unit_kwargs,
            )
        symbolic = expression.get_infix_sympy(evaluate_consts=True)
        if isinstance(symbolic, (list, tuple)):
            symbolic = symbolic[0]
        train_prediction_scaled = _sympy_predict(
            symbolic,
            safe_names,
            scaled_matrix[train],
        )
        valid_prediction_scaled = _sympy_predict(
            symbolic,
            safe_names,
            scaled_matrix[valid],
        )
        train_prediction = target_scale * train_prediction_scaled
        valid_prediction = target_scale * valid_prediction_scaled
        display_expression = _restore_names(
            str(symbolic),
            safe_names,
            problem.feature_names,
        )
        train_r2 = _r2(problem.target_values[train], train_prediction)
        validation_r2 = _r2(problem.target_values[valid], valid_prediction)
        return PhysicsEquation(
            equation_id="physo-0",
            target=problem.target,
            expression=(
                f"{problem.target}/{target_scale:.8g} = {display_expression}"
            ),
            backend=self.name,
            feature_names=problem.feature_names,
            train_r2=train_r2,
            validation_r2=validation_r2,
            train_rmse=_rmse(problem.target_values[train], train_prediction),
            validation_rmse=_rmse(
                problem.target_values[valid],
                valid_prediction,
            ),
            complexity=_expression_complexity(str(symbolic)),
            unit_consistent=True if units_known else None,
            normalized=True,
            stability=_bounded_score(validation_r2),
            environment_consistency=_environment_consistency(
                problem.target_values[valid],
                valid_prediction,
                tuple(problem.environments[index] for index in valid),
            ),
            metadata={
                "feature_reference_scales": dict(
                    zip(
                        problem.feature_names,
                        feature_scale.tolist(),
                        strict=True,
                    )
                ),
                "target_reference_scale": target_scale,
                "unit_engine": (
                    "PhySO dimensional analysis"
                    if units_known
                    else "disabled: one or more dimensions are unknown"
                ),
                "workspace": str(work),
            },
            caveats=(
                "The displayed variables are divided by the reference scales "
                "stored in metadata.",
                *(
                    ()
                    if units_known
                    else (
                        "At least one variable dimension is unknown, so this "
                        "PhySO search was not dimension constrained.",
                    )
                ),
            ),
        )


class PhyE2EBackend:
    name = "phye2e"

    def status(self, config: PhysicsLearningConfig) -> BackendStatus:
        repository = (
            None
            if config.phye2e_repository is None
            else Path(config.phye2e_repository).expanduser()
        )
        checkpoint = (
            None
            if config.phye2e_checkpoint is None
            else Path(config.phye2e_checkpoint).expanduser()
        )
        available = bool(
            repository is not None
            and checkpoint is not None
            and (repository / "PhysicsRegression.py").is_file()
            and checkpoint.is_file()
        )
        return BackendStatus(
            name=self.name,
            available=available,
            executed=False,
            detail=(
                "Official PhyE2E checkpoint with derivative divide-and-conquer, "
                "Transformer generation, MCTS, and genetic programming"
                if available
                else (
                    "Set phye2e_repository and phye2e_checkpoint to the official "
                    "PhysicsRegression release"
                )
            ),
            version="official-v1.0.0" if available else None,
        )

    def fit(
        self,
        problem: SymbolicProblem,
        config: PhysicsLearningConfig,
    ) -> PhysicsEquation:
        status = self.status(config)
        if not status.available:
            raise RuntimeError(status.detail)
        repository = Path(config.phye2e_repository).expanduser().resolve()
        checkpoint = Path(config.phye2e_checkpoint).expanduser().resolve()
        module = _load_phye2e(repository)
        model = module.PhyReg(
            str(checkpoint),
            max_len=config.phye2e_max_points,
            device=config.phye2e_device,
        )
        train = problem.train_indices
        valid = problem.validation_indices
        units = [
            *(
                (
                    spec.dimension.phye2e_unit()
                    if spec.dimension_known
                    else None
                )
                for spec in problem.feature_specs
            ),
            (
                problem.target_spec.dimension.phye2e_unit()
                if problem.target_spec.dimension_known
                else None
            ),
        ]
        complete_units = all(unit is not None for unit in units)
        work = problem.workspace / "phye2e"
        work.mkdir(parents=True, exist_ok=True)
        with _pushd(work):
            model.fit(
                problem.matrix[train],
                problem.target_values[train],
                units=units,
                use_Divide=config.phye2e_use_divide,
                use_MCTS=config.phye2e_use_mcts,
                use_GP=config.phye2e_use_gp,
                oracle_epoch=config.phye2e_oracle_epochs,
                verbose=False,
            )
        generations = (
            getattr(model, "best_gens_refined", None)
            or getattr(model, "best_gens_gp", None)
            or getattr(model, "best_gens_mcts", None)
            or getattr(model, "best_gens", None)
        )
        if not generations:
            raise RuntimeError("PhyE2E did not return an expression")
        expression = _phye2e_expression(generations[0]["predicted_tree"])
        safe_names = tuple(f"x_{index}" for index in range(len(problem.feature_names)))
        train_prediction = _sympy_predict(
            expression,
            safe_names,
            problem.matrix[train],
        )
        valid_prediction = _sympy_predict(
            expression,
            safe_names,
            problem.matrix[valid],
        )
        display = _restore_names(
            expression,
            safe_names,
            problem.feature_names,
        )
        train_r2 = _r2(problem.target_values[train], train_prediction)
        validation_r2 = _r2(problem.target_values[valid], valid_prediction)
        return PhysicsEquation(
            equation_id="phye2e-0",
            target=problem.target,
            expression=f"{problem.target} = {display}",
            backend=self.name,
            feature_names=problem.feature_names,
            train_r2=train_r2,
            validation_r2=validation_r2,
            train_rmse=_rmse(problem.target_values[train], train_prediction),
            validation_rmse=_rmse(
                problem.target_values[valid],
                valid_prediction,
            ),
            complexity=_expression_complexity(expression),
            unit_consistent=True if complete_units else None,
            normalized=False,
            stability=_bounded_score(validation_r2),
            environment_consistency=_environment_consistency(
                problem.target_values[valid],
                valid_prediction,
                tuple(problem.environments[index] for index in valid),
            ),
            metadata={
                "divide_and_conquer": config.phye2e_use_divide,
                "mcts": config.phye2e_use_mcts,
                "genetic_programming": config.phye2e_use_gp,
                "official_repository": str(repository),
                "checkpoint": str(checkpoint),
                "workspace": str(work),
                "unit_hints_complete": complete_units,
            },
            caveats=(
                ()
                if complete_units
                else (
                    "PhyE2E received partial or no unit hints because at least "
                    "one variable dimension is unknown or unsupported.",
                )
            ),
        )


BACKENDS: dict[str, type] = {
    "native": NativeSymbolicBackend,
    "pse": PSEBackend,
    "pysr": PySRBackend,
    "physo": PhySOBackend,
    "phye2e": PhyE2EBackend,
}


def create_backend(name: str) -> SymbolicBackend:
    try:
        backend_type = BACKENDS[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"unknown symbolic backend: {name}") from exc
    return backend_type()


def _load_phye2e(repository: Path):
    source = repository / "PhysicsRegression.py"
    module_name = "_zynnova_phye2e_official"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load PhyE2E from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(repository))
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        try:
            sys.path.remove(str(repository))
        except ValueError:
            pass
    return module


def _phye2e_expression(tree: Any) -> str:
    expression = str(tree)
    for source, target in {
        "add": "+",
        "mul": "*",
        "sub": "-",
        "pow": "**",
        "inv": "1/",
        "neg": "-",
    }.items():
        expression = expression.replace(source, target)
    return expression


def _pse_row(row: Any) -> tuple[str, Any, Any, int]:
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
        raise TypeError(f"unexpected PSE result row: {type(row).__name__}")
    if len(row) < 4:
        raise ValueError("PSE result row must contain expression, reward, loss, complexity")
    return str(row[0]), row[1], row[2], int(row[3])


def _normalize_symbolic_expression(expression: str) -> str:
    result = str(expression).strip().replace("^", "**")
    result = re.sub(r"\bln\s*\(", "log(", result)
    return result


def _physo_run_config(physo):
    try:
        return physo.config.config1.config1
    except AttributeError:
        return None


def _sympy_predict(
    expression: Any,
    names: Sequence[str],
    matrix: np.ndarray,
) -> np.ndarray:
    try:
        import sympy
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("SymPy is required to evaluate symbolic expressions") from exc
    symbolic = (
        sympy.sympify(expression)
        if isinstance(expression, str)
        else expression
    )
    variables = [sympy.Symbol(name) for name in names]
    function = sympy.lambdify(variables, symbolic, modules="numpy")
    values = function(*np.asarray(matrix, dtype=float).T)
    result = np.asarray(values, dtype=float)
    if result.ndim == 0:
        result = np.full(len(matrix), float(result), dtype=float)
    return result.reshape(-1)


def _safe_feature_names(names: Sequence[str]) -> tuple[str, ...]:
    result = []
    used: set[str] = set()
    for index, name in enumerate(names):
        candidate = _safe_name(name)
        if not candidate or candidate in used:
            candidate = f"x_{index}"
        used.add(candidate)
        result.append(candidate)
    return tuple(result)


def _safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if not value or value[0].isdigit():
        value = f"x_{value}"
    return value


def _restore_names(
    expression: str,
    safe_names: Sequence[str],
    original_names: Sequence[str],
) -> str:
    result = str(expression)
    pairs = sorted(
        zip(safe_names, original_names, strict=True),
        key=lambda item: -len(item[0]),
    )
    for safe, original in pairs:
        result = re.sub(rf"\b{re.escape(safe)}\b", str(original), result)
    return result


def _pareto_positions(
    equation_count: int,
    maximum_candidates: int,
) -> tuple[int, ...]:
    if equation_count <= 0:
        return ()
    retained = min(int(maximum_candidates), int(equation_count))
    if retained == equation_count:
        return tuple(range(equation_count))
    values = np.linspace(0, equation_count - 1, retained)
    return tuple(sorted({int(round(value)) for value in values}))


class _DimensionMismatch(ValueError):
    pass


def _sympy_dimension_consistent(
    expression: Any,
    names: Sequence[str],
    specs: Sequence[VariableSpec],
    target_spec: VariableSpec,
) -> bool | None:
    try:
        import sympy
    except ImportError:  # pragma: no cover - PySR itself normally installs it
        return None
    symbolic = sympy.sympify(expression)
    dimensions = {
        str(name): spec.dimension
        for name, spec in zip(names, specs, strict=True)
    }

    def evaluate(node):
        if node.is_Number:
            return DIMENSIONLESS
        if node.is_Symbol:
            return dimensions.get(str(node))
        if node.is_Add:
            child_dimensions = [evaluate(child) for child in node.args]
            if any(value is None for value in child_dimensions):
                return None
            reference = child_dimensions[0]
            if any(
                not reference.close_to(value)
                for value in child_dimensions[1:]
            ):
                raise _DimensionMismatch
            return reference
        if node.is_Mul:
            result = DIMENSIONLESS
            for child in node.args:
                child_dimension = evaluate(child)
                if child_dimension is None:
                    return None
                result = result * child_dimension
            return result
        if node.is_Pow:
            base, exponent = node.args
            exponent_dimension = evaluate(exponent)
            if (
                exponent_dimension is None
                or not exponent_dimension.is_dimensionless
                or not exponent.is_number
            ):
                raise _DimensionMismatch
            base_dimension = evaluate(base)
            if base_dimension is None:
                return None
            return base_dimension ** float(exponent)
        if node.func in {
            sympy.exp,
            sympy.log,
            sympy.sin,
            sympy.cos,
            sympy.tan,
            sympy.asin,
            sympy.acos,
            sympy.atan,
        }:
            argument = evaluate(node.args[0])
            if argument is None:
                return None
            if not argument.is_dimensionless:
                raise _DimensionMismatch
            return DIMENSIONLESS
        if node.func is sympy.Abs:
            return evaluate(node.args[0])
        if node.func is sympy.sign:
            return DIMENSIONLESS
        return None

    try:
        result = evaluate(symbolic)
    except _DimensionMismatch:
        return False
    return (
        None
        if result is None
        else result.close_to(target_spec.dimension)
    )


def _reference_scales(
    matrix: np.ndarray,
    specs: Sequence[VariableSpec] | None = None,
) -> np.ndarray:
    scales = np.median(np.abs(np.asarray(matrix, dtype=float)), axis=0)
    scales[scales < 1.0e-12] = 1.0
    if specs is not None:
        if len(specs) != len(scales):
            raise ValueError("variable specs do not match the feature matrix")
        for index, spec in enumerate(specs):
            if spec.reference_scale is not None:
                scales[index] = float(spec.reference_scale)
    return scales


def _expression_complexity(expression: str) -> int:
    tokens = re.findall(
        r"[A-Za-z_]\w*|[-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?|[+\-*/^()]",
        str(expression),
    )
    return max(len(tokens), 1)


def _mapping_value(value: Any, key: str, default: Any) -> Any:
    if hasattr(value, "get"):
        return value.get(key, default)
    try:
        return value[key]
    except (KeyError, IndexError, TypeError):
        return default


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(predicted)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    observed = observed[finite]
    predicted = predicted[finite]
    residual = float(np.sum((observed - predicted) ** 2))
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    return float("nan") if total < 1.0e-15 else 1.0 - residual / total


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(predicted)
    if not np.any(finite):
        return float("nan")
    return float(np.sqrt(np.mean((observed[finite] - predicted[finite]) ** 2)))


def _rmse_from_r2(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(np.sqrt(max(1.0 - value, 0.0)))


def _bounded_score(value: float) -> float:
    return 0.0 if not np.isfinite(value) else float(np.clip(value, 0.0, 1.0))


def _environment_consistency(
    observed: np.ndarray,
    predicted: np.ndarray,
    environments: Sequence[str],
) -> float:
    scores = []
    environment_array = np.asarray(environments, dtype=object)
    for environment in sorted(set(environments)):
        indices = np.flatnonzero(environment_array == environment)
        if len(indices) < 3:
            continue
        score = _r2(observed[indices], predicted[indices])
        if np.isfinite(score):
            scores.append(float(np.clip(score, 0.0, 1.0)))
    return float(np.mean(scores)) if scores else _bounded_score(_r2(observed, predicted))


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


__all__ = [
    "BACKENDS",
    "NativeSymbolicBackend",
    "PSEBackend",
    "PhyE2EBackend",
    "PhySOBackend",
    "PySRBackend",
    "SymbolicBackend",
    "SymbolicProblem",
    "create_backend",
]
