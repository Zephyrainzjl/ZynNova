from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .numerics import (
    bic_score,
    make_folds,
    ordinary_least_squares,
    prediction,
    r2_score,
    standard_scale,
    stratified_bootstrap_indices,
)
from .schema import DiscoveredLaw, Observation


@dataclass(frozen=True, slots=True)
class _Term:
    name: str
    values: np.ndarray
    center: float
    scale: float


class SymbolicLawMiner:
    """Small, auditable symbolic search over standardized polymer descriptors."""

    def __init__(
        self,
        *,
        max_terms: int = 4,
        max_base_features: int = 12,
        maximum_pair_terms: int = 36,
        min_bic_improvement: float = 1.0,
        bootstrap_repeats: int = 256,
        random_seed: int = 42,
    ) -> None:
        if max_terms < 1 or max_base_features < 1:
            raise ValueError("symbolic search sizes must be positive")
        if bootstrap_repeats < 20:
            raise ValueError("bootstrap_repeats must be at least 20")
        self.max_terms = int(max_terms)
        self.max_base_features = int(max_base_features)
        self.maximum_pair_terms = int(maximum_pair_terms)
        self.min_bic_improvement = float(min_bic_improvement)
        self.bootstrap_repeats = int(bootstrap_repeats)
        self.random_seed = int(random_seed)

    def discover(
        self,
        observations: Sequence[Observation],
        target: str,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> DiscoveredLaw:
        rows = tuple(
            observation
            for observation in observations
            if target in observation.targets
            and np.isfinite(float(observation.targets[target]))
        )
        if len(rows) < 12:
            raise ValueError("at least 12 target rows are required")
        if feature_names is None:
            names = sorted(
                {
                    name
                    for observation in rows
                    for name in observation.explanatory_values
                }
            )
        else:
            names = [str(name) for name in feature_names]

        target_raw = np.asarray([observation.targets[target] for observation in rows])
        if float(np.std(target_raw)) < 1.0e-12:
            raise ValueError(f"target {target!r} is constant")
        target_z_matrix, _target_center, _target_scale = standard_scale(
            target_raw[:, None]
        )
        target_z = target_z_matrix[:, 0]
        raw_columns = []
        retained = []
        for name in names:
            column = np.asarray(
                [
                    observation.explanatory_values.get(name, float("nan"))
                    for observation in rows
                ],
                dtype=float,
            )
            if np.isfinite(column).sum() < max(8, int(0.6 * len(rows))):
                continue
            column = np.where(np.isfinite(column), column, np.nanmedian(column))
            if float(np.std(column)) < 1.0e-12:
                continue
            raw_columns.append(column)
            retained.append(name)
        if not raw_columns:
            raise ValueError("no supported non-constant features")
        raw = np.column_stack(raw_columns)
        base_z, _center, _scale = standard_scale(raw)
        correlations = np.asarray(
            [abs(_correlation(base_z[:, index], target_z)) for index in range(base_z.shape[1])]
        )
        selected_base = np.argsort(-correlations)[: self.max_base_features]
        terms = self._candidate_terms(
            raw[:, selected_base],
            base_z[:, selected_base],
            tuple(retained[index] for index in selected_base),
            target_z,
        )
        selected_indices, intercept, coefficients, bic = self._forward_select(
            terms,
            target_z,
        )
        selected_terms = tuple(terms[index] for index in selected_indices)
        selected_matrix = np.column_stack([term.values for term in selected_terms])
        train_prediction = prediction(intercept, coefficients, selected_matrix)
        validation_r2 = self._validation_r2(
            selected_matrix,
            target_z,
            tuple(observation.environment for observation in rows),
        )
        coefficient_ci = self._bootstrap_ci(
            selected_matrix,
            target_z,
            tuple(observation.environment for observation in rows),
        )
        expression = _expression(intercept, selected_terms, coefficients, target)
        return DiscoveredLaw(
            target=target,
            expression=expression,
            intercept=float(intercept),
            terms=tuple(_normalized_term(term) for term in selected_terms),
            coefficients=tuple(float(value) for value in coefficients),
            coefficient_ci=coefficient_ci,
            train_r2=r2_score(target_z, train_prediction),
            validation_r2=validation_r2,
            bic=float(bic),
            sample_count=len(rows),
            environments=tuple(sorted({observation.environment for observation in rows})),
        )

    def _candidate_terms(
        self,
        raw: np.ndarray,
        base_z: np.ndarray,
        names: tuple[str, ...],
        target_z: np.ndarray,
    ) -> tuple[_Term, ...]:
        candidates: list[tuple[str, np.ndarray]] = []
        for index, name in enumerate(names):
            candidates.append((f"z({name})", base_z[:, index]))
            candidates.append((f"z({name})^2", base_z[:, index] ** 2))
            candidates.append(
                (
                    f"log1p_abs_z({name})",
                    np.log1p(np.abs(base_z[:, index])),
                )
            )
        pair_candidates = []
        correlations = np.asarray(
            [abs(_correlation(base_z[:, index], target_z)) for index in range(len(names))]
        )
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                score = correlations[left] * correlations[right]
                pair_candidates.append((score, left, right))
        pair_candidates.sort(reverse=True)
        for _score, left, right in pair_candidates[: self.maximum_pair_terms]:
            candidates.append(
                (
                    f"z({names[left]})*z({names[right]})",
                    base_z[:, left] * base_z[:, right],
                )
            )

        lookup = {name: raw[:, index] for index, name in enumerate(names)}
        candidates.extend(_physical_templates(lookup))
        result = []
        seen: set[str] = set()
        for name, values in candidates:
            if name in seen or np.any(~np.isfinite(values)) or float(np.std(values)) < 1e-12:
                continue
            scaled, center, scale = standard_scale(np.asarray(values)[:, None])
            result.append(
                _Term(
                    name=name,
                    values=scaled[:, 0],
                    center=float(center[0]),
                    scale=float(scale[0]),
                )
            )
            seen.add(name)
        return tuple(result)

    def _forward_select(
        self,
        terms: Sequence[_Term],
        target: np.ndarray,
    ) -> tuple[tuple[int, ...], float, np.ndarray, float]:
        if not terms:
            raise ValueError("symbolic candidate library is empty")
        selected: list[int] = []
        remaining = set(range(len(terms)))
        current_bic = float("inf")
        current_intercept = float(np.mean(target))
        current_coefficients = np.empty(0, dtype=float)
        for _ in range(self.max_terms):
            best = None
            for index in sorted(remaining):
                trial = [*selected, index]
                matrix = np.column_stack([terms[position].values for position in trial])
                if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
                    continue
                intercept, coefficients = ordinary_least_squares(matrix, target)
                score = bic_score(
                    target,
                    prediction(intercept, coefficients, matrix),
                    parameter_count=len(trial) + 1,
                )
                if best is None or score < best[0]:
                    best = (score, index, intercept, coefficients)
            if best is None:
                break
            score, index, intercept, coefficients = best
            if selected and current_bic - score < self.min_bic_improvement:
                break
            selected.append(index)
            remaining.remove(index)
            current_bic = float(score)
            current_intercept = float(intercept)
            current_coefficients = np.asarray(coefficients, dtype=float)
        if not selected:
            raise ValueError("no symbolic term improved the intercept-only model")
        return (
            tuple(selected),
            current_intercept,
            current_coefficients,
            current_bic,
        )

    def _validation_r2(
        self,
        matrix: np.ndarray,
        target: np.ndarray,
        environments: tuple[str, ...],
    ) -> float:
        unique = sorted(set(environments))
        predictions = np.full_like(target, np.nan, dtype=float)
        environment_array = np.asarray(environments, dtype=object)
        enough_per_environment = all(
            np.count_nonzero(environment_array == item) >= 3 for item in unique
        )
        if len(unique) > 1 and enough_per_environment:
            for environment in unique:
                valid = np.flatnonzero(environment_array == environment)
                train = np.flatnonzero(environment_array != environment)
                if len(train) <= matrix.shape[1] + 1:
                    continue
                intercept, coefficients = ordinary_least_squares(matrix[train], target[train])
                predictions[valid] = prediction(intercept, coefficients, matrix[valid])
        else:
            folds = make_folds(
                len(target),
                min(5, len(target)),
                seed=self.random_seed,
            )
            all_indices = np.arange(len(target))
            for valid in folds:
                train = np.setdiff1d(all_indices, valid)
                intercept, coefficients = ordinary_least_squares(matrix[train], target[train])
                predictions[valid] = prediction(intercept, coefficients, matrix[valid])
        finite = np.isfinite(predictions)
        if np.count_nonzero(finite) < max(4, matrix.shape[1] + 2):
            return float("nan")
        return r2_score(target[finite], predictions[finite])

    def _bootstrap_ci(
        self,
        matrix: np.ndarray,
        target: np.ndarray,
        environments: tuple[str, ...],
    ) -> tuple[tuple[float, float], ...]:
        rng = np.random.default_rng(self.random_seed)
        values = []
        for _ in range(self.bootstrap_repeats):
            indices = stratified_bootstrap_indices(environments, rng=rng)
            _intercept, coefficients = ordinary_least_squares(
                matrix[indices],
                target[indices],
                ridge=1.0e-8,
            )
            values.append(coefficients)
        bootstrap = np.stack(values)
        return tuple(
            (
                float(np.quantile(bootstrap[:, index], 0.025)),
                float(np.quantile(bootstrap[:, index], 0.975)),
            )
            for index in range(matrix.shape[1])
        )


def _physical_templates(
    values: dict[str, np.ndarray],
) -> list[tuple[str, np.ndarray]]:
    result = []

    def add(required: Sequence[str], name: str, function: Callable[..., np.ndarray]) -> None:
        if all(key in values for key in required):
            result.append((name, function(*(values[key] for key in required))))

    add(
        (
            "breakdown_strength_MV_m",
            "maximum_polarization_C_m2",
            "remanent_polarization_C_m2",
        ),
        "Eb*(Pm-Pr)",
        lambda eb, pm, pr: eb * (pm - pr),
    )
    add(
        ("dielectric_constant", "breakdown_strength_MV_m"),
        "epsilon_r*Eb^2",
        lambda epsilon, eb: epsilon * eb**2,
    )
    add(
        ("bond_configurational_entropy_R", "barrier_standard_deviation_eV"),
        "S_bond*barrier_std",
        lambda entropy, barrier: entropy * barrier,
    )
    add(
        ("crosslink_density_fraction", "phase_energy_gap_eV"),
        "crosslink/(abs(phase_gap)+eps)",
        lambda density, gap: density / (np.abs(gap) + 1.0e-6),
    )
    add(
        ("bandgap_eV", "cohesive_energy_density_J_cm3"),
        "bandgap*sqrt(CED)",
        lambda gap, ced: gap * np.sqrt(np.clip(ced, 0.0, None)),
    )
    return result


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) < 1.0e-12 or float(np.std(right)) < 1.0e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _expression(
    intercept: float,
    terms: Sequence[_Term],
    coefficients: Sequence[float],
    target: str,
) -> str:
    pieces = [f"{intercept:.6g}"]
    for coefficient, term in zip(coefficients, terms, strict=True):
        sign = "+" if coefficient >= 0 else "-"
        pieces.append(
            f" {sign} {abs(float(coefficient)):.6g}*{_normalized_term(term)}"
        )
    return f"z({target}) = " + "".join(pieces)


def _normalized_term(term: _Term) -> str:
    return f"({term.name} - {term.center:.6g})/{term.scale:.6g}"


__all__ = ["SymbolicLawMiner"]
