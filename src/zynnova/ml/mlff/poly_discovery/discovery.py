from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .config import MechanismDiscoveryConfig
from .numerics import (
    elastic_net_coordinate_descent,
    ordinary_least_squares,
    robust_scale,
    select_elastic_net_alpha,
    standard_scale,
    stratified_bootstrap_indices,
)
from .priors import MechanismPrior, priors_for_target
from .schema import (
    DiscoveryReport,
    EvidenceLevel,
    FeatureEffect,
    MatchedPairEffect,
    MechanismHypothesis,
    MediationResult,
    Observation,
)

if TYPE_CHECKING:
    from .physics_learning import PhysicsLearningConfig, VariableSpec


@dataclass(slots=True)
class _PreparedData:
    observations: tuple[Observation, ...]
    target_raw: np.ndarray
    target_z: np.ndarray
    target_center: float
    target_scale: float
    sample_weight: np.ndarray
    uncertainty_support: int
    base_raw: np.ndarray
    base_z: np.ndarray
    base_names: tuple[str, ...]
    base_center: np.ndarray
    base_scale: np.ndarray
    base_support: np.ndarray
    candidate_matrix: np.ndarray
    candidate_names: tuple[str, ...]
    candidate_center: np.ndarray
    candidate_scale: np.ndarray
    candidate_support: tuple[int, ...]
    interaction_flags: tuple[bool, ...]
    nonlinear_flags: tuple[bool, ...]
    environments: tuple[str, ...]


class MechanismDiscoveryEngine:
    """Discover stable, falsifiable polymer structure-property hypotheses.

    The engine intentionally reports associations unless randomized or explicit
    interventions are present. It combines sparse stability selection, within-
    environment sign checks, matched controls, mediation tests, and a small
    symbolic-law search. This is designed to propose mechanisms and next tests,
    not to turn a correlation into a causal claim.
    """

    def __init__(self, config: MechanismDiscoveryConfig | None = None) -> None:
        self.config = config or MechanismDiscoveryConfig()
        self.config.__post_init__()

    def discover(
        self,
        observations: Sequence[Observation],
        target: str,
        *,
        feature_names: Sequence[str] | None = None,
        control_names: Sequence[str] = (),
        include_symbolic_law: bool = True,
        priors: Sequence[MechanismPrior] | None = None,
        physics_learning_config: PhysicsLearningConfig | None = None,
        variable_specs: Mapping[str, VariableSpec] | None = None,
    ) -> DiscoveryReport:
        prepared = self._prepare(observations, target, feature_names)
        alpha, cv_scores = select_elastic_net_alpha(
            prepared.candidate_matrix,
            prepared.target_z,
            self.config.alpha_grid,
            l1_ratio=self.config.elastic_net_l1_ratio,
            fold_count=self.config.cross_validation_folds,
            environments=prepared.environments,
            sample_weight=prepared.sample_weight,
            seed=self.config.random_seed,
        )
        _intercept, coefficients = elastic_net_coordinate_descent(
            prepared.candidate_matrix,
            prepared.target_z,
            alpha=alpha,
            l1_ratio=self.config.elastic_net_l1_ratio,
            sample_weight=prepared.sample_weight,
        )
        bootstrap = self._bootstrap_coefficients(prepared, alpha)
        environment_fraction = self._environment_sign_fraction(
            prepared,
            alpha,
            coefficients,
        )
        effects = self._effects(
            prepared,
            coefficients,
            bootstrap,
            environment_fraction,
        )

        resolved_priors = tuple(priors) if priors is not None else priors_for_target(target)
        mediations = self._mediation_tests(
            prepared.observations,
            target,
            resolved_priors,
            control_names=control_names,
        )
        matched = self._matched_group_effects(
            prepared.observations,
            target,
            effects,
            control_names=control_names,
        )
        hypotheses = self._hypotheses(
            prepared.observations,
            target,
            effects,
            mediations,
            resolved_priors,
        )
        law = None
        warnings = list(self._warnings(prepared, effects))
        if include_symbolic_law:
            try:
                from .symbolic import SymbolicLawMiner

                law = SymbolicLawMiner(
                    max_terms=self.config.max_symbolic_terms,
                    bootstrap_repeats=self.config.bootstrap_repeats,
                    random_seed=self.config.random_seed,
                ).discover(
                    prepared.observations,
                    target,
                    feature_names=prepared.base_names,
                )
            except ValueError as exc:
                warnings.append(f"Symbolic law was not reported: {exc}")

        physics_learning = None
        if (
            physics_learning_config is not None
            and physics_learning_config.enabled
        ):
            try:
                from .physics_learning import PhysicsLearningEngine

                physics_learning = PhysicsLearningEngine(
                    physics_learning_config
                ).discover(
                    prepared.observations,
                    target,
                    feature_names=prepared.base_names,
                    variable_specs=variable_specs,
                )
            except Exception as exc:
                if physics_learning_config.strict_backend_failures:
                    raise
                warnings.append(
                    "Advanced physics learning was not reported: "
                    f"{type(exc).__name__}: {exc}"
                )

        return DiscoveryReport(
            target=target,
            sample_count=len(prepared.observations),
            feature_names=prepared.base_names,
            effects=effects,
            hypotheses=hypotheses,
            mediations=mediations,
            matched_effects=matched,
            law=law,
            physics_learning=physics_learning,
            environments=tuple(sorted(set(prepared.environments))),
            diagnostics={
                "selected_alpha": alpha,
                "cross_validation_mse": cv_scores,
                "target_center": prepared.target_center,
                "target_scale": prepared.target_scale,
                "feature_center": dict(
                    zip(
                        prepared.base_names,
                        prepared.base_center.tolist(),
                        strict=True,
                    )
                ),
                "feature_scale": dict(
                    zip(
                        prepared.base_names,
                        prepared.base_scale.tolist(),
                        strict=True,
                    )
                ),
                "candidate_count": len(prepared.candidate_names),
                "candidate_term_center": dict(
                    zip(
                        prepared.candidate_names,
                        prepared.candidate_center.tolist(),
                        strict=True,
                    )
                ),
                "candidate_term_scale": dict(
                    zip(
                        prepared.candidate_names,
                        prepared.candidate_scale.tolist(),
                        strict=True,
                    )
                ),
                "bootstrap_repeats": self.config.bootstrap_repeats,
                "target_uncertainty_support": prepared.uncertainty_support,
                "effective_sample_size": _effective_sample_size(
                    prepared.sample_weight
                ),
                "effect_units": (
                    "standard deviations of target per standard deviation of term"
                ),
                "causal_status": (
                    "hypothesis-generating unless explicit intervention evidence is present"
                ),
            },
            warnings=tuple(warnings),
            schema_version="1.1" if physics_learning is not None else "1.0",
        )

    def estimate_matched_effect(
        self,
        observations: Sequence[Observation],
        *,
        exposure: str,
        outcome: str,
        control_names: Sequence[str],
        threshold: float | None = None,
        caliper: float = 2.5,
    ) -> MatchedPairEffect:
        rows = []
        for observation in observations:
            x = observation.get(exposure)
            y = observation.targets.get(outcome)
            controls = [observation.get(name) for name in control_names]
            if x is None or y is None or any(value is None for value in controls):
                continue
            values = [float(x), float(y), *(float(value) for value in controls)]
            if all(np.isfinite(values)):
                rows.append((observation, *values))
        if len(rows) < self.config.min_feature_support:
            raise ValueError("too few complete cases for matched-pair estimation")
        exposure_values = np.asarray([row[1] for row in rows], dtype=float)
        outcome_values = np.asarray([row[2] for row in rows], dtype=float)
        controls = np.asarray([row[3:] for row in rows], dtype=float)
        if threshold is None:
            threshold = 0.0 if np.any(exposure_values == 0.0) else float(
                np.median(exposure_values)
            )
        treated = np.flatnonzero(exposure_values > threshold)
        untreated = np.flatnonzero(exposure_values <= threshold)
        if treated.size < 2 or untreated.size < 2:
            raise ValueError("exposure threshold does not create two supported groups")
        controls_z, _center, _scale = robust_scale(controls)
        differences = []
        distances = []
        for index in treated:
            candidate_distance = np.linalg.norm(
                controls_z[untreated] - controls_z[index],
                axis=1,
            )
            nearest_position = int(np.argmin(candidate_distance))
            distance = float(candidate_distance[nearest_position])
            if distance > caliper:
                continue
            control_index = int(untreated[nearest_position])
            differences.append(outcome_values[index] - outcome_values[control_index])
            distances.append(distance)
        if len(differences) < 2:
            raise ValueError("no matched pairs remained inside the requested caliper")
        difference_array = np.asarray(differences, dtype=float)
        rng = np.random.default_rng(self.config.random_seed)
        boot = np.asarray(
            [
                np.mean(rng.choice(difference_array, size=len(difference_array), replace=True))
                for _ in range(self.config.bootstrap_repeats)
            ],
            dtype=float,
        )
        tail = (1.0 - self.config.confidence_level) / 2.0
        level = _evidence_level(tuple(row[0] for row in rows), (exposure,))
        return MatchedPairEffect(
            exposure=exposure,
            outcome=outcome,
            threshold=float(threshold),
            matched_pairs=len(differences),
            average_difference=float(np.mean(difference_array)),
            ci_low=float(np.quantile(boot, tail)),
            ci_high=float(np.quantile(boot, 1.0 - tail)),
            median_control_distance=float(np.median(distances)),
            evidence_level=level,
            caveat=(
                "Nearest-neighbour matching controls only measured covariates; "
                "unmeasured chemistry, morphology, and process confounding can remain."
            ),
        )

    def _prepare(
        self,
        observations: Sequence[Observation],
        target: str,
        feature_names: Sequence[str] | None,
    ) -> _PreparedData:
        eligible = tuple(
            observation
            for observation in observations
            if target in observation.targets
            and np.isfinite(float(observation.targets[target]))
        )
        if len(eligible) < self.config.min_samples:
            raise ValueError(
                f"target {target!r} has {len(eligible)} finite rows; "
                f"at least {self.config.min_samples} are required"
            )
        if feature_names is None:
            candidates = sorted(
                {
                    name
                    for observation in eligible
                    for name, value in observation.explanatory_values.items()
                    if np.isfinite(float(value))
                }
            )
        else:
            candidates = [str(name) for name in feature_names]

        raw_columns = []
        supports = []
        retained_names = []
        for name in candidates:
            column = np.asarray(
                [
                    observation.explanatory_values.get(name, float("nan"))
                    for observation in eligible
                ],
                dtype=float,
            )
            support = int(np.isfinite(column).sum())
            if support < self.config.min_feature_support:
                continue
            median = float(np.nanmedian(column))
            column = np.where(np.isfinite(column), column, median)
            if float(np.std(column)) < 1.0e-12:
                continue
            raw_columns.append(column)
            supports.append(support)
            retained_names.append(name)
        if not raw_columns:
            raise ValueError("no non-constant feature has enough support")
        base_raw = np.column_stack(raw_columns)
        base_z, base_center, base_scale = robust_scale(base_raw)
        target_raw = np.asarray(
            [observation.targets[target] for observation in eligible],
            dtype=float,
        )
        if float(np.std(target_raw)) < 1.0e-12:
            raise ValueError(f"target {target!r} is constant")
        target_matrix, target_center_array, target_scale_array = standard_scale(
            target_raw[:, None]
        )
        target_z = target_matrix[:, 0]
        target_center = float(target_center_array[0])
        target_scale = float(target_scale_array[0])
        sample_weight, uncertainty_support = _precision_weights(eligible, target)

        candidate_values: list[np.ndarray] = [base_z[:, index] for index in range(base_z.shape[1])]
        candidate_names = list(retained_names)
        candidate_support = list(supports)
        interaction_flags = [False] * len(candidate_names)
        nonlinear_flags = [False] * len(candidate_names)

        if self.config.include_squared_terms:
            for index, name in enumerate(retained_names):
                candidate_values.append(base_z[:, index] ** 2)
                candidate_names.append(f"{name}^2")
                candidate_support.append(supports[index])
                interaction_flags.append(False)
                nonlinear_flags.append(True)

        if self.config.include_interactions and len(retained_names) > 1:
            correlations = np.asarray(
                [
                    abs(_safe_correlation(base_z[:, index], target_z))
                    for index in range(base_z.shape[1])
                ]
            )
            ranked = np.argsort(-correlations)[: self.config.max_interaction_features]
            pairs = []
            for left_position, left in enumerate(ranked):
                for right in ranked[left_position + 1 :]:
                    score = correlations[left] * correlations[right]
                    pairs.append((float(score), int(left), int(right)))
            pairs.sort(reverse=True)
            for _score, left, right in pairs[: self.config.max_interactions]:
                candidate_values.append(base_z[:, left] * base_z[:, right])
                candidate_names.append(f"{retained_names[left]} × {retained_names[right]}")
                candidate_support.append(min(supports[left], supports[right]))
                interaction_flags.append(True)
                nonlinear_flags.append(True)

        raw_candidate_matrix = np.column_stack(candidate_values)
        candidate_matrix, candidate_center, candidate_scale = standard_scale(
            raw_candidate_matrix
        )
        finite_columns = np.std(candidate_matrix, axis=0) > 1.0e-12
        candidate_matrix = candidate_matrix[:, finite_columns]
        candidate_center = candidate_center[finite_columns]
        candidate_scale = candidate_scale[finite_columns]

        def selected(values: Sequence[Any]) -> tuple[Any, ...]:
            return tuple(value for value, keep in zip(values, finite_columns, strict=True) if keep)

        return _PreparedData(
            observations=eligible,
            target_raw=target_raw,
            target_z=target_z,
            target_center=target_center,
            target_scale=target_scale,
            sample_weight=sample_weight,
            uncertainty_support=uncertainty_support,
            base_raw=base_raw,
            base_z=base_z,
            base_names=tuple(retained_names),
            base_center=base_center,
            base_scale=base_scale,
            base_support=np.asarray(supports, dtype=int),
            candidate_matrix=candidate_matrix,
            candidate_names=selected(candidate_names),
            candidate_center=candidate_center,
            candidate_scale=candidate_scale,
            candidate_support=selected(candidate_support),
            interaction_flags=selected(interaction_flags),
            nonlinear_flags=selected(nonlinear_flags),
            environments=tuple(observation.environment for observation in eligible),
        )

    def _bootstrap_coefficients(
        self,
        prepared: _PreparedData,
        alpha: float,
    ) -> np.ndarray:
        rng = np.random.default_rng(self.config.random_seed)
        result = np.zeros(
            (self.config.bootstrap_repeats, prepared.candidate_matrix.shape[1]),
            dtype=float,
        )
        for repeat in range(self.config.bootstrap_repeats):
            indices = stratified_bootstrap_indices(prepared.environments, rng=rng)
            _intercept, result[repeat] = elastic_net_coordinate_descent(
                prepared.candidate_matrix[indices],
                prepared.target_z[indices],
                alpha=alpha,
                l1_ratio=self.config.elastic_net_l1_ratio,
                sample_weight=prepared.sample_weight[indices],
            )
        return result

    def _environment_sign_fraction(
        self,
        prepared: _PreparedData,
        alpha: float,
        full_coefficients: np.ndarray,
    ) -> np.ndarray:
        environments = np.asarray(prepared.environments, dtype=object)
        signs: list[np.ndarray] = []
        for environment in sorted(set(prepared.environments)):
            indices = np.flatnonzero(environments == environment)
            if len(indices) < self.config.min_environment_samples:
                continue
            _intercept, coefficients = elastic_net_coordinate_descent(
                prepared.candidate_matrix[indices],
                prepared.target_z[indices],
                alpha=alpha,
                l1_ratio=self.config.elastic_net_l1_ratio,
                sample_weight=prepared.sample_weight[indices],
            )
            signs.append(np.sign(coefficients))
        if not signs:
            return np.ones_like(full_coefficients)
        sign_matrix = np.stack(signs, axis=0)
        reference = np.sign(full_coefficients)
        fraction = np.mean(sign_matrix == reference[None, :], axis=0)
        fraction[reference == 0] = np.mean(sign_matrix[:, reference == 0] == 0, axis=0)
        return fraction

    def _effects(
        self,
        prepared: _PreparedData,
        coefficients: np.ndarray,
        bootstrap: np.ndarray,
        environment_fraction: np.ndarray,
    ) -> tuple[FeatureEffect, ...]:
        tail = (1.0 - self.config.confidence_level) / 2.0
        effects = []
        for index, name in enumerate(prepared.candidate_names):
            values = bootstrap[:, index]
            selected = np.abs(values) > 1.0e-10
            selection_frequency = float(np.mean(selected))
            nonzero = values[selected]
            if nonzero.size:
                dominant = np.sign(np.median(nonzero))
                sign_consistency = float(np.mean(np.sign(nonzero) == dominant))
            else:
                sign_consistency = 0.0
            non_positive = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
            non_negative = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
            p_value = min(1.0, 2.0 * min(non_positive, non_negative))
            effects.append(
                FeatureEffect(
                    term=name,
                    coefficient=float(coefficients[index]),
                    ci_low=float(np.quantile(values, tail)),
                    ci_high=float(np.quantile(values, 1.0 - tail)),
                    selection_frequency=selection_frequency,
                    sign_consistency=sign_consistency,
                    environment_sign_fraction=float(environment_fraction[index]),
                    support=int(prepared.candidate_support[index]),
                    bootstrap_p_value=float(p_value),
                    is_interaction=prepared.interaction_flags[index],
                    is_nonlinear=prepared.nonlinear_flags[index],
                )
            )
        effects.sort(
            key=lambda effect: (
                effect.selection_frequency,
                effect.sign_consistency,
                abs(effect.coefficient),
            ),
            reverse=True,
        )
        return tuple(effects)

    def _mediation_tests(
        self,
        observations: Sequence[Observation],
        target: str,
        priors: Sequence[MechanismPrior],
        *,
        control_names: Sequence[str],
    ) -> tuple[MediationResult, ...]:
        results = []
        available = {
            name
            for observation in observations
            for name in observation.explanatory_values
        }
        for prior in priors:
            if prior.exposure not in available:
                continue
            controls = tuple(
                dict.fromkeys(
                    [
                        *control_names,
                        *(name for name in prior.context_features if name in available),
                    ]
                )
            )
            for mediator in prior.mediators:
                if mediator not in available:
                    continue
                try:
                    results.append(
                        self._estimate_mediation(
                            observations,
                            exposure=prior.exposure,
                            mediator=mediator,
                            outcome=target,
                            control_names=controls,
                        )
                    )
                except ValueError:
                    continue
        return tuple(results)

    def _estimate_mediation(
        self,
        observations: Sequence[Observation],
        *,
        exposure: str,
        mediator: str,
        outcome: str,
        control_names: Sequence[str],
    ) -> MediationResult:
        rows: list[tuple[Observation, list[float]]] = []
        names = (exposure, mediator, *control_names)
        for observation in observations:
            y = observation.targets.get(outcome)
            values = [observation.explanatory_values.get(name) for name in names]
            if y is None or any(value is None for value in values):
                continue
            numeric = [float(y), *(float(value) for value in values)]
            if all(np.isfinite(numeric)):
                rows.append((observation, numeric))
        if len(rows) < self.config.min_feature_support:
            raise ValueError("too few complete mediation rows")
        matrix = np.asarray([row[1] for row in rows], dtype=float)
        sample_weight, _uncertainty_support = _precision_weights(
            tuple(row[0] for row in rows),
            outcome,
        )
        scaled, _center, _scale = standard_scale(matrix)
        y = scaled[:, 0]
        exposure_values = scaled[:, 1]
        mediator_values = scaled[:, 2]
        controls = scaled[:, 3:]
        mediator_design = np.column_stack((exposure_values, controls))
        _intercept_m, mediator_coefficients = ordinary_least_squares(
            mediator_design,
            mediator_values,
            sample_weight=sample_weight,
        )
        outcome_design = np.column_stack((exposure_values, mediator_values, controls))
        _intercept_y, outcome_coefficients = ordinary_least_squares(
            outcome_design,
            y,
            sample_weight=sample_weight,
        )
        a = float(mediator_coefficients[0])
        direct = float(outcome_coefficients[0])
        b = float(outcome_coefficients[1])
        indirect = a * b

        environments = tuple(row[0].environment for row in rows)
        rng = np.random.default_rng(self.config.random_seed)
        boot = []
        for _ in range(self.config.bootstrap_repeats):
            indices = stratified_bootstrap_indices(environments, rng=rng)
            _im, cm = ordinary_least_squares(
                mediator_design[indices],
                mediator_values[indices],
                sample_weight=sample_weight[indices],
            )
            _iy, cy = ordinary_least_squares(
                outcome_design[indices],
                y[indices],
                sample_weight=sample_weight[indices],
            )
            boot.append(float(cm[0] * cy[1]))
        tail = (1.0 - self.config.confidence_level) / 2.0
        total = direct + indirect
        fraction = indirect / total if abs(total) > 1.0e-10 else None
        source_observations = tuple(row[0] for row in rows)
        return MediationResult(
            exposure=exposure,
            mediator=mediator,
            outcome=outcome,
            exposure_to_mediator=a,
            mediator_to_outcome=b,
            direct_effect=direct,
            indirect_effect=indirect,
            indirect_ci_low=float(np.quantile(boot, tail)),
            indirect_ci_high=float(np.quantile(boot, 1.0 - tail)),
            mediated_fraction=None if fraction is None else float(fraction),
            sample_count=len(rows),
            evidence_level=_evidence_level(source_observations, (exposure,)),
            caveat=(
                "Linear mediation assumes the measured mediator order is correct and "
                "that exposure-mediator and mediator-outcome confounding is controlled."
            ),
        )

    def _matched_group_effects(
        self,
        observations: Sequence[Observation],
        target: str,
        effects: Sequence[FeatureEffect],
        *,
        control_names: Sequence[str],
    ) -> tuple[MatchedPairEffect, ...]:
        robust_groups = [
            effect.term
            for effect in effects
            if effect.term.startswith("group_")
            and " × " not in effect.term
            and not effect.term.endswith("^2")
            and effect.selection_frequency >= self.config.stability_threshold
        ][:6]
        available = {
            name
            for observation in observations
            for name in observation.explanatory_values
        }
        default_controls = tuple(
            name
            for name in (
                "heavy_atom_count",
                "heteroatom_fraction",
                "aromatic_atom_fraction",
                "crystallinity_fraction",
                "temperature_K",
                "electric_field_MV_m",
            )
            if name in available
        )
        controls = tuple(control_names) or default_controls
        if not controls:
            return ()
        results = []
        for exposure in robust_groups:
            try:
                results.append(
                    self.estimate_matched_effect(
                        observations,
                        exposure=exposure,
                        outcome=target,
                        control_names=controls,
                        threshold=0.0,
                    )
                )
            except ValueError:
                continue
        return tuple(results)

    def _hypotheses(
        self,
        observations: Sequence[Observation],
        target: str,
        effects: Sequence[FeatureEffect],
        mediations: Sequence[MediationResult],
        priors: Sequence[MechanismPrior],
    ) -> tuple[MechanismHypothesis, ...]:
        robust = [
            effect
            for effect in effects
            if effect.selection_frequency >= self.config.stability_threshold
            and effect.sign_consistency >= self.config.sign_consistency_threshold
            and effect.environment_sign_fraction
            >= self.config.environment_invariance_threshold
            and effect.direction != "uncertain"
        ]
        hypotheses: list[MechanismHypothesis] = []
        for index, effect in enumerate(robust[:20]):
            base_drivers = tuple(
                part.removesuffix("^2").strip() for part in effect.term.split(" × ")
            )
            matched_prior = next(
                (
                    prior
                    for prior in priors
                    if _term_matches_prior(effect.term, prior)
                ),
                None,
            )
            related_mediation = tuple(
                result
                for result in mediations
                if result.exposure in base_drivers
                and result.indirect_ci_low * result.indirect_ci_high > 0
            )
            mediators = tuple(result.mediator for result in related_mediation)
            direction = "increases" if effect.coefficient > 0 else "decreases"
            statement = (
                f"Within the measured applicability domain, {effect.term} is a stable "
                f"conditional driver that {direction} {target}."
            )
            if matched_prior is not None:
                statement += f" Candidate pathway: {matched_prior.expected_relation}"
            confidence = float(
                np.clip(
                    effect.selection_frequency
                    * effect.sign_consistency
                    * effect.environment_sign_fraction,
                    0.0,
                    1.0,
                )
            )
            citations = () if matched_prior is None else matched_prior.citations
            tests = (
                (
                    f"Intervene on {base_drivers[0]} while matching all reported controls.",
                    f"Repeat the measurement of {target} in a held-out polymer family.",
                )
                if matched_prior is None
                else matched_prior.falsification_tests
            )
            caveats = [
                "The coefficient is standardized and conditional on the included features.",
                "A stable association is not a causal effect without intervention.",
            ]
            if effect.is_interaction or effect.is_nonlinear:
                caveats.append("Interpret the term jointly; a marginal monotonic claim is invalid.")
            hypotheses.append(
                MechanismHypothesis(
                    hypothesis_id=f"{target}:H{index + 1:02d}",
                    statement=statement,
                    target=target,
                    drivers=base_drivers,
                    mediators=mediators,
                    evidence_level=_evidence_level(observations, base_drivers),
                    confidence=confidence,
                    supporting_effects=(effect.term,),
                    falsification_tests=tests,
                    citations=citations,
                    caveats=tuple(caveats),
                )
            )
        return tuple(hypotheses)

    def _warnings(
        self,
        prepared: _PreparedData,
        effects: Sequence[FeatureEffect],
    ) -> tuple[str, ...]:
        warnings = []
        environment_count = len(set(prepared.environments))
        if environment_count < 2:
            warnings.append(
                "Only one environment is represented; invariance across datasets, "
                "polymer families, or fidelities was not testable."
            )
        if len(prepared.observations) < 5 * len(prepared.base_names):
            warnings.append(
                "The feature-to-sample ratio is high; treat weak and interaction terms "
                "as exploratory even after regularization."
            )
        if not any(
            effect.selection_frequency >= self.config.stability_threshold
            for effect in effects
        ):
            warnings.append(
                "No term met the stability threshold; gather targeted data before "
                "claiming a mechanism."
            )
        if not any(observation.intervention for observation in prepared.observations):
            warnings.append(
                "No explicit intervention metadata was found; causal language is disabled."
            )
        if 0 < prepared.uncertainty_support < len(prepared.observations):
            warnings.append(
                "Target uncertainty was missing for some rows; the median reported "
                "uncertainty was used for precision weighting."
            )
        if prepared.uncertainty_support:
            warnings.append(
                "Sparse effects and mediation use target-uncertainty weights; the "
                "reported symbolic law is an unweighted compact approximation."
            )
        return tuple(warnings)


def _precision_weights(
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
    supported = np.isfinite(uncertainty) & (uncertainty >= 0)
    support = int(np.count_nonzero(supported))
    if support == 0:
        return np.ones(len(observations), dtype=float), 0
    median = float(np.median(uncertainty[supported]))
    resolved = np.where(supported, uncertainty, median)
    floor = max(median * 0.05, 1.0e-12)
    precision = 1.0 / (resolved * resolved + floor * floor)
    if precision.size >= 4:
        lower, upper = np.quantile(precision, (0.05, 0.95))
        precision = np.clip(precision, lower, upper)
    precision /= float(np.mean(precision))
    return precision, support


def _effective_sample_size(sample_weight: np.ndarray) -> float:
    weights = np.asarray(sample_weight, dtype=float)
    return float(np.sum(weights) ** 2 / max(float(weights @ weights), 1.0e-15))


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) < 1.0e-12 or float(np.std(right)) < 1.0e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _term_matches_prior(term: str, prior: MechanismPrior) -> bool:
    components = tuple(part.removesuffix("^2").strip() for part in term.split(" × "))
    if prior.exposure == "functional_group_identity" and any(
        component.startswith("group_") for component in components
    ):
        return True
    return prior.exposure in components or any(
        mediator in components for mediator in prior.mediators
    )


def _evidence_level(
    observations: Sequence[Observation],
    drivers: Sequence[str],
) -> EvidenceLevel:
    if any(
        observation.provenance.get("controlled_intervention") is True
        and any(observation.was_intervened(driver) for driver in drivers)
        for observation in observations
    ):
        return EvidenceLevel.INTERVENTION
    fidelities = {observation.fidelity.lower() for observation in observations}
    has_simulation = any(
        token in fidelity
        for fidelity in fidelities
        for token in ("simulation", "dft", "md", "neb")
    )
    has_experiment = any(
        token in fidelity
        for fidelity in fidelities
        for token in ("experiment", "measured", "literature")
    )
    if has_simulation and has_experiment:
        return EvidenceLevel.TRIANGULATED
    if has_simulation:
        return EvidenceLevel.SIMULATION
    if len({observation.environment for observation in observations}) > 1:
        return EvidenceLevel.MULTI_ENVIRONMENT
    return EvidenceLevel.ASSOCIATION


__all__ = ["MechanismDiscoveryEngine"]
