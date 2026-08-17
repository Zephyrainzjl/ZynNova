from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .physics_learning import physics_learning_report_from_dict
from .schema import (
    DiscoveredLaw,
    DiscoveryReport,
    EvidenceLevel,
    FeatureEffect,
    MatchedPairEffect,
    MechanismHypothesis,
    MediationResult,
)


def save_discovery_report(
    report: DiscoveryReport,
    path: str | Path,
    *,
    markdown: bool | None = None,
) -> Path:
    """Save a report as machine-readable JSON or an auditable Markdown summary."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    use_markdown = (
        destination.suffix.lower() in {".md", ".markdown"}
        if markdown is None
        else bool(markdown)
    )
    if use_markdown:
        destination.write_text(render_discovery_markdown(report), encoding="utf-8")
    else:
        destination.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return destination


def render_discovery_markdown(report: DiscoveryReport) -> str:
    """Render conclusions together with the checks that could falsify them."""

    lines = [
        f"# Polymer mechanism discovery: `{report.target}`",
        "",
        f"- Samples: {report.sample_count}",
        f"- Environments: {', '.join(report.environments) or 'unknown'}",
        f"- Candidate features: {len(report.feature_names)}",
        (
            "- Causal status: "
            + str(
                report.diagnostics.get(
                    "causal_status",
                    "hypothesis-generating unless intervention evidence is present",
                )
            )
        ),
        "",
        "## Stable effects",
        "",
        "| Term | Direction | Coefficient | 95% bootstrap CI | Selection | Environment sign |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    robust = report.robust_effects()
    if robust:
        for effect in robust:
            lines.append(
                "| {term} | {direction} | {coefficient:.3f} | "
                "[{low:.3f}, {high:.3f}] | {selection:.0%} | {environment:.0%} |".format(
                    term=_escape_table(effect.term),
                    direction=effect.direction,
                    coefficient=effect.coefficient,
                    low=effect.ci_low,
                    high=effect.ci_high,
                    selection=effect.selection_frequency,
                    environment=effect.environment_sign_fraction,
                )
            )
    else:
        lines.append("| _No term passed the configured stability criteria_ |  |  |  |  |  |")

    lines.extend(["", "## Mechanism hypotheses", ""])
    if not report.hypotheses:
        lines.append(
            "No mechanism statement passed stability, sign, and environment checks."
        )
    for hypothesis in report.hypotheses:
        lines.extend(
            [
                f"### {hypothesis.hypothesis_id}",
                "",
                hypothesis.statement,
                "",
                (
                    f"Evidence: `{hypothesis.evidence_level.value}`; "
                    f"confidence score: {hypothesis.confidence:.2f}."
                ),
                "",
                "Falsification tests:",
                "",
                *[f"- {test}" for test in hypothesis.falsification_tests],
            ]
        )
        if hypothesis.citations:
            lines.extend(
                [
                    "",
                    "Literature priors:",
                    "",
                    *[f"- {citation}" for citation in hypothesis.citations],
                ]
            )
        if hypothesis.caveats:
            lines.extend(
                [
                    "",
                    "Caveats:",
                    "",
                    *[f"- {caveat}" for caveat in hypothesis.caveats],
                ]
            )
        lines.append("")

    if report.mediations:
        lines.extend(
            [
                "## Mediation tests",
                "",
                "| Exposure → mediator → outcome | Indirect effect | 95% CI | Evidence |",
                "|---|---:|---:|---|",
            ]
        )
        for result in report.mediations:
            lines.append(
                "| {path} | {effect:.3f} | [{low:.3f}, {high:.3f}] | {level} |".format(
                    path=_escape_table(
                        f"{result.exposure} → {result.mediator} → {result.outcome}"
                    ),
                    effect=result.indirect_effect,
                    low=result.indirect_ci_low,
                    high=result.indirect_ci_high,
                    level=result.evidence_level.value,
                )
            )
        lines.append("")

    if report.matched_effects:
        lines.extend(
            [
                "## Matched controls",
                "",
                "| Exposure | Pairs | Outcome difference | 95% CI | Match distance |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for result in report.matched_effects:
            lines.append(
                "| {exposure} | {pairs} | {difference:.4g} | "
                "[{low:.4g}, {high:.4g}] | {distance:.3f} |".format(
                    exposure=_escape_table(result.exposure),
                    pairs=result.matched_pairs,
                    difference=result.average_difference,
                    low=result.ci_low,
                    high=result.ci_high,
                    distance=result.median_control_distance,
                )
            )
        lines.append("")

    if report.law is not None:
        lines.extend(
            [
                "## Sparse empirical law",
                "",
                f"`{report.law.expression}`",
                "",
                (
                    f"Training R²: {report.law.train_r2:.3f}; "
                    f"held-environment/fold R²: {report.law.validation_r2:.3f}; "
                    f"BIC: {report.law.bic:.2f}."
                ),
                "",
                f"_{report.law.caveat}_",
                "",
            ]
        )

    if report.physics_learning is not None:
        physics = report.physics_learning
        lines.extend(
            [
                "## Neural-symbolic physics learning",
                "",
                (
                    f"Features: {len(physics.feature_names)}; "
                    f"target unit: `{physics.target_unit}`."
                ),
                "",
            ]
        )
        if physics.equations:
            lines.extend(
                [
                    "| Rank | Backend | Equation | Validation R² | Complexity | Units |",
                    "|---:|---|---|---:|---:|---|",
                ]
            )
            for rank, equation in enumerate(physics.equations[:5], start=1):
                unit_status = {
                    True: "constrained",
                    False: "inconsistent",
                    None: "unchecked",
                }[equation.unit_consistent]
                lines.append(
                    "| {rank} | {backend} | `{equation}` | {r2:.3f} | "
                    "{complexity} | {units} |".format(
                        rank=rank,
                        backend=equation.backend,
                        equation=_escape_table(equation.expression),
                        r2=equation.validation_r2,
                        complexity=equation.complexity,
                        units=unit_status,
                    )
                )
            lines.append("")
        else:
            lines.extend(
                [
                    "No symbolic backend returned a validated equation.",
                    "",
                ]
            )

        interaction = physics.interaction_decomposition
        if interaction is not None:
            lines.extend(
                [
                    (
                        f"Interaction oracle: `{interaction.oracle}`; "
                        f"validation R²: {interaction.oracle_validation_r2:.3f}."
                    ),
                    "",
                ]
            )
            if interaction.edges:
                lines.extend(
                    [
                        "| Non-additive feature pair | Hessian score | Signed score |",
                        "|---|---:|---:|",
                    ]
                )
                for edge in interaction.edges[:8]:
                    lines.append(
                        "| {pair} | {score:.4g} | {signed:.4g} |".format(
                            pair=_escape_table(f"{edge.left} × {edge.right}"),
                            score=edge.score,
                            signed=edge.signed_score,
                        )
                    )
                lines.append("")
            lines.extend([f"_{interaction.caveat}_", ""])

        if physics.backend_status:
            lines.extend(
                [
                    "Backend audit:",
                    "",
                    *[
                        (
                            f"- `{status.name}`: "
                            f"{'executed' if status.executed else 'not executed'}; "
                            f"{status.detail}"
                        )
                        for status in physics.backend_status
                    ],
                    "",
                ]
            )
        if physics.warnings:
            lines.extend(
                [
                    "Physics-learning warnings:",
                    "",
                    *[f"- {warning}" for warning in physics.warnings],
                    "",
                ]
            )

    if report.warnings:
        lines.extend(
            [
                "## Warnings",
                "",
                *[f"- {warning}" for warning in report.warnings],
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def discovery_report_from_dict(payload: Mapping[str, Any]) -> DiscoveryReport:
    """Rehydrate the versioned JSON form produced by ``DiscoveryReport.to_dict``."""

    law_payload = payload.get("law")
    law = (
        None
        if law_payload is None
        else DiscoveredLaw(
            target=str(law_payload["target"]),
            expression=str(law_payload["expression"]),
            intercept=float(law_payload["intercept"]),
            terms=tuple(law_payload["terms"]),
            coefficients=tuple(float(value) for value in law_payload["coefficients"]),
            coefficient_ci=tuple(
                (float(interval[0]), float(interval[1]))
                for interval in law_payload["coefficient_ci"]
            ),
            train_r2=float(law_payload["train_r2"]),
            validation_r2=float(law_payload["validation_r2"]),
            bic=float(law_payload["bic"]),
            sample_count=int(law_payload["sample_count"]),
            environments=tuple(law_payload.get("environments", ())),
            normalized=bool(law_payload.get("normalized", True)),
            caveat=str(
                law_payload.get(
                    "caveat",
                    (
                        "This is a compact empirical relation in standardized "
                        "variables; it is not a dimensionally exact physical law "
                        "until independently validated."
                    ),
                )
            ),
        )
    )
    physics_payload = payload.get("physics_learning")
    physics_learning = (
        None
        if physics_payload is None
        else physics_learning_report_from_dict(physics_payload)
    )
    return DiscoveryReport(
        target=str(payload["target"]),
        sample_count=int(payload["sample_count"]),
        feature_names=tuple(payload["feature_names"]),
        effects=tuple(FeatureEffect(**item) for item in payload.get("effects", ())),
        hypotheses=tuple(
            MechanismHypothesis(
                **{
                    **item,
                    "drivers": tuple(item["drivers"]),
                    "mediators": tuple(item["mediators"]),
                    "supporting_effects": tuple(item["supporting_effects"]),
                    "falsification_tests": tuple(item["falsification_tests"]),
                    "citations": tuple(item.get("citations", ())),
                    "caveats": tuple(item.get("caveats", ())),
                    "evidence_level": EvidenceLevel(item["evidence_level"]),
                }
            )
            for item in payload.get("hypotheses", ())
        ),
        mediations=tuple(
            MediationResult(
                **{
                    **item,
                    "evidence_level": EvidenceLevel(item["evidence_level"]),
                }
            )
            for item in payload.get("mediations", ())
        ),
        matched_effects=tuple(
            MatchedPairEffect(
                **{
                    **item,
                    "evidence_level": EvidenceLevel(item["evidence_level"]),
                }
            )
            for item in payload.get("matched_effects", ())
        ),
        law=law,
        physics_learning=physics_learning,
        environments=tuple(payload.get("environments", ())),
        diagnostics=dict(payload.get("diagnostics", {})),
        warnings=tuple(payload.get("warnings", ())),
        schema_version=str(payload.get("schema_version", "1.0")),
    )


def _escape_table(value: str) -> str:
    return str(value).replace("|", "\\|")


__all__ = [
    "discovery_report_from_dict",
    "render_discovery_markdown",
    "save_discovery_report",
]
