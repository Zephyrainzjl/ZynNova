"""Plan → validate → allowlisted execute → verify orchestration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Mapping

from ..exceptions import LLMProtocolError
from .providers import StructuredProvider
from .schema import (
    SimulationPlan,
    VerificationReport,
    simulation_plan_schema,
    verification_report_schema,
)


ToolCallable = Callable[..., object]
ValidatorCallable = Callable[[Mapping[str, object]], tuple[bool, str]]


@dataclass(frozen=True, slots=True)
class SimulationTool:
    name: str
    function: ToolCallable
    description: str
    input_schema: Mapping[str, object] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
    )
    side_effect: str = "none"

    def __post_init__(self) -> None:
        if self.side_effect not in {"none", "local_write", "external"}:
            raise ValueError("tool side_effect is invalid")
        if self.input_schema.get("type") != "object":
            raise ValueError("simulation tool input_schema must describe an object")


@dataclass(slots=True)
class SafeToolRegistry:
    _tools: dict[str, SimulationTool] = field(default_factory=dict, init=False)

    def register(self, tool: SimulationTool) -> None:
        if not tool.name or tool.name in self._tools:
            raise ValueError(f"duplicate or empty simulation tool {tool.name!r}")
        self._tools[tool.name] = tool

    def descriptions(self) -> list[dict[str, str]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": json.dumps(
                    tool.input_schema, ensure_ascii=False, separators=(",", ":")
                ),
                "side_effect": tool.side_effect,
            }
            for tool in self._tools.values()
        ]

    def execute(
        self,
        plan: SimulationPlan,
        *,
        approve_local_writes: bool = False,
        approve_external_actions: bool = False,
    ) -> dict[str, object]:
        if plan.unresolved:
            raise LLMProtocolError(
                "plan has unresolved inputs and cannot execute: " + "; ".join(plan.unresolved)
            )
        results: dict[str, object] = {}
        for step in plan.steps:
            try:
                tool = self._tools[step.tool]
            except KeyError as exc:
                raise LLMProtocolError(
                    f"plan requested non-allowlisted tool {step.tool!r}"
                ) from exc
            if tool.side_effect == "local_write" and not approve_local_writes:
                raise LLMProtocolError(f"tool {tool.name!r} requires local-write approval")
            if tool.side_effect == "external" and not approve_external_actions:
                raise LLMProtocolError(f"tool {tool.name!r} requires external-action approval")
            if tool.side_effect != "none" and not plan.approval_required:
                raise LLMProtocolError(
                    f"plan failed to declare approval for side-effecting tool {tool.name!r}"
                )
            kwargs = step.kwargs()
            _validate_tool_arguments(kwargs, tool.input_schema, tool.name)
            kwargs["dependencies"] = {
                dependency: results[dependency] for dependency in step.depends_on
            }
            results[step.id] = tool.function(**kwargs)
        return results


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    plan: SimulationPlan
    execution: Mapping[str, object] | None
    deterministic_checks: tuple[str, ...]
    verification: VerificationReport | None
    audit: tuple[Mapping[str, object], ...]


class SimulationOrchestrator:
    """Use an LLM for bounded planning and critique, never as a numerical kernel."""

    def __init__(
        self,
        provider: StructuredProvider,
        registry: SafeToolRegistry,
        *,
        validators: tuple[ValidatorCallable, ...] = (),
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.validators = validators
        self._audit: list[Mapping[str, object]] = []

    def plan(self, request: str, context: Mapping[str, object]) -> SimulationPlan:
        if not request.strip():
            raise ValueError("simulation request cannot be empty")
        system = (
            "You plan scientific simulations for ZynSim. Be concise and outcome-focused. "
            "Use only the supplied allowlisted tools. Never invent material constants, units, "
            "mesh convergence, boundary conditions, or experimental calibration. Put missing "
            "facts in unresolved. A plan may inspect or compute locally, but must mark approval "
            "for any write or external action. Do not emit code or shell commands. Include "
            "conservation, residual, mesh/time-step convergence, and domain checks that are "
            "material to the requested model. The deterministic solver and validators, not "
            "your prose, decide numerical correctness."
        )
        user_payload = {
            "goal": request,
            "context": dict(context),
            "available_tools": self.registry.descriptions(),
            "success_contract": (
                "Return a complete executable plan when inputs suffice; otherwise return "
                "a non-executable plan with every material ambiguity listed in unresolved."
            ),
        }
        response = self.provider.generate(
            system=system,
            user=json.dumps(user_payload, ensure_ascii=False, default=_json_default),
            schema_name="zynsim_simulation_plan",
            schema=simulation_plan_schema(),
        )
        plan = SimulationPlan.from_mapping(response.data)
        known = {description["name"] for description in self.registry.descriptions()}
        unknown = sorted({step.tool for step in plan.steps} - known)
        if unknown:
            raise LLMProtocolError(f"plan references unavailable tools: {unknown}")
        self._record(
            "plan",
            {
                "request_hash": _hash(request),
                "model": response.model,
                "response_id": response.response_id,
                "step_count": len(plan.steps),
                "unresolved_count": len(plan.unresolved),
            },
        )
        return plan

    def execute(
        self,
        plan: SimulationPlan,
        *,
        approve_local_writes: bool = False,
        approve_external_actions: bool = False,
    ) -> tuple[Mapping[str, object], tuple[str, ...]]:
        execution = self.registry.execute(
            plan,
            approve_local_writes=approve_local_writes,
            approve_external_actions=approve_external_actions,
        )
        checks: list[str] = []
        serializable = {key: _json_safe(value) for key, value in execution.items()}
        for validator in self.validators:
            passed, message = validator(serializable)
            checks.append(("PASS: " if passed else "FAIL: ") + message)
            if not passed:
                self._record("deterministic_validation_failed", {"message": message})
                raise LLMProtocolError(
                    f"deterministic validation failed; LLM verification cannot override it: "
                    f"{message}"
                )
        self._record(
            "execute",
            {
                "steps": list(execution),
                "result_hash": _hash(json.dumps(serializable, sort_keys=True)),
            },
        )
        return execution, tuple(checks)

    def verify(
        self,
        request: str,
        plan: SimulationPlan,
        execution: Mapping[str, object],
        deterministic_checks: tuple[str, ...],
    ) -> VerificationReport:
        system = (
            "You are an independent simulation-results reviewer. Assess only the supplied "
            "plan, deterministic evidence, and result summaries. Do not claim a check occurred "
            "unless its evidence is present. Missing calibration, conservation, convergence, "
            "or domain evidence makes the verdict inconclusive or fail. You cannot override a "
            "deterministic failure and cannot call tools or change numerical results."
        )
        evidence = {
            "request": request,
            "plan": _json_safe(plan),
            "result_summaries": _json_safe(execution),
            "deterministic_checks": deterministic_checks,
        }
        response = self.provider.generate(
            system=system,
            user=json.dumps(evidence, ensure_ascii=False, default=_json_default),
            schema_name="zynsim_verification_report",
            schema=verification_report_schema(),
        )
        report = VerificationReport.from_mapping(response.data)
        self._record(
            "verify",
            {
                "verdict": report.verdict,
                "model": response.model,
                "response_id": response.response_id,
            },
        )
        return report

    def run(
        self,
        request: str,
        context: Mapping[str, object],
        *,
        execute: bool = True,
        verify: bool = True,
        approve_local_writes: bool = False,
        approve_external_actions: bool = False,
    ) -> OrchestrationResult:
        plan = self.plan(request, context)
        if not execute or plan.unresolved:
            return OrchestrationResult(
                plan=plan,
                execution=None,
                deterministic_checks=(),
                verification=None,
                audit=tuple(self._audit),
            )
        execution, checks = self.execute(
            plan,
            approve_local_writes=approve_local_writes,
            approve_external_actions=approve_external_actions,
        )
        report = (
            self.verify(request, plan, execution, checks) if verify else None
        )
        return OrchestrationResult(
            plan=plan,
            execution=execution,
            deterministic_checks=checks,
            verification=report,
            audit=tuple(self._audit),
        )

    def _record(self, event: str, payload: Mapping[str, object]) -> None:
        self._audit.append({"index": len(self._audit), "event": event, **dict(payload)})


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_safe(value: object) -> object:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _json_safe(getattr(value, name))
            for name in value.__dataclass_fields__  # type: ignore[attr-defined]
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _json_default(value: object) -> object:
    return _json_safe(value)


def _validate_tool_arguments(
    arguments: Mapping[str, object],
    schema: Mapping[str, object],
    tool_name: str,
) -> None:
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    additional = schema.get("additionalProperties", False)
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        raise ValueError(f"tool {tool_name!r} has an invalid input schema")
    missing = [name for name in required if name not in arguments]
    if missing:
        raise LLMProtocolError(
            f"tool {tool_name!r} is missing required arguments {missing}"
        )
    if not additional:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise LLMProtocolError(
                f"tool {tool_name!r} received unknown arguments {unknown}"
            )
    expected_python_types = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    for name, value in arguments.items():
        definition = properties.get(name)
        if not isinstance(definition, Mapping) or "type" not in definition:
            continue
        expected = definition["type"]
        if expected not in expected_python_types:
            raise ValueError(
                f"tool {tool_name!r} uses unsupported schema type {expected!r}"
            )
        if expected in {"number", "integer"} and isinstance(value, bool):
            valid = False
        else:
            valid = isinstance(value, expected_python_types[expected])
        if not valid:
            raise LLMProtocolError(
                f"tool {tool_name!r} argument {name!r} must be {expected}"
            )


__all__ = [
    "OrchestrationResult",
    "SafeToolRegistry",
    "SimulationOrchestrator",
    "SimulationTool",
    "ToolCallable",
    "ValidatorCallable",
]
