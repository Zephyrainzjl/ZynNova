"""Strict schemas for plans, tool execution, and verification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from ..exceptions import LLMProtocolError


SUPPORTED_MODELS = {
    "p2d",
    "battery3d",
    "scalar_fem",
    "linear_elasticity",
    "neo_hookean",
}


@dataclass(frozen=True, slots=True)
class PlanArgument:
    name: str
    value_json: str

    def value(self) -> object:
        try:
            return json.loads(self.value_json)
        except json.JSONDecodeError as exc:
            raise LLMProtocolError(f"argument {self.name!r} contains invalid JSON") from exc


@dataclass(frozen=True, slots=True)
class PlanStep:
    id: str
    tool: str
    arguments: tuple[PlanArgument, ...]
    depends_on: tuple[str, ...]
    purpose: str

    def kwargs(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for argument in self.arguments:
            if argument.name in result:
                raise LLMProtocolError(
                    f"step {self.id!r} repeats argument {argument.name!r}"
                )
            result[argument.name] = argument.value()
        return result


@dataclass(frozen=True, slots=True)
class ValidationRule:
    metric: str
    operator: str
    threshold: float
    unit: str
    rationale: str


@dataclass(frozen=True, slots=True)
class SimulationPlan:
    version: str
    objective: str
    model: str
    assumptions: tuple[str, ...]
    steps: tuple[PlanStep, ...]
    validations: tuple[ValidationRule, ...]
    requested_outputs: tuple[str, ...]
    unresolved: tuple[str, ...]
    approval_required: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SimulationPlan:
        _exact_keys(
            value,
            {
                "version",
                "objective",
                "model",
                "assumptions",
                "steps",
                "validations",
                "requested_outputs",
                "unresolved",
                "approval_required",
            },
            "simulation plan",
        )
        steps: list[PlanStep] = []
        for raw_step in _object_list(value["steps"], "steps"):
            _exact_keys(
                raw_step,
                {"id", "tool", "arguments", "depends_on", "purpose"},
                "plan step",
            )
            arguments: list[PlanArgument] = []
            for raw_argument in _object_list(raw_step["arguments"], "arguments"):
                _exact_keys(raw_argument, {"name", "value_json"}, "plan argument")
                arguments.append(
                    PlanArgument(
                        _string(raw_argument["name"], "argument.name"),
                        _string(raw_argument["value_json"], "argument.value_json"),
                    )
                )
            steps.append(
                PlanStep(
                    id=_string(raw_step["id"], "step.id"),
                    tool=_string(raw_step["tool"], "step.tool"),
                    arguments=tuple(arguments),
                    depends_on=_string_tuple(raw_step["depends_on"], "step.depends_on"),
                    purpose=_string(raw_step["purpose"], "step.purpose"),
                )
            )
        validations: list[ValidationRule] = []
        for raw_rule in _object_list(value["validations"], "validations"):
            _exact_keys(
                raw_rule,
                {"metric", "operator", "threshold", "unit", "rationale"},
                "validation rule",
            )
            threshold = raw_rule["threshold"]
            if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
                raise LLMProtocolError("validation threshold must be numeric")
            validations.append(
                ValidationRule(
                    metric=_string(raw_rule["metric"], "validation.metric"),
                    operator=_string(raw_rule["operator"], "validation.operator"),
                    threshold=float(threshold),
                    unit=_string(raw_rule["unit"], "validation.unit"),
                    rationale=_string(raw_rule["rationale"], "validation.rationale"),
                )
            )
        approval = value["approval_required"]
        if not isinstance(approval, bool):
            raise LLMProtocolError("approval_required must be boolean")
        plan = cls(
            version=_string(value["version"], "version"),
            objective=_string(value["objective"], "objective"),
            model=_string(value["model"], "model"),
            assumptions=_string_tuple(value["assumptions"], "assumptions"),
            steps=tuple(steps),
            validations=tuple(validations),
            requested_outputs=_string_tuple(
                value["requested_outputs"], "requested_outputs"
            ),
            unresolved=_string_tuple(value["unresolved"], "unresolved"),
            approval_required=approval,
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        if self.version != "1.0":
            raise LLMProtocolError("unsupported simulation-plan version")
        if self.model not in SUPPORTED_MODELS:
            raise LLMProtocolError(f"unsupported simulation model {self.model!r}")
        if not self.objective.strip() or not self.requested_outputs:
            raise LLMProtocolError("plan objective and requested outputs cannot be empty")
        if len(self.steps) > 32:
            raise LLMProtocolError("simulation plan exceeds the 32-step safety limit")
        identifiers = [step.id for step in self.steps]
        if len(set(identifiers)) != len(identifiers) or any(not item for item in identifiers):
            raise LLMProtocolError("plan step identifiers must be unique and non-empty")
        seen: set[str] = set()
        forbidden_arguments = {"code", "python", "shell", "command", "script"}
        for step in self.steps:
            if any(dependency not in seen for dependency in step.depends_on):
                raise LLMProtocolError(
                    f"step {step.id!r} has a missing or forward dependency"
                )
            if any(argument.name.lower() in forbidden_arguments for argument in step.arguments):
                raise LLMProtocolError(
                    f"step {step.id!r} requests executable-code argument"
                )
            step.kwargs()
            seen.add(step.id)
        for rule in self.validations:
            if rule.operator not in {"<", "<=", ">", ">=", "=="}:
                raise LLMProtocolError(
                    f"unsupported validation operator {rule.operator!r}"
                )


@dataclass(frozen=True, slots=True)
class VerificationReport:
    verdict: str
    summary: str
    findings: tuple[str, ...]
    required_actions: tuple[str, ...]
    evidence_used: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> VerificationReport:
        _exact_keys(
            value,
            {"verdict", "summary", "findings", "required_actions", "evidence_used"},
            "verification report",
        )
        verdict = _string(value["verdict"], "verdict")
        if verdict not in {"pass", "fail", "inconclusive"}:
            raise LLMProtocolError("verification verdict is invalid")
        return cls(
            verdict=verdict,
            summary=_string(value["summary"], "summary"),
            findings=_string_tuple(value["findings"], "findings"),
            required_actions=_string_tuple(value["required_actions"], "required_actions"),
            evidence_used=_string_tuple(value["evidence_used"], "evidence_used"),
        )


def simulation_plan_schema() -> dict[str, object]:
    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "version": {"type": "string", "enum": ["1.0"]},
            "objective": {"type": "string"},
            "model": {"type": "string", "enum": sorted(SUPPORTED_MODELS)},
            "assumptions": string_array,
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "tool": {"type": "string"},
                        "arguments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "value_json": {"type": "string"},
                                },
                                "required": ["name", "value_json"],
                                "additionalProperties": False,
                            },
                        },
                        "depends_on": string_array,
                        "purpose": {"type": "string"},
                    },
                    "required": ["id", "tool", "arguments", "depends_on", "purpose"],
                    "additionalProperties": False,
                },
            },
            "validations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string"},
                        "operator": {
                            "type": "string",
                            "enum": ["<", "<=", ">", ">=", "=="],
                        },
                        "threshold": {"type": "number"},
                        "unit": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["metric", "operator", "threshold", "unit", "rationale"],
                    "additionalProperties": False,
                },
            },
            "requested_outputs": string_array,
            "unresolved": string_array,
            "approval_required": {"type": "boolean"},
        },
        "required": [
            "version",
            "objective",
            "model",
            "assumptions",
            "steps",
            "validations",
            "requested_outputs",
            "unresolved",
            "approval_required",
        ],
        "additionalProperties": False,
    }


def verification_report_schema() -> dict[str, object]:
    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["pass", "fail", "inconclusive"],
            },
            "summary": {"type": "string"},
            "findings": string_array,
            "required_actions": string_array,
            "evidence_used": string_array,
        },
        "required": [
            "verdict",
            "summary",
            "findings",
            "required_actions",
            "evidence_used",
        ],
        "additionalProperties": False,
    }


def _exact_keys(value: Mapping[str, object], keys: set[str], label: str) -> None:
    if set(value) != keys:
        missing = sorted(keys - set(value))
        extra = sorted(set(value) - keys)
        raise LLMProtocolError(f"{label} keys differ; missing={missing}, extra={extra}")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise LLMProtocolError(f"{label} must be a string")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise LLMProtocolError(f"{label} must be an array of strings")
    return tuple(value)


def _object_list(value: object, label: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise LLMProtocolError(f"{label} must be an array of objects")
    return value


__all__ = [
    "PlanArgument",
    "PlanStep",
    "SUPPORTED_MODELS",
    "SimulationPlan",
    "ValidationRule",
    "VerificationReport",
    "simulation_plan_schema",
    "verification_report_schema",
]
