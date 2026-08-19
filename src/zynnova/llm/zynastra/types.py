"""Provider-neutral message, tool and response types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class Message:
    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple["ToolCall", ...] = ()

    def as_openai(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            out["name"] = self.name
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            out["tool_calls"] = [call.as_openai() for call in self.tool_calls]
        return out


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def as_openai(self) -> dict[str, Any]:
        import json
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(dict(self.arguments))},
        }


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]

    def as_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


@dataclass(slots=True)
class ProviderResponse:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    raw: Any = None


@dataclass(slots=True)
class AgentResult:
    text: str
    messages: tuple[Message, ...]
    steps: int
    session_id: str
    usage: Mapping[str, int] = field(default_factory=dict)


__all__ = ["AgentResult", "Message", "ProviderResponse", "ToolCall", "ToolSpec"]
