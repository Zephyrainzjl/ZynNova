"""Asynchronous JSON-schema tool registry."""
from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from ..types import ToolSpec

ToolFn = Callable[..., Any] | Callable[..., Awaitable[Any]]


@dataclass(slots=True)
class RegisteredTool:
    spec: ToolSpec
    fn: ToolFn


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, fn: ToolFn, *, replace: bool = False) -> None:
        if spec.name in self._tools and not replace:
            raise KeyError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = RegisteredTool(spec, fn)

    def add(
        self,
        name: str,
        fn: ToolFn,
        *,
        description: str,
        parameters: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> None:
        schema = parameters or {"type": "object", "properties": {}, "additionalProperties": True}
        self.register(ToolSpec(name, description, schema), fn, replace=replace)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(item.spec for item in self._tools.values())

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    async def execute(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        fn = self._tools[name].fn
        if inspect.iscoroutinefunction(fn):
            return await fn(**dict(arguments))
        return await asyncio.to_thread(fn, **dict(arguments))


__all__ = ["RegisteredTool", "ToolRegistry"]
