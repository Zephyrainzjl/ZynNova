"""Provider protocol."""
from __future__ import annotations

from typing import Protocol, Sequence

from ..types import Message, ProviderResponse, ToolSpec


class ModelProvider(Protocol):
    name: str
    async def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()) -> ProviderResponse: ...
    async def aclose(self) -> None: ...


__all__ = ["ModelProvider"]
