"""Optional LiteLLM provider for broad hosted/local endpoint coverage."""
from __future__ import annotations

import json
from typing import Any, Sequence

from ..config import ProviderConfig
from ..types import Message, ProviderResponse, ToolCall, ToolSpec


class LiteLLMProvider:
    name = "litellm"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError("install zynnova[llm-providers] to use LiteLLM") from exc
        self._litellm = litellm

    async def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.as_openai() for m in messages],
            **dict(self.config.extra_body),
        }
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.base_url and self.config.base_url != "litellm://":
            kwargs["api_base"] = self.config.base_url
        if tools:
            kwargs["tools"] = [t.as_openai() for t in tools]
            kwargs["tool_choice"] = "auto"
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.max_output_tokens is not None:
            kwargs["max_tokens"] = self.config.max_output_tokens
        result = await self._litellm.acompletion(**kwargs)
        choice = result.choices[0]
        message = choice.message
        calls: list[ToolCall] = []
        for item in getattr(message, "tool_calls", None) or []:
            raw_args = item.function.arguments or "{}"
            try: args = json.loads(raw_args)
            except Exception: args = {"_raw": raw_args}
            calls.append(ToolCall(str(item.id), str(item.function.name), args))
        usage_obj = getattr(result, "usage", None)
        usage = {}
        if usage_obj:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage_obj, key, None)
                if value is not None: usage[key] = int(value)
        return ProviderResponse(str(message.content or ""), tuple(calls), getattr(choice, "finish_reason", None), usage, result)

    async def aclose(self) -> None:
        return None


__all__ = ["LiteLLMProvider"]
