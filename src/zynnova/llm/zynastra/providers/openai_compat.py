"""Dependency-light OpenAI Responses / Chat-Completions compatible provider."""
from __future__ import annotations

import json
from typing import Any, Sequence

from ..config import ProviderConfig
from ..types import Message, ProviderResponse, ToolCall, ToolSpec


class OpenAICompatibleProvider:
    name = "openai-compatible"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("install zynnova[llm] to use HTTP providers") from exc
        headers = {"Content-Type": "application/json", **dict(config.extra_headers)}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=config.timeout_seconds,
        )

    def _style(self) -> str:
        if self.config.api_style != "auto":
            return self.config.api_style
        host = self.config.base_url.lower()
        return "responses" if "api.openai.com" in host else "chat-completions"

    async def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()) -> ProviderResponse:
        style = self._style()
        if style == "responses":
            return await self._responses(messages, tools)
        return await self._chat(messages, tools)

    async def _chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.as_openai() for m in messages],
            **dict(self.config.extra_body),
        }
        if tools:
            payload["tools"] = [t.as_openai() for t in tools]
            payload["tool_choice"] = "auto"
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.max_output_tokens is not None:
            payload["max_tokens"] = self.config.max_output_tokens
        if self.config.response_format is not None:
            payload["response_format"] = dict(self.config.response_format)
        response = await self._client.post("chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        message = choice.get("message", {})
        calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            fn = item.get("function", {})
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            calls.append(ToolCall(str(item.get("id", "tool")), str(fn.get("name", "")), args))
        usage = data.get("usage") or {}
        return ProviderResponse(
            text=message.get("content") or "",
            tool_calls=tuple(calls),
            finish_reason=choice.get("finish_reason"),
            usage={str(k): int(v) for k, v in usage.items() if isinstance(v, (int, float))},
            raw=data,
        )

    async def _responses(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ProviderResponse:
        # Responses accepts role/content input items. Tool outputs are represented explicitly.
        input_items: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": m.tool_call_id,
                    "output": m.content or "",
                })
            else:
                item: dict[str, Any] = {"role": m.role, "content": m.content or ""}
                input_items.append(item)
                for call in m.tool_calls:
                    input_items.append({
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": json.dumps(dict(call.arguments)),
                    })
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": input_items,
            **dict(self.config.extra_body),
        }
        response_tools = [
            {"type": "function", "name": t.name, "description": t.description, "parameters": dict(t.parameters)}
            for t in tools
        ]
        response_tools.extend(dict(item) for item in self.config.native_tools)
        if response_tools:
            payload["tools"] = response_tools
        if self.config.max_output_tokens is not None:
            payload["max_output_tokens"] = self.config.max_output_tokens
        if self.config.reasoning_effort:
            payload["reasoning"] = {"effort": self.config.reasoning_effort}
        if self.config.response_format is not None:
            payload["text"] = {"format": dict(self.config.response_format)}
        response = await self._client.post("responses", json=payload)
        response.raise_for_status()
        data = response.json()
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for item in data.get("output") or []:
            kind = item.get("type")
            if kind == "function_call":
                raw_args = item.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}
                calls.append(ToolCall(str(item.get("call_id") or item.get("id")), str(item.get("name")), args))
            elif kind == "message":
                for content in item.get("content") or []:
                    if content.get("type") in {"output_text", "text"}:
                        text_parts.append(str(content.get("text", "")))
        if not text_parts and data.get("output_text"):
            text_parts.append(str(data["output_text"]))
        usage = data.get("usage") or {}
        flat_usage = {str(k): int(v) for k, v in usage.items() if isinstance(v, (int, float))}
        return ProviderResponse("".join(text_parts), tuple(calls), data.get("status"), flat_usage, data)

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["OpenAICompatibleProvider"]
