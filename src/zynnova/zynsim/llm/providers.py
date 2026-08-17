"""Minimal Responses API and OpenAI-compatible structured-output clients."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from ..exceptions import ConfigurationError, LLMProtocolError
from .config import ProviderConfig


class JSONTransport(Protocol):
    def __call__(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_s: float,
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class StructuredLLMResponse:
    data: Mapping[str, object]
    provider: str
    model: str
    response_id: str | None
    usage: Mapping[str, object] = field(default_factory=dict)
    raw_status: str | None = None


class StructuredProvider(Protocol):
    def generate(
        self,
        *,
        system: str,
        user: str,
        schema_name: str,
        schema: Mapping[str, object],
    ) -> StructuredLLMResponse: ...


class HTTPStatusError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = int(status_code)

    @property
    def retryable(self) -> bool:
        return self.status_code == 429 or self.status_code >= 500


def httpx_transport(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, object],
    timeout_s: float,
) -> Mapping[str, object]:
    try:
        import httpx
    except ImportError as exc:
        raise ImportError("LLM HTTP calls require zynnova[llm] (httpx)") from exc
    response = httpx.post(url, headers=dict(headers), json=dict(payload), timeout=timeout_s)
    if response.status_code >= 400:
        request_id = response.headers.get("x-request-id", "<unknown>")
        raise HTTPStatusError(
            response.status_code,
            f"LLM endpoint returned HTTP {response.status_code}; request_id={request_id}",
        )
    value = response.json()
    if not isinstance(value, dict):
        raise LLMProtocolError("LLM endpoint returned a non-object JSON payload")
    return value


@dataclass(slots=True)
class _BaseProvider:
    config: ProviderConfig
    transport: JSONTransport = httpx_transport
    api_key_getter: Callable[[str], str | None] = os.environ.get

    def _post(self, path: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        api_key = self.api_key_getter(self.config.api_key_env)
        if not api_key:
            raise ConfigurationError(
                f"missing API key environment variable {self.config.api_key_env!r}"
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return self.transport(
                    f"{self.config.base_url}/{path.lstrip('/')}",
                    headers,
                    payload,
                    self.config.timeout_s,
                )
            except HTTPStatusError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self.config.max_retries:
                    raise
            except (TimeoutError, ConnectionError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    raise
            time.sleep(min(0.25 * 2**attempt, 2.0))
        assert last_error is not None
        raise last_error


@dataclass(slots=True)
class OpenAIResponsesProvider(_BaseProvider):
    """Schema-constrained client for ``POST /v1/responses``."""

    def __post_init__(self) -> None:
        if self.config.kind != "openai_responses":
            raise ValueError("OpenAIResponsesProvider needs kind='openai_responses'")

    def generate(
        self,
        *,
        system: str,
        user: str,
        schema_name: str,
        schema: Mapping[str, object],
    ) -> StructuredLLMResponse:
        format_payload: dict[str, object] = {
            "type": "json_schema",
            "name": schema_name,
            "strict": True,
            "schema": dict(schema),
        }
        payload: dict[str, object] = {
            "model": self.config.model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "store": self.config.store,
            "text": {
                "format": format_payload,
                "verbosity": self.config.text_verbosity,
            },
        }
        if self.config.reasoning_effort is not None:
            reasoning: dict[str, object] = {
                "effort": self.config.reasoning_effort,
                "context": self.config.reasoning_context,
            }
            if self.config.reasoning_mode == "pro":
                reasoning["mode"] = "pro"
            payload["reasoning"] = reasoning
        if self.config.max_output_tokens is not None:
            payload["max_output_tokens"] = self.config.max_output_tokens
        if self.config.safety_identifier_env:
            identifier = self.api_key_getter(self.config.safety_identifier_env)
            if identifier:
                payload["safety_identifier"] = identifier
        raw = self._post("responses", payload)
        text = _responses_output_text(raw)
        data = _parse_json_object(text)
        return StructuredLLMResponse(
            data=data,
            provider="openai",
            model=str(raw.get("model", self.config.model)),
            response_id=_optional_string(raw.get("id")),
            usage=_mapping(raw.get("usage")),
            raw_status=_optional_string(raw.get("status")),
        )


@dataclass(slots=True)
class OpenAICompatibleProvider(_BaseProvider):
    """Chat Completions adapter for SiliconFlow and compatible public APIs.

    ``response_format={"type": "json_object"}`` guarantees JSON syntax on
    many compatible endpoints, but it does not guarantee conformance to the
    requested schema.  The adapter therefore validates the returned object
    locally and asks the model to repair a protocol-invalid response before
    exposing it to the orchestrator.
    """

    def __post_init__(self) -> None:
        if self.config.kind != "openai_compatible":
            raise ValueError("OpenAICompatibleProvider needs kind='openai_compatible'")

    def generate(
        self,
        *,
        system: str,
        user: str,
        schema_name: str,
        schema: Mapping[str, object],
    ) -> StructuredLLMResponse:
        required = schema.get("required", [])
        required_keys = (
            ", ".join(str(item) for item in required)
            if isinstance(required, list)
            else "<defined by the schema>"
        )
        schema_instruction = (
            "\nReturn one new JSON object conforming exactly to the JSON Schema "
            f"named {schema_name!r}. Do not echo the user's input envelope. "
            "Do not wrap the object in Markdown or add explanatory text. "
            f"The required top-level keys are: {required_keys}.\nJSON Schema:\n"
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system + schema_instruction},
            {"role": "user", "content": user},
        ]
        last_error: LLMProtocolError | None = None

        for attempt in range(self.config.max_retries + 1):
            payload: dict[str, object] = {
                "model": self.config.model,
                "messages": messages,
                "stream": False,
            }
            if self.config.compatible_json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": dict(schema),
                    },
                }
            else:
                payload["response_format"] = {"type": "json_object"}
            if self.config.temperature is not None:
                payload["temperature"] = self.config.temperature
            if self.config.max_output_tokens is not None:
                payload["max_tokens"] = self.config.max_output_tokens

            raw = self._post("chat/completions", payload)
            text = ""
            try:
                text = _compatible_output_text(raw)
                data = _parse_json_object(text)
                _validate_json_schema(data, schema)
            except LLMProtocolError as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    raise LLMProtocolError(
                        "compatible endpoint failed structured-output validation "
                        f"after {attempt + 1} attempts: {exc}"
                    ) from exc
                messages = [
                    *messages[:2],
                    {"role": "assistant", "content": _response_preview(text)},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response is protocol-invalid: "
                            f"{exc}. Discard it and generate a corrected JSON object "
                            "from scratch. Do not echo the input object. The output "
                            f"must have exactly these top-level keys: {required_keys}."
                        ),
                    },
                ]
                continue

            return StructuredLLMResponse(
                data=data,
                provider="openai-compatible",
                model=str(raw.get("model", self.config.model)),
                response_id=_optional_string(raw.get("id")),
                usage=_mapping(raw.get("usage")),
            )

        assert last_error is not None
        raise last_error


def create_provider(
    config: ProviderConfig,
    *,
    transport: JSONTransport = httpx_transport,
    api_key_getter: Callable[[str], str | None] = os.environ.get,
) -> StructuredProvider:
    if config.kind == "openai_responses":
        return OpenAIResponsesProvider(config, transport, api_key_getter)
    return OpenAICompatibleProvider(config, transport, api_key_getter)


def _responses_output_text(raw: Mapping[str, object]) -> str:
    if isinstance(raw.get("output_text"), str):
        return str(raw["output_text"])
    output = raw.get("output")
    if not isinstance(output, list):
        raise LLMProtocolError("Responses payload omitted its typed output array")
    texts: list[str] = []
    refusals: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(str(part["text"]))
            elif part.get("type") == "refusal":
                refusals.append(str(part.get("refusal", "request refused")))
    if refusals:
        raise LLMProtocolError("LLM refused the structured request: " + "; ".join(refusals))
    if not texts:
        raise LLMProtocolError("Responses payload contained no output_text item")
    return "".join(texts)


def _parse_json_object(text: str) -> Mapping[str, object]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMProtocolError(f"LLM returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMProtocolError("LLM structured output must be a JSON object")
    return value


def _compatible_output_text(raw: Mapping[str, object]) -> str:
    try:
        choices = raw["choices"]
        message = choices[0]["message"]  # type: ignore[index]
        text = message["content"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMProtocolError(
            "compatible endpoint omitted choices[0].message.content"
        ) from exc
    if not isinstance(text, str):
        raise LLMProtocolError("compatible endpoint content is not text")
    return text


def _response_preview(text: str, limit: int = 4000) -> str:
    if not text:
        return "<no valid text content>"
    if len(text) <= limit:
        return text
    return text[:limit] + "\n<truncated>"


def _validate_json_schema(
    value: object,
    schema: Mapping[str, object],
    *,
    path: str = "$",
) -> None:
    """Validate the JSON-Schema subset used by ZynSim structured protocols."""

    expected = schema.get("type")
    if isinstance(expected, str) and not _matches_json_type(value, expected):
        raise LLMProtocolError(
            f"structured output schema mismatch at {path}: expected {expected}"
        )

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        raise LLMProtocolError(
            f"structured output schema mismatch at {path}: value is not in enum"
        )

    if expected == "object":
        if not isinstance(value, Mapping):
            return
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            raise ValueError(f"invalid object schema at {path}")
        missing = [str(name) for name in required if name not in value]
        additional = schema.get("additionalProperties", True)
        unknown = sorted(str(name) for name in set(value) - set(properties))
        key_errors: list[str] = []
        if missing:
            key_errors.append(f"missing required keys {missing}")
        if additional is False and unknown:
            key_errors.append(f"unexpected keys {unknown}")
        if key_errors:
            raise LLMProtocolError(
                f"structured output schema mismatch at {path}: " + "; ".join(key_errors)
            )
        for name, item in value.items():
            definition = properties.get(name)
            if isinstance(definition, Mapping):
                _validate_json_schema(item, definition, path=f"{path}.{name}")
            elif isinstance(additional, Mapping):
                _validate_json_schema(item, additional, path=f"{path}.{name}")

    if expected == "array":
        if not isinstance(value, list):
            return
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                _validate_json_schema(item, items, path=f"{path}[{index}]")


def _matches_json_type(value: object, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ValueError(f"unsupported JSON Schema type {expected!r}")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, dict) else {}


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


__all__ = [
    "HTTPStatusError",
    "JSONTransport",
    "OpenAICompatibleProvider",
    "OpenAIResponsesProvider",
    "StructuredLLMResponse",
    "StructuredProvider",
    "create_provider",
    "httpx_transport",
]
