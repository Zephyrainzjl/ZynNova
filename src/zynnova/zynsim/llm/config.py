"""External LLM provider configuration; no model weights or secrets are stored."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse


ProviderKind = Literal["openai_responses", "openai_compatible"]
ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    kind: ProviderKind
    model: str
    base_url: str
    api_key_env: str
    timeout_s: float = 120.0
    max_retries: int = 2
    reasoning_effort: ReasoningEffort | None = None
    reasoning_mode: Literal["standard", "pro"] = "standard"
    reasoning_context: Literal["auto", "current_turn", "all_turns"] = "current_turn"
    text_verbosity: Literal["low", "medium", "high"] = "low"
    store: bool = False
    safety_identifier_env: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    compatible_json_schema: bool = False

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.api_key_env.strip():
            raise ValueError("LLM model and API-key environment variable are required")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValueError("LLM base_url must be an absolute HTTP(S) URL")
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("remote LLM endpoints must use HTTPS")
        if self.timeout_s <= 0.0 or self.max_retries < 0:
            raise ValueError("LLM timeout/retry settings are invalid")
        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must lie in [0, 2]")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @classmethod
    def openai(
        cls,
        *,
        model: str = "gpt-5.6",
        api_key_env: str = "OPENAI_API_KEY",
        reasoning_effort: ReasoningEffort = "medium",
        reasoning_mode: Literal["standard", "pro"] = "standard",
        safety_identifier_env: str | None = None,
    ) -> ProviderConfig:
        """Current OpenAI Responses API profile.

        The ``gpt-5.6`` alias follows the latest flagship route. Pin a snapshot
        in application configuration when reproducibility is more important
        than automatically following the alias.
        """

        return cls(
            kind="openai_responses",
            model=model,
            base_url="https://api.openai.com/v1",
            api_key_env=api_key_env,
            reasoning_effort=reasoning_effort,
            reasoning_mode=reasoning_mode,
            safety_identifier_env=safety_identifier_env,
        )

    @classmethod
    def siliconflow(
        cls,
        *,
        model: str,
        api_key_env: str = "SILICONFLOW_API_KEY",
    ) -> ProviderConfig:
        """SiliconFlow's documented OpenAI-compatible Chat Completions profile."""

        return cls(
            kind="openai_compatible",
            model=model,
            base_url="https://api.siliconflow.cn/v1",
            api_key_env=api_key_env,
            temperature=0.0,
            compatible_json_schema=False,
        )


__all__ = ["ProviderConfig", "ProviderKind", "ReasoningEffort"]
