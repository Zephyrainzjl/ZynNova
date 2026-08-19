"""Configuration for the independent ZynAstra agent framework."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """A model endpoint. Secrets are read from the environment, never persisted."""

    kind: str = "openai-compatible"
    model: str = "gpt-5.6"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    api_style: str = "auto"  # auto | responses | chat-completions
    timeout_seconds: float = 180.0
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    extra_body: Mapping[str, object] = field(default_factory=dict)
    native_tools: tuple[Mapping[str, object], ...] = ()
    response_format: Mapping[str, object] | None = None

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)

    @classmethod
    def openai(cls, model: str = "gpt-5.6", **kwargs: object) -> "ProviderConfig":
        return cls(
            kind="openai-compatible",
            model=model,
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            api_style="responses",
            **kwargs,
        )

    @classmethod
    def siliconflow(cls, model: str, **kwargs: object) -> "ProviderConfig":
        return cls(
            kind="openai-compatible",
            model=model,
            base_url="https://api.siliconflow.cn/v1",
            api_key_env="SILICONFLOW_API_KEY",
            api_style="chat-completions",
            **kwargs,
        )

    @classmethod
    def modelscope(cls, model: str, **kwargs: object) -> "ProviderConfig":
        return cls(
            kind="openai-compatible",
            model=model,
            base_url="https://api-inference.modelscope.cn/v1",
            api_key_env="MODELSCOPE_API_KEY",
            api_style="chat-completions",
            **kwargs,
        )

    @classmethod
    def litellm(
        cls, model: str, *, api_key_env: str = "LITELLM_API_KEY", base_url: str = "litellm://", **kwargs: object
    ) -> "ProviderConfig":
        return cls(kind="litellm", model=model, base_url=base_url, api_key_env=api_key_env, **kwargs)

    @classmethod
    def local(cls, model_path: str | Path, **kwargs: object) -> "ProviderConfig":
        return cls(kind="local-transformers", model=str(model_path), base_url="local://", api_key_env="", **kwargs)


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    name: str
    transport: str = "stdio"  # stdio | streamable-http
    command: str | None = None
    args: tuple[str, ...] = ()
    url: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    name: str = "ZynAstra"
    system_prompt: str = (
        "You are ZynAstra, ZynNova's scientific agent. Use tools when they improve "
        "accuracy. Never invent successful simulations, generated files, or tool results."
    )
    max_steps: int = 16
    max_tool_output_chars: int = 80_000
    session_memory: bool = True
    auto_load_skills: bool = True
    enable_zynnova_tools: bool = True
    allowed_zynnova_roots: tuple[str, ...] = (
        "core", "data", "dynamics", "geometry", "ml", "structure", "tools",
        "visualization", "zynform", "zynmorph", "zynsim", "zynvista", "zynvox",
    )
    mcp_servers: tuple[MCPServerConfig, ...] = ()

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.max_tool_output_chars < 1000:
            raise ValueError("max_tool_output_chars must be >= 1000")


__all__ = ["AgentConfig", "MCPServerConfig", "ProviderConfig"]
