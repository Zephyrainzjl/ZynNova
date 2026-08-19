"""Provider construction."""
from __future__ import annotations

from ..config import ProviderConfig
from .base import ModelProvider


def create_provider(config: ProviderConfig) -> ModelProvider:
    kind = config.kind.strip().lower()
    if kind in {"openai", "openai-compatible", "http"}:
        from .openai_compat import OpenAICompatibleProvider
        return OpenAICompatibleProvider(config)
    if kind == "litellm":
        from .litellm_provider import LiteLLMProvider
        return LiteLLMProvider(config)
    if kind in {"local", "local-transformers", "transformers"}:
        from .local_transformers import LocalTransformersProvider
        return LocalTransformersProvider(config)
    raise ValueError(f"unknown model provider kind: {config.kind!r}")


__all__ = ["create_provider"]
