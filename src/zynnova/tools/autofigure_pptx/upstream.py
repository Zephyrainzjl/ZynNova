from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from .config import AgentConfig, UpstreamMode
from .exceptions import UpstreamAutoFigureUnavailable


@dataclass(slots=True)
class UpstreamStatus:
    available: bool
    version: str | None = None
    error: str | None = None


class UpstreamAutoFigureBridge:
    """Compatibility bridge preserving the complete upstream AutoFigure API.

    The bridge does not fork or rewrite upstream behavior.  When the MIT-
    licensed ``autofigure`` package is installed, every original argument is
    passed through unchanged, including SVG/mxGraph output, iterative judge
    refinement, PDF/Markdown extraction and image enhancement.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._module: Any | None = None
        self._agent: Any | None = None

    def status(self) -> UpstreamStatus:
        try:
            module = import_module("autofigure")
        except Exception as exc:
            return UpstreamStatus(False, error=f"{type(exc).__name__}: {exc}")
        return UpstreamStatus(True, version=getattr(module, "__version__", None))

    def _load(self) -> Any:
        if self.config.upstream_mode == UpstreamMode.DISABLED:
            raise UpstreamAutoFigureUnavailable("upstream AutoFigure is disabled")
        if self._agent is not None:
            return self._agent
        try:
            module = import_module("autofigure")
            Config = getattr(module, "Config")
            AutoFigureAgent = getattr(module, "AutoFigureAgent")
        except Exception as exc:
            raise UpstreamAutoFigureUnavailable(
                "Install AutoFigure to use the original SVG/mxGraph/enhancement pipeline: "
                "pip install -e /path/to/AutoFigure"
            ) from exc
        kwargs = dict(self.config.upstream_options)
        kwargs.setdefault("generation_api_key", self.config.generation_api_key)
        kwargs.setdefault("generation_provider", self.config.generation_provider)
        if self.config.generation_model is not None:
            kwargs.setdefault("generation_model", self.config.generation_model)
        if self.config.evaluation_api_key is not None:
            kwargs.setdefault("evaluation_api_key", self.config.evaluation_api_key)
        if self.config.evaluation_provider is not None:
            kwargs.setdefault("evaluation_provider", self.config.evaluation_provider)
        if self.config.evaluation_model is not None:
            kwargs.setdefault("evaluation_model", self.config.evaluation_model)
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        self._module = module
        self._agent = AutoFigureAgent(Config(**kwargs))
        return self._agent

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self._load().generate(*args, **kwargs)

    def generate_from_paper(self, *args: Any, **kwargs: Any) -> Any:
        return self._load().generate_from_paper(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        agent = self._load()
        return getattr(agent, name)
