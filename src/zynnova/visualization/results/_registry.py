from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class PlotDescriptor:
    name: str
    function: Callable[..., Any]
    category: str
    description: str
    aliases: tuple[str, ...] = ()


_REGISTRY: dict[str, PlotDescriptor] = {}
_LOADED = False
_MODULES = (
    "statistics",
    "distributions",
    "model_evaluation",
    "uncertainty",
    "explainability",
    "embeddings",
    "optimization",
    "materials",
    "atomistic",
    "electrochemistry",
    "battery",
    "phase_field",
    "multiscale",
    "biology",
    "fields",
    "networks",
    "training",
    "panels",
    "animation",
    "interactive",
)


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "-")


def register_plot(
    name: str | None = None,
    *,
    category: str = "general",
    aliases: tuple[str, ...] = (),
    description: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        key = normalize_name(name or function.__name__)
        descriptor = PlotDescriptor(
            name=key,
            function=function,
            category=category,
            description=description or (function.__doc__ or "").strip().split("\n")[0],
            aliases=tuple(normalize_name(alias) for alias in aliases),
        )
        for candidate in (key, *descriptor.aliases):
            if candidate in _REGISTRY and _REGISTRY[candidate].function is not function:
                raise KeyError(f"plot name or alias {candidate!r} already registered")
            _REGISTRY[candidate] = descriptor
        return function

    return decorator


def autoload() -> None:
    global _LOADED
    if _LOADED:
        return
    package = __package__
    for module in _MODULES:
        import_module(f"{package}.{module}")
    _LOADED = True


def get_plot(name: str) -> Callable[..., Any]:
    autoload()
    key = normalize_name(name)
    if key not in _REGISTRY:
        raise KeyError(f"unknown plot {name!r}; available={available_plots()}")
    return _REGISTRY[key].function


def plot(name: str, /, *args: Any, **kwargs: Any) -> Any:
    return get_plot(name)(*args, **kwargs)


def available_plots(*, category: str | None = None) -> tuple[str, ...]:
    autoload()
    descriptors = {descriptor.name: descriptor for descriptor in _REGISTRY.values()}
    names = [
        name
        for name, descriptor in descriptors.items()
        if category is None or descriptor.category == category
    ]
    return tuple(sorted(names))


def plot_catalog() -> tuple[PlotDescriptor, ...]:
    autoload()
    unique = {descriptor.name: descriptor for descriptor in _REGISTRY.values()}
    return tuple(unique[name] for name in sorted(unique))
