"""Lazy, inspectable backend registry used to isolate incompatible ML stacks."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import shutil
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

from .exceptions import BackendUnavailableError


class Backend(Protocol):
    """Minimum protocol implemented by a ZynNova backend."""

    name: str

    def availability(self) -> Availability:
        """Return a side-effect-free availability report."""


@dataclass(frozen=True, slots=True)
class Availability:
    available: bool
    reason: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def require(self, backend_name: str) -> None:
        if not self.available:
            message = self.reason or "backend is unavailable"
            raise BackendUnavailableError(f"{backend_name}: {message}")


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    """Static metadata independent of heavyweight model imports."""

    name: str
    task: str
    factory: Callable[..., Backend]
    summary: str
    license_id: str = "UNKNOWN"
    source: str | None = None
    default_rank: int = 100
    extras: tuple[str, ...] = ()


B = TypeVar("B", bound=Backend)


class BackendRegistry(Generic[B]):
    """Deterministic registry with explicit duplicate and availability handling."""

    def __init__(self, task: str) -> None:
        self.task = str(task)
        self._descriptors: dict[str, BackendDescriptor] = {}

    def register(self, descriptor: BackendDescriptor, *, replace: bool = False) -> None:
        if descriptor.task != self.task:
            raise ValueError(
                f"backend task {descriptor.task!r} does not match registry {self.task!r}"
            )
        key = descriptor.name.strip().lower()
        if not key:
            raise ValueError("backend name cannot be empty")
        if key in self._descriptors and not replace:
            raise KeyError(f"backend {key!r} is already registered")
        self._descriptors[key] = descriptor

    def decorator(
        self,
        name: str,
        *,
        summary: str,
        license_id: str = "UNKNOWN",
        source: str | None = None,
        default_rank: int = 100,
        extras: tuple[str, ...] = (),
    ) -> Callable[[Callable[..., B]], Callable[..., B]]:
        def apply(factory: Callable[..., B]) -> Callable[..., B]:
            self.register(
                BackendDescriptor(
                    name=name,
                    task=self.task,
                    factory=factory,
                    summary=summary,
                    license_id=license_id,
                    source=source,
                    default_rank=default_rank,
                    extras=extras,
                )
            )
            return factory

        return apply

    def create(self, name: str, /, **kwargs: Any) -> B:
        descriptor = self.describe(name)
        backend = _call_factory(descriptor.factory, kwargs)
        backend.availability().require(descriptor.name)
        return backend  # type: ignore[return-value]

    def describe(self, name: str) -> BackendDescriptor:
        key = name.strip().lower()
        try:
            return self._descriptors[key]
        except KeyError as exc:
            choices = ", ".join(sorted(self._descriptors)) or "<none>"
            raise KeyError(f"unknown {self.task} backend {name!r}; choices: {choices}") from exc

    def choose(self, preferred: str | None = None, /, **kwargs: Any) -> B:
        if preferred is not None and preferred != "auto":
            return self.create(preferred, **kwargs)
        failures: list[str] = []
        for descriptor in sorted(
            self._descriptors.values(), key=lambda item: (item.default_rank, item.name)
        ):
            try:
                backend = _call_factory(descriptor.factory, kwargs)
            except (TypeError, ValueError) as exc:
                failures.append(f"{descriptor.name}: configuration rejected ({exc})")
                continue
            availability = backend.availability()
            if availability.available:
                return backend  # type: ignore[return-value]
            failures.append(f"{descriptor.name}: {availability.reason}")
        raise BackendUnavailableError(
            f"no {self.task} backend is available; " + "; ".join(failures)
        )

    def status(self) -> list[Mapping[str, Any]]:
        reports: list[Mapping[str, Any]] = []
        for descriptor in sorted(self._descriptors.values(), key=lambda item: item.name):
            try:
                availability = _call_factory(descriptor.factory, {}).availability()
            except Exception as exc:  # diagnostics must not crash listing
                availability = Availability(False, f"diagnostic failed: {exc}")
            reports.append(
                {
                    "name": descriptor.name,
                    "task": descriptor.task,
                    "available": availability.available,
                    "reason": availability.reason,
                    "license": descriptor.license_id,
                    "source": descriptor.source,
                    "summary": descriptor.summary,
                    "extras": descriptor.extras,
                    "details": dict(availability.details),
                }
            )
        return reports

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.strip().lower() in self._descriptors

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._descriptors))

    def __len__(self) -> int:
        return len(self._descriptors)



def _call_factory(factory: Callable[..., Backend], kwargs: Mapping[str, Any]) -> Backend:
    """Call a backend factory with only supported keyword arguments.

    Auto-selection commonly receives options intended for one heavyweight backend.
    Filtering here prevents an unrelated dependency-light fallback from failing merely
    because it does not accept those options. Factories with ``**kwargs`` still receive
    the complete mapping.
    """

    signature = inspect.signature(factory)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return factory(**dict(kwargs))
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    return factory(**{key: value for key, value in kwargs.items() if key in accepted})


def module_availability(module: str) -> Availability:
    spec = importlib.util.find_spec(module)
    if spec is None:
        return Availability(False, f"Python module {module!r} is not installed")
    return Availability(True, details={"module": module, "origin": spec.origin})


def executable_availability(executable: str) -> Availability:
    path = shutil.which(executable)
    if path is None:
        return Availability(False, f"executable {executable!r} was not found on PATH")
    return Availability(True, details={"executable": executable, "path": path})


def import_symbol(module: str, symbol: str) -> Any:
    """Import one symbol only after a backend has been selected."""

    loaded = importlib.import_module(module)
    try:
        return getattr(loaded, symbol)
    except AttributeError as exc:
        raise BackendUnavailableError(f"{module!r} has no symbol {symbol!r}") from exc


__all__ = [
    "Availability",
    "Backend",
    "BackendDescriptor",
    "BackendRegistry",
    "executable_availability",
    "import_symbol",
    "module_availability",
]
