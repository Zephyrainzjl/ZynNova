"""Generic, auditable bridge from the agent to public ZynNova APIs."""
from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import types
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, get_args, get_origin, get_type_hints

from ..workspace import Workspace
from .registry import ToolRegistry


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Path): return str(value)
    if isinstance(value, Enum): return value.value
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)): return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try: return value.tolist()
        except Exception: pass
    try:
        json.dumps(value); return value
    except Exception:
        return repr(value)


def _coerce(value: Any, annotation: Any) -> Any:
    """Best-effort JSON -> annotated Python value conversion for public APIs."""
    if annotation in {inspect.Signature.empty, Any, object, None}:
        return value
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {types.UnionType, getattr(__import__('typing'), 'Union', object)}:
        if value is None and type(None) in args:
            return None
        for option in args:
            if option is type(None):
                continue
            try:
                return _coerce(value, option)
            except (TypeError, ValueError, KeyError):
                pass
        return value
    if origin in {list, tuple, set}:
        item_type = args[0] if args else Any
        seq = [_coerce(item, item_type) for item in value]
        return origin(seq) if origin is not tuple else tuple(seq)
    if origin is dict:
        key_t, val_t = args if len(args) == 2 else (Any, Any)
        return {_coerce(k, key_t): _coerce(v, val_t) for k, v in value.items()}
    if annotation is Path:
        return Path(value)
    try:
        if inspect.isclass(annotation) and issubclass(annotation, Enum):
            return annotation(value)
    except TypeError:
        pass
    if inspect.isclass(annotation) and dataclasses.is_dataclass(annotation) and isinstance(value, dict):
        hints = get_type_hints(annotation)
        fields = {field.name: field for field in dataclasses.fields(annotation)}
        kwargs = {}
        for key, item in value.items():
            if key not in fields:
                raise TypeError(f"unknown field {key!r} for {annotation.__name__}")
            kwargs[key] = _coerce(item, hints.get(key, fields[key].type))
        return annotation(**kwargs)
    return value


def _coerce_call(fn: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}
    signature = inspect.signature(fn)
    out = {}
    for key, value in arguments.items():
        parameter = signature.parameters.get(key)
        if parameter is None:
            out[key] = value
            continue
        out[key] = _coerce(value, hints.get(key, parameter.annotation))
    return out

def _resolve_public_callable(dotted: str, allowed_roots: Iterable[str]):
    dotted = dotted.strip()
    if dotted.startswith("zynnova."):
        dotted = dotted[len("zynnova."):]
    parts = dotted.split(".")
    if len(parts) < 2 or parts[0] not in set(allowed_roots):
        raise PermissionError(f"callable must be under an allowed zynnova root: {sorted(allowed_roots)}")
    if any(part.startswith("_") for part in parts):
        raise PermissionError("private ZynNova attributes are not agent-callable")
    # Find the longest importable module prefix, then traverse public attributes.
    module = None
    split_at = 0
    for i in range(len(parts), 0, -1):
        try:
            module = importlib.import_module("zynnova." + ".".join(parts[:i]))
            split_at = i
            break
        except ModuleNotFoundError:
            continue
    if module is None:
        raise ImportError(f"unable to import zynnova module for {dotted!r}")
    obj: Any = module
    for attr in parts[split_at:]:
        obj = getattr(obj, attr)
    if not callable(obj):
        raise TypeError(f"{dotted!r} is not callable")
    return obj


def install_zynnova_tools(registry: ToolRegistry, workspace: Workspace, allowed_roots: tuple[str, ...]) -> None:
    def list_api(namespace: str = "") -> dict[str, Any]:
        root = namespace.strip().removeprefix("zynnova.")
        if root and root.split(".", 1)[0] not in allowed_roots:
            raise PermissionError(f"namespace not allowed: {root}")
        module = importlib.import_module("zynnova" + ("." + root if root else ""))
        names = []
        for name in getattr(module, "__all__", dir(module)):
            if str(name).startswith("_"): continue
            try: value = getattr(module, name)
            except Exception: continue
            if callable(value):
                try: sig = str(inspect.signature(value))
                except Exception: sig = "(...)"
                names.append({"name": name, "signature": sig, "doc": (inspect.getdoc(value) or "").split("\n", 1)[0]})
        return {"namespace": "zynnova" + ("." + root if root else ""), "callables": names[:500]}

    def call_api(callable: str, arguments: dict[str, Any] | None = None) -> Any:
        fn = _resolve_public_callable(callable, allowed_roots)
        kwargs = _coerce_call(fn, dict(arguments or {}))
        result = fn(**kwargs)
        return _jsonable(result)

    registry.add(
        "zynnova_list_api", list_api,
        description="List public callable APIs in a ZynNova namespace before invoking scientific workflows.",
        parameters={"type":"object","properties":{"namespace":{"type":"string"}},"additionalProperties":False},
    )
    registry.add(
        "zynnova_call", call_api,
        description="Call a public ZynNova function. Use dotted paths such as zynvista.run_scene or zynform.run_object. Arguments must match the Python API; construct dataclass inputs in Python code or use higher-level wrapper tools when needed.",
        parameters={
            "type":"object",
            "properties":{
                "callable":{"type":"string"},
                "arguments":{"type":"object"},
            },
            "required":["callable"],
            "additionalProperties":False,
        },
    )
    registry.add(
        "workspace_path", lambda category="artifacts": str(getattr(workspace, category)),
        description="Return an external ZynNova workspace path (models, finetunes, runs, skills, cache, artifacts, memory, mcp).",
        parameters={"type":"object","properties":{"category":{"type":"string"}},"additionalProperties":False},
    )


__all__ = ["install_zynnova_tools"]
