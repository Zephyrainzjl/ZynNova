from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class TrainingResult:
    run_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    history: list[dict[str, float]]
    best_metric: float
    model: Any


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required; install zynnova[ml]") from exc
    return torch


def resolve_device(device: str = "auto"):
    torch = require_torch()
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def move_to_device(value: Any, device: Any) -> Any:
    torch = require_torch()
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    torch = require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(path: str | Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    torch = require_torch()
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def config_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return _jsonable(asdict(config))
    if isinstance(config, dict):
        return _jsonable(config)
    raise TypeError("config must be a dataclass or dictionary")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


__all__ = [
    "TrainingResult",
    "append_jsonl",
    "config_dict",
    "load_checkpoint",
    "move_to_device",
    "require_torch",
    "resolve_device",
    "save_checkpoint",
    "seed_everything",
]
