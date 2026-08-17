from __future__ import annotations

import json
import os
import platform
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _default_workspace_root() -> Path:
    override = os.environ.get("ZYNNOVA_WORKSPACE")
    if override:
        return Path(override).expanduser()
    system = platform.system().lower()
    if system == "windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif system == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "zynnova"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


@dataclass(frozen=True, slots=True)
class RunPaths:
    root: Path
    checkpoints: Path
    logs: Path
    samples: Path
    exports: Path
    config: Path
    history: Path


class MLWorkspace:
    """External storage for datasets, runs, checkpoints and exported models.

    Nothing is created inside the installed :mod:`zynnova` package. The root can
    be selected explicitly or through ``ZYNNOVA_WORKSPACE``. The default follows
    the operating system's user-data location.
    """

    def __init__(self, root: str | Path | None = None, *, create: bool = True) -> None:
        self.root = Path(root).expanduser() if root is not None else _default_workspace_root()
        self.root = self.root.resolve()
        package_dir = Path(__file__).resolve().parents[1]
        if _is_relative_to(self.root, package_dir):
            raise ValueError(
                "ML workspace cannot be inside the installed zynnova package; "
                "choose an external directory"
            )
        if create:
            for directory in (
                self.root,
                self.datasets_dir,
                self.runs_dir,
                self.exports_dir,
                self.cache_dir,
            ):
                directory.mkdir(parents=True, exist_ok=True)

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    def dataset_dir(self, name: str) -> Path:
        path = self.datasets_dir / _safe_component(name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def create_run(
        self,
        category: str,
        model: str,
        *,
        name: str | None = None,
        config: Any | None = None,
    ) -> RunPaths:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = _safe_component(name or f"{timestamp}-{uuid.uuid4().hex[:8]}")
        root = self.runs_dir / _safe_component(category) / _safe_component(model) / run_name
        root.mkdir(parents=True, exist_ok=False)
        checkpoints = root / "checkpoints"
        logs = root / "logs"
        samples = root / "samples"
        exports = root / "exports"
        for directory in (checkpoints, logs, samples, exports):
            directory.mkdir(parents=True, exist_ok=True)
        config_path = root / "config.json"
        history_path = logs / "history.jsonl"
        metadata = {
            "category": category,
            "model": model,
            "name": run_name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": _jsonable(config) if config is not None else None,
        }
        config_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return RunPaths(
            root=root,
            checkpoints=checkpoints,
            logs=logs,
            samples=samples,
            exports=exports,
            config=config_path,
            history=history_path,
        )


def _safe_component(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value)
    cleaned = cleaned.strip("-.")
    if not cleaned:
        raise ValueError("workspace path component cannot be empty")
    return cleaned


__all__ = ["MLWorkspace", "RunPaths"]
