"""External workspace layout. Heavy assets never live inside the installed package."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Workspace:
    root: Path

    def __init__(self, root: str | Path | None = None) -> None:
        value = root or os.environ.get("ZYNNOVA_WORKSPACE") or "~/.zynnova"
        object.__setattr__(self, "root", Path(value).expanduser().resolve())

    @property
    def models(self) -> Path: return self.root / "models"
    @property
    def finetunes(self) -> Path: return self.root / "finetunes"
    @property
    def runs(self) -> Path: return self.root / "runs"
    @property
    def skills(self) -> Path: return self.root / "skills"
    @property
    def cache(self) -> Path: return self.root / "cache"
    @property
    def artifacts(self) -> Path: return self.root / "artifacts"
    @property
    def memory(self) -> Path: return self.root / "memory"
    @property
    def mcp(self) -> Path: return self.root / "mcp"

    def ensure(self) -> "Workspace":
        for path in (
            self.root, self.models, self.finetunes, self.runs, self.skills,
            self.cache, self.artifacts, self.memory, self.mcp,
        ):
            path.mkdir(parents=True, exist_ok=True)
        meta = self.root / "workspace.json"
        if not meta.exists():
            meta.write_text(json.dumps({"format": 1, "package": "zynnova"}, indent=2), encoding="utf-8")
        return self

    def assert_external_to(self, package_path: str | Path) -> None:
        pkg = Path(package_path).resolve()
        try:
            self.root.relative_to(pkg)
        except ValueError:
            return
        raise ValueError(f"workspace must be outside installed package: {pkg}")


__all__ = ["Workspace"]
