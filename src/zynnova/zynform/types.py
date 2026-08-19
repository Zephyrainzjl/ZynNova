"""Normalized object-backend outputs and final workflow results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from ..geometry import TriangleMesh, VolumeMesh


@dataclass(frozen=True, slots=True)
class ObjectBackendOutput:
    backend: str
    mesh: TriangleMesh | None = None
    native_mesh: Path | None = None
    preview: Path | None = None
    auxiliary_assets: Mapping[str, Path] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        native = None if self.native_mesh is None else Path(self.native_mesh)
        preview = None if self.preview is None else Path(self.preview)
        assets = {str(key): Path(value) for key, value in self.auxiliary_assets.items()}
        if self.mesh is None and native is None:
            raise ValueError("object backend must return a parsed mesh or native mesh file")
        for path in (native, preview, *assets.values()):
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)
        object.__setattr__(self, "native_mesh", native)
        object.__setattr__(self, "preview", preview)
        object.__setattr__(self, "auxiliary_assets", assets)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class ObjectResult:
    surface_mesh: TriangleMesh
    volume_mesh: VolumeMesh | None
    run_directory: Path
    exported_surface_files: tuple[Path, ...]
    exported_volume_files: tuple[Path, ...]
    manifest_path: Path
    fem_surface_mesh: TriangleMesh | None = None
    exported_fem_surface_files: tuple[Path, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


__all__ = ["ObjectBackendOutput", "ObjectResult"]
