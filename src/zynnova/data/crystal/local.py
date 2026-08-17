from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from ..adapters import as_structure_data
from ..record import MaterialSample
from ..registry import DATASETS
from .base import CrystalDatasetSource


@DATASETS.register("local_crystal", aliases=("crystal_files",))
class LocalCrystalDatasetSource(CrystalDatasetSource):
    name = "local_crystal"

    def __init__(
        self,
        path: str | Path,
        *,
        pattern: str = "**/*",
        label_loader: Callable[[Path], dict[str, Any]] | None = None,
        feature_loader: Callable[[Path], dict[str, Any]] | None = None,
        root: str | Path | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.pattern = pattern
        self.label_loader = label_loader
        self.feature_loader = feature_loader
        super().__init__(root or self.path, download=False, prepare=False)

    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        supported = {".cif", ".vasp", ".poscar", ".xyz", ".traj", ".json"}
        paths = [path for path in self.path.glob(self.pattern) if path.is_file()]
        for path in sorted(paths):
            if path.suffix.lower() not in supported and path.name.upper() not in {
                "POSCAR",
                "CONTCAR",
            }:
                continue
            yield MaterialSample(
                id=str(path.relative_to(self.path)),
                material_type=self.material_type,
                structure=as_structure_data(path, kind="crystal"),
                features=self.feature_loader(path) if self.feature_loader else {},
                labels=self.label_loader(path) if self.label_loader else {},
                provenance={"dataset": self.name, "source": str(path)},
            )
