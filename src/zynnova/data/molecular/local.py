from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from ..adapters import as_structure_data
from ..record import MaterialSample
from ..registry import DATASETS
from .base import MolecularDatasetSource


@DATASETS.register("local_molecular", aliases=("molecule_files", "local_molecule"))
class LocalMolecularDatasetSource(MolecularDatasetSource):
    name = "local_molecular"

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
        supported = {".xyz", ".sdf", ".mol", ".mol2", ".pdb", ".traj"}
        candidates = (
            candidate for candidate in self.path.glob(self.pattern) if candidate.is_file()
        )
        for path in sorted(candidates):
            if path.suffix.lower() not in supported:
                continue
            yield MaterialSample(
                id=str(path.relative_to(self.path)),
                material_type=self.material_type,
                structure=as_structure_data(path, kind="molecular"),
                features=self.feature_loader(path) if self.feature_loader else {},
                labels=self.label_loader(path) if self.label_loader else {},
                provenance={"dataset": self.name, "source": str(path)},
            )
