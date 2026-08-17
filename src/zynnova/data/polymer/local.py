from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from ..record import MaterialSample
from ..registry import DATASETS
from .base import PolymerDatasetSource


@DATASETS.register("local_polymer", aliases=("polymer_files",))
class LocalPolymerDatasetSource(PolymerDatasetSource):
    """Read local ``.zpoly``/polymer JSON records recursively."""

    name = "local_polymer"

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
        from ...structure.polymer import load_json, load_zpoly

        for path in sorted(item for item in self.path.glob(self.pattern) if item.is_file()):
            suffix = path.suffix.lower()
            if suffix == ".zpoly":
                record = load_zpoly(path)
            elif suffix == ".json":
                try:
                    record = load_json(path)
                except (KeyError, TypeError, ValueError):
                    continue
            else:
                continue
            yield MaterialSample(
                id=str(path.relative_to(self.path)),
                material_type=self.material_type,
                structure=record,
                features=self.feature_loader(path) if self.feature_loader else {},
                labels=self.label_loader(path) if self.label_loader else {},
                provenance={"dataset": self.name, "source": str(path)},
            )
