from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..adapters import smiles_to_structure
from ..local_input import find_local_files, iter_records, load_split_mapping
from ..record import MaterialSample
from ..registry import DATASETS
from .base import MolecularDatasetSource


@DATASETS.register("pcqm4mv2", aliases=("pcqm4m_v2",))
class PCQM4Mv2DatasetSource(MolecularDatasetSource):
    name = "pcqm4mv2"
    homepage = "https://ogb.stanford.edu/docs/lsc/pcqm4mv2/"
    license = "CC BY 4.0"

    def __init__(
        self,
        root: str | Path,
        *,
        embed_3d: bool = False,
        include_structure: bool = True,
        limit: int | None = None,
        local_file: str | Path | None = None,
        local_dir: str | Path | None = None,
        split_file: str | Path | None = None,
        smiles_column: str | None = None,
        target_column: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.embed_3d = embed_3d
        self.include_structure = include_structure
        self.limit = limit
        self.split_file = (
            None if split_file is None else Path(split_file).expanduser().resolve()
        )
        self.smiles_column = smiles_column
        self.target_column = target_column
        self._local_root: Path | None = None
        super().__init__(
            root,
            local_file=local_file,
            local_dir=local_dir,
            **kwargs,
        )

    def download(self, *, force: bool = False) -> None:
        local = self.materialize_local_input(
            force=force,
            extract_subdir="pcqm4mv2-local",
        )
        if local is not None:
            self._local_root = local

    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        if self._local_root is not None:
            yield from self._iter_local(split)
            return
        try:
            from ogb.lsc import PCQM4Mv2Dataset
        except ImportError as exc:
            raise ImportError(
                "ogb is required for automatic PCQM4Mv2 download; "
                "install zynnova[data-molecular] or pass local_file/local_dir"
            ) from exc
        dataset = PCQM4Mv2Dataset(root=str(self.raw_dir), only_smiles=True)
        split_indices = dataset.get_idx_split()
        split_map = {
            int(index): name
            for name, indices in split_indices.items()
            for index in indices
        }
        count = 0
        for index in range(len(dataset)):
            sample_split = split_map.get(index)
            if split is not None and sample_split != split:
                continue
            if self.limit is not None and count >= self.limit:
                break
            smiles, gap = dataset[index]
            yield self._make_sample(index, smiles, gap, sample_split)
            count += 1

    def _iter_local(self, split: str | None) -> Iterator[MaterialSample]:
        table = _local_table(self._local_root)
        split_path = self.split_file or _discover_split_file(self._local_root)
        split_map = load_split_mapping(split_path)
        count = 0
        for row_index, row in enumerate(iter_records(table)):
            index = int(row.get("index", row.get("idx", row_index)))
            smiles_name = self.smiles_column or _first_key(
                row,
                ("smiles", "SMILES", "molecule"),
            )
            target_name = self.target_column or _first_key(
                row,
                ("homolumogap", "homo_lumo_gap", "gap", "target"),
            )
            if smiles_name is None:
                raise ValueError(f"PCQM4Mv2 local table {table} has no SMILES column")
            smiles = str(row[smiles_name])
            gap = _optional_float(row.get(target_name)) if target_name else None
            sample_split = (
                str(row["split"])
                if row.get("split") not in {None, ""}
                else split_map.get(index)
            )
            if split is not None and sample_split != split:
                continue
            if self.limit is not None and count >= self.limit:
                break
            yield self._make_sample(index, smiles, gap, sample_split)
            count += 1

    def _make_sample(
        self,
        index: int,
        smiles: str,
        gap: Any,
        sample_split: str | None,
    ) -> MaterialSample:
        structure = (
            smiles_to_structure(smiles, embed_3d=self.embed_3d, seed=index)
            if self.include_structure
            else None
        )
        label = _optional_float(gap)
        return MaterialSample(
            id=f"pcqm4mv2:{index}",
            material_type=self.material_type,
            structure=structure,
            labels={"homo_lumo_gap": label},
            metadata={"smiles": smiles},
            provenance={"dataset": "PCQM4Mv2", "index": index},
            split=sample_split,
        )


def _local_table(root: Path) -> Path:
    patterns = (
        "data.csv.gz",
        "data.csv",
        "*.parquet",
        "*.pq",
        "*.jsonl",
        "*.json",
        "*.csv.gz",
        "*.csv",
    )
    candidates = find_local_files(root, patterns)
    if not candidates:
        raise FileNotFoundError(
            "PCQM4Mv2 local input must contain data.csv.gz, CSV, JSONL, JSON, "
            f"or Parquet data: {root}"
        )
    return candidates[0]


def _discover_split_file(root: Path) -> Path | None:
    patterns = (
        "split_dict.pt",
        "split_dict.npz",
        "split.npz",
        "split.json",
        "splits.json",
        "split.csv",
    )
    candidates = find_local_files(root, patterns)
    return candidates[0] if candidates else None


def _first_key(mapping: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    lower = {str(key).lower(): str(key) for key in mapping}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return None if number != number else number
