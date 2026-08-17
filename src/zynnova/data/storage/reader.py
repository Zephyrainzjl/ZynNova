from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..exceptions import StorageError
from ..record import MaterialSample
from ..source import DatasetSource
from .serialization import loads, sample_from_payload, structure_from_payload


class PreparedDataset(DatasetSource):
    name = "prepared"

    def __init__(self, path: str | Path, *, format: str | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        self.format = (format or _infer_format(self.path)).lower()
        self.root = self.path if self.path.is_dir() else self.path.parent
        self.raw_dir = self.root
        self.processed_dir = self.root
        self.options = {}

    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        iterator = _read_samples(self.path, self.format)
        for sample in iterator:
            if split is None or sample.split == split:
                yield sample


def load_dataset(path: str | Path, *, format: str | None = None) -> PreparedDataset:
    return PreparedDataset(path, format=format)


def _read_samples(path: Path, format: str) -> Iterator[MaterialSample]:
    if format in {"dir", "directory", "folder"}:
        yield from _read_directory(path)
    elif format in {"jsonl", "jsonlines"}:
        yield from _read_jsonl(path)
    elif format == "csv":
        yield from _read_csv(path)
    elif format in {"h5", "hdf5"}:
        yield from _read_hdf5(path)
    elif format == "npz":
        yield from _read_npz(path)
    else:
        raise StorageError(f"unsupported prepared dataset format: {format!r}")


def _read_directory(path: Path) -> Iterator[MaterialSample]:
    with (path / "samples.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            payload = loads(line)
            reference = payload.pop("structure_ref", None)
            if reference:
                payload["structure"] = loads((path / reference).read_text(encoding="utf-8"))
            yield sample_from_payload(payload)


def _read_jsonl(path: Path) -> Iterator[MaterialSample]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield sample_from_payload(loads(line))


def _read_csv(path: Path) -> Iterator[MaterialSample]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("features_json"):
                features = loads(row["features_json"])
                labels = loads(row.get("labels_json") or "{}")
                conditions = loads(row.get("conditions_json") or "{}")
                metadata = loads(row.get("metadata_json") or "{}")
                provenance = loads(row.get("provenance_json") or "{}")
            else:
                # Backward-compatible reader for the first flattened CSV format.
                features = {}
                labels = {}
                conditions = {}
                metadata = {}
                provenance = {}
                for key, value in row.items():
                    if key.startswith("features."):
                        features[key[9:]] = _parse_scalar(value)
                    elif key.startswith("labels."):
                        labels[key[7:]] = _parse_scalar(value)
                    elif key.startswith("conditions."):
                        conditions[key[11:]] = _parse_scalar(value)
                    elif key.startswith("metadata."):
                        metadata[key[9:]] = _parse_scalar(value)
            yield MaterialSample(
                id=row["id"],
                material_type=row["material_type"],
                structure=structure_from_payload(loads(row["structure_json"])),
                features=features,
                labels=labels,
                conditions=conditions,
                metadata=metadata,
                provenance=provenance,
                split=row.get("split") or None,
            )


def _read_hdf5(path: Path) -> Iterator[MaterialSample]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required; install zynnova[data-hdf5]") from exc
    with h5py.File(path, "r") as handle:
        for name in sorted(handle["samples"], key=int):
            group = handle["samples"][name]
            payload = loads(group.attrs["sample_json"])
            payload["structure"] = loads(group.attrs["structure_json"])
            yield sample_from_payload(payload)


def _read_npz(path: Path) -> Iterator[MaterialSample]:
    with np.load(path, allow_pickle=True) as archive:
        for payload in archive["samples"]:
            yield sample_from_payload(dict(payload))


def _parse_scalar(value: str | None) -> Any:
    if value is None or value == "":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _infer_format(path: Path) -> str:
    if path.is_dir() or not path.suffix:
        return "directory"
    return {".hdf5": "h5"}.get(path.suffix.lower(), path.suffix.lower().lstrip("."))
