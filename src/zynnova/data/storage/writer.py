from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ..exceptions import StorageError
from ..record import MaterialSample
from ..utils import atomic_write_json
from .serialization import dumps, sample_to_payload, safe_sample_name, structure_to_payload


def save_dataset(
    samples: Iterable[MaterialSample],
    path: str | Path,
    *,
    format: str | None = None,
    overwrite: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path).expanduser().resolve()
    selected = (format or _infer_format(destination)).lower()
    if selected in {"dir", "directory", "folder"}:
        return _write_directory(samples, destination, overwrite=overwrite, metadata=metadata)
    if selected in {"jsonl", "jsonlines"}:
        return _write_jsonl(samples, destination, overwrite=overwrite)
    if selected == "csv":
        return _write_csv(samples, destination, overwrite=overwrite)
    if selected in {"h5", "hdf5"}:
        return _write_hdf5(samples, destination, overwrite=overwrite, metadata=metadata)
    if selected == "npz":
        return _write_npz(samples, destination, overwrite=overwrite)
    raise StorageError(f"unsupported dataset format: {selected!r}")


def _write_directory(
    samples: Iterable[MaterialSample],
    destination: Path,
    *,
    overwrite: bool,
    metadata: dict[str, Any] | None,
) -> Path:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(destination)
        shutil.rmtree(destination)
    structures = destination / "structures"
    structures.mkdir(parents=True)
    rows_path = destination / "samples.jsonl"
    count = 0
    with rows_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = sample_to_payload(sample, include_structure=False)
            if sample.structure is not None:
                structure_name = f"{count:09d}-{safe_sample_name(sample.id)}.json"
                structure_path = structures / structure_name
                structure_path.write_text(
                    dumps(structure_to_payload(sample.structure)),
                    encoding="utf-8",
                )
                payload["structure_ref"] = f"structures/{structure_name}"
            handle.write(dumps(payload) + "\n")
            count += 1
    atomic_write_json(
        destination / "manifest.json",
        {
            "format": "zynnova-directory-v1",
            "count": count,
            "samples": "samples.jsonl",
            "metadata": metadata or {},
        },
    )
    return destination


def _write_jsonl(samples: Iterable[MaterialSample], destination: Path, *, overwrite: bool) -> Path:
    _prepare_file(destination, overwrite)
    with destination.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(dumps(sample_to_payload(sample)) + "\n")
    return destination


def _write_csv(samples: Iterable[MaterialSample], destination: Path, *, overwrite: bool) -> Path:
    """Write a portable CSV with complete mappings encoded as JSON columns.

    Scalar fields are additionally flattened for inspection in ordinary table
    tools, while arrays and nested fields remain losslessly available in the
    ``*_json`` columns.
    """
    rows: list[dict[str, Any]] = []
    fields: set[str] = {
        "id",
        "material_type",
        "split",
        "structure_json",
        "features_json",
        "labels_json",
        "conditions_json",
        "metadata_json",
        "provenance_json",
    }
    for sample in samples:
        row: dict[str, Any] = {
            "id": sample.id,
            "material_type": sample.material_type.value,
            "split": sample.split,
            "structure_json": dumps(structure_to_payload(sample.structure)),
            "features_json": dumps(sample.features),
            "labels_json": dumps(sample.labels),
            "conditions_json": dumps(sample.conditions),
            "metadata_json": dumps(sample.metadata),
            "provenance_json": dumps(sample.provenance),
        }
        for name, value in sample.iter_scalar_fields():
            row[name] = value
            fields.add(name)
        rows.append(row)
    _prepare_file(destination, overwrite)
    leading = [
        "id",
        "material_type",
        "split",
        "structure_json",
        "features_json",
        "labels_json",
        "conditions_json",
        "metadata_json",
        "provenance_json",
    ]
    fieldnames = [*leading, *sorted(fields - set(leading))]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _write_npz(samples: Iterable[MaterialSample], destination: Path, *, overwrite: bool) -> Path:
    _prepare_file(destination, overwrite)
    payloads = np.asarray([sample_to_payload(sample) for sample in samples], dtype=object)
    np.savez_compressed(destination, samples=payloads)
    return destination


def _write_hdf5(
    samples: Iterable[MaterialSample],
    destination: Path,
    *,
    overwrite: bool,
    metadata: dict[str, Any] | None,
) -> Path:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required; install zynnova[data-hdf5]") from exc
    _prepare_file(destination, overwrite)
    with h5py.File(destination, "w") as handle:
        handle.attrs["format"] = "zynnova-hdf5-v1"
        handle.attrs["metadata_json"] = dumps(metadata or {})
        group = handle.create_group("samples")
        count = 0
        for count, sample in enumerate(samples, start=1):
            item = group.create_group(f"{count - 1:09d}")
            payload = sample_to_payload(sample, include_structure=False)
            item.attrs["sample_json"] = dumps(payload)
            structure = structure_to_payload(sample.structure)
            item.attrs["structure_json"] = dumps(structure)
        handle.attrs["count"] = count
    return destination


def _prepare_file(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    if path.exists():
        path.unlink()


def _infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if not suffix:
        return "directory"
    return {".jsonl": "jsonl", ".csv": "csv", ".h5": "h5", ".hdf5": "h5", ".npz": "npz"}.get(
        suffix,
        suffix.lstrip("."),
    )
