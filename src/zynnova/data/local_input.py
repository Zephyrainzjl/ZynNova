from __future__ import annotations

import bz2
import csv
import gzip
import json
import lzma
import pickle
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .download import DownloadManager, DownloadSpec

_ARCHIVE_SUFFIXES = (
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".tgz",
    ".zip",
    ".tar",
)
_COMPRESSED_SUFFIXES = (".gz", ".bz2", ".xz")
_TABLE_SUFFIXES = (
    ".json",
    ".jsonl",
    ".ndjson",
    ".csv",
    ".parquet",
    ".pq",
    ".pkl",
    ".pickle",
    ".json.gz",
    ".jsonl.gz",
    ".csv.gz",
    ".json.bz2",
    ".jsonl.bz2",
    ".csv.bz2",
    ".json.xz",
    ".jsonl.xz",
    ".csv.xz",
)


@dataclass(frozen=True, slots=True)
class LocalDatasetInput:
    """Validated local file or directory supplied to a dataset plugin.

    Local inputs always take precedence over network download. Archives are
    extracted into the dataset's external ``raw`` cache, while ordinary files and
    directories are read in place.
    """

    path: Path

    @classmethod
    def create(
        cls,
        *,
        local_file: str | Path | None = None,
        local_dir: str | Path | None = None,
    ) -> LocalDatasetInput | None:
        if local_file is not None and local_dir is not None:
            raise ValueError("local_file and local_dir are mutually exclusive")
        value = local_file if local_file is not None else local_dir
        if value is None:
            return None
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"local dataset input does not exist: {path}")
        if local_file is not None and not path.is_file():
            raise ValueError(f"local_file must point to a file: {path}")
        if local_dir is not None and not path.is_dir():
            raise ValueError(f"local_dir must point to a directory: {path}")
        return cls(path)

    @property
    def is_archive(self) -> bool:
        name = self.path.name.lower()
        return any(name.endswith(suffix) for suffix in _ARCHIVE_SUFFIXES)

    def materialize(
        self,
        manager: DownloadManager,
        *,
        force: bool = False,
        extract_subdir: str = "local-input",
    ) -> Path:
        if not self.is_archive:
            return self.path
        spec = DownloadSpec(
            urls=(self.path.as_uri(),),
            filename=self.path.name,
            extract=True,
            extract_subdir=extract_subdir,
            description="user-supplied local dataset archive",
        )
        return manager.extract(self.path, spec=spec, force=force)


def find_local_files(
    root: str | Path,
    patterns: Sequence[str],
    *,
    recursive: bool = True,
) -> list[Path]:
    path = Path(root)
    if path.is_file():
        return [path] if _matches(path, patterns) else []
    files: list[Path] = []
    for pattern in patterns:
        iterator = path.rglob(pattern) if recursive else path.glob(pattern)
        files.extend(candidate for candidate in iterator if candidate.is_file())
    return sorted(set(files))


def require_local_file(
    root: str | Path,
    patterns: Sequence[str],
    *,
    description: str,
) -> Path:
    candidates = find_local_files(root, patterns)
    if not candidates:
        joined = ", ".join(patterns)
        raise FileNotFoundError(
            f"no {description} found in {Path(root)}; expected one of: {joined}"
        )
    return candidates[0]


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read common downloaded table/export formats without network access."""

    source = Path(path)
    if source.is_dir():
        files = find_local_files(source, tuple(f"*{suffix}" for suffix in _TABLE_SUFFIXES))
        if not files:
            raise FileNotFoundError(f"no supported table files found in {source}")
        for file_path in files:
            yield from iter_records(file_path)
        return

    suffix = _logical_suffix(source)
    if suffix in {".jsonl", ".ndjson"}:
        with _open_text(source) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"{source}:{line_number} must contain a JSON object"
                    )
                yield payload
        return

    if suffix == ".json":
        with _open_text(source) as handle:
            payload = json.load(handle)
        yield from _records_from_json(payload)
        return

    if suffix == ".csv":
        with _open_text(source, newline="") as handle:
            yield from csv.DictReader(handle)
        return

    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas and pyarrow are required for Parquet input") from exc
        yield from pd.read_parquet(source).to_dict(orient="records")
        return

    if suffix in {".pkl", ".pickle"}:
        with source.open("rb") as handle:
            payload = pickle.load(handle)
        if hasattr(payload, "to_dict"):
            yield from payload.to_dict(orient="records")
            return
        yield from _records_from_json(payload)
        return

    raise ValueError(f"unsupported local dataset format: {source}")


def load_split_mapping(path: str | Path | None) -> dict[int, str]:
    if path is None:
        return {}
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"split file not found: {source}")
    suffix = _logical_suffix(source)
    if suffix == ".npz":
        import numpy as np

        with np.load(source, allow_pickle=False) as payload:
            return {
                int(index): str(name)
                for name in payload.files
                for index in payload[name].reshape(-1).tolist()
            }
    if suffix in {".pt", ".pth"}:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to read a .pt split file") from exc
        payload = torch.load(source, map_location="cpu", weights_only=False)
        return _split_dict_to_mapping(payload)
    if suffix in {".json", ".jsonl", ".ndjson", ".csv"}:
        records = list(iter_records(source))
        if suffix == ".json" and len(records) == 1:
            first = records[0]
            if all(isinstance(value, (list, tuple)) for value in first.values()):
                return _split_dict_to_mapping(first)
        mapping: dict[int, str] = {}
        for row in records:
            index = row.get("index", row.get("idx", row.get("id")))
            split = row.get("split")
            if index is not None and split is not None:
                mapping[int(index)] = str(split)
        return mapping
    raise ValueError(f"unsupported split format: {source}")


def link_or_copy_directory(source: Path, destination: Path) -> Path:
    """Expose a large local raw directory under an external cache without copying."""

    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() == source:
            return destination
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    try:
        destination.symlink_to(source, target_is_directory=True)
    except OSError:
        shutil.copytree(source, destination)
    return destination


def _matches(path: Path, patterns: Sequence[str]) -> bool:
    return any(path.match(pattern) or path.name == pattern for pattern in patterns)


def _logical_suffix(path: Path) -> str:
    name = path.name.lower()
    for compression in _COMPRESSED_SUFFIXES:
        if name.endswith(compression):
            name = name[: -len(compression)]
            break
    return Path(name).suffix


def _open_text(path: Path, *, newline: str | None = None):
    name = path.name.lower()
    kwargs = {"mode": "rt", "encoding": "utf-8-sig", "newline": newline}
    if name.endswith(".gz"):
        return gzip.open(path, **kwargs)
    if name.endswith(".bz2"):
        return bz2.open(path, **kwargs)
    if name.endswith(".xz"):
        return lzma.open(path, **kwargs)
    return path.open(mode="r", encoding="utf-8-sig", newline=newline)


def _records_from_json(payload: Any) -> Iterator[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("JSON dataset lists must contain objects")
            yield item
        return
    if not isinstance(payload, dict):
        raise ValueError("JSON dataset must be an object or list of objects")
    for key in ("data", "records", "entries", "documents"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    raise ValueError(f"JSON {key!r} list must contain objects")
                yield item
            return
    yield payload


def _split_dict_to_mapping(payload: Mapping[str, Any]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for name, indices in payload.items():
        if hasattr(indices, "detach"):
            indices = indices.detach().cpu().numpy()
        if hasattr(indices, "reshape"):
            indices = indices.reshape(-1).tolist()
        for index in indices:
            mapping[int(index)] = str(name)
    return mapping


__all__ = [
    "LocalDatasetInput",
    "find_local_files",
    "iter_records",
    "link_or_copy_directory",
    "load_split_mapping",
    "require_local_file",
]
