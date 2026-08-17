from __future__ import annotations

import json
import zipfile
from pathlib import Path

from ..core.polymer import PolymerRecord
from .json_codec import record_from_dict, record_to_dict


def save_json(record: PolymerRecord, path: str | Path, *, indent: int = 2) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record_to_dict(record), ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )
    return output


def load_json(path: str | Path) -> PolymerRecord:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return record_from_dict(data)


def save_zpoly(record: PolymerRecord, path: str | Path) -> Path:
    """Save a portable .zpoly ZIP container.

    Version 0.1 stores the complete record in JSON. Future versions can move large
    trajectories and fields into Zarr/HDF5 without changing the PolymerRecord API.
    """

    output = Path(path)
    if output.suffix != ".zpoly":
        output = output.with_suffix(".zpoly")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "ZPoly",
        "container_version": "0.1.0",
        "schema_version": record.schema_version,
        "record_path": "record.json",
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        archive.writestr(
            "record.json",
            json.dumps(record_to_dict(record), ensure_ascii=False, separators=(",", ":")),
        )
    return output


def load_zpoly(path: str | Path) -> PolymerRecord:
    with zipfile.ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        data = json.loads(archive.read(manifest["record_path"]))
    return record_from_dict(data)
