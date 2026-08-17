from __future__ import annotations

import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..download import DownloadSpec
from ..record import MaterialSample, MaterialType
from ..registry import DATASETS
from ..source import DatasetSource


@DATASETS.register("oc20", aliases=("open_catalyst_20",))
class OC20DatasetSource(DatasetSource):
    """Open Catalyst LMDB adapter for S2EF/IS2RE style samples.

    A split URL can be supplied explicitly because official releases contain many
    differently sized archives. Existing LMDB directories can be opened without
    downloading by passing ``lmdb_path``.
    """

    name = "oc20"
    material_type = MaterialType.SPECIAL
    homepage = "https://opencatalystproject.org/"
    license = "CC BY 4.0"

    def __init__(
        self,
        root: str | Path,
        *,
        lmdb_path: str | Path | None = None,
        local_file: str | Path | None = None,
        local_dir: str | Path | None = None,
        url: str | None = None,
        filename: str = "oc20.tar.gz",
        checksum: str | None = None,
        task: str = "s2ef",
        **kwargs: Any,
    ) -> None:
        if lmdb_path is not None and (local_file is not None or local_dir is not None):
            raise ValueError(
                "lmdb_path and local_file/local_dir are aliases; pass only one"
            )
        self._lmdb_path = Path(lmdb_path).expanduser().resolve() if lmdb_path else None
        self.url = url
        self.filename = filename
        self.checksum = checksum
        self.task = task.lower()
        super().__init__(
            root,
            local_file=local_file,
            local_dir=local_dir,
            **kwargs,
        )

    def download(self, *, force: bool = False) -> None:
        if self._lmdb_path is not None:
            if not self._lmdb_path.exists():
                raise FileNotFoundError(f"OC20 LMDB path not found: {self._lmdb_path}")
            return
        local = self.materialize_local_input(
            force=force,
            extract_subdir="oc20-local",
        )
        if local is not None:
            self._lmdb_path = _discover_lmdb(local)
            return
        if self.url is None:
            raise ValueError("OC20 requires either lmdb_path or an official split URL")
        extracted = self.download_manager.fetch(
            DownloadSpec(
                urls=(self.url,),
                filename=self.filename,
                checksum=self.checksum,
                extract=True,
                description="Open Catalyst dataset split",
            ),
            force=force,
        )
        self._lmdb_path = _discover_lmdb(extracted)

    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        if self._lmdb_path is None:
            raise RuntimeError("OC20 LMDB path is not available")
        try:
            import lmdb
        except ImportError as exc:
            raise ImportError("lmdb is required; install zynnova[data-catalysis]") from exc
        from ...structure import StructureData

        if self._lmdb_path.is_file():
            files = [self._lmdb_path]
        elif (self._lmdb_path / "data.mdb").exists():
            files = [self._lmdb_path]
        else:
            files = sorted(self._lmdb_path.glob("*.lmdb"))
            files.extend(
                path.parent
                for path in self._lmdb_path.rglob("data.mdb")
                if path.parent not in files
            )
        if not files:
            raise FileNotFoundError(f"no LMDB environments found in {self._lmdb_path}")
        for file_path in files:
            environment = lmdb.open(
                str(file_path),
                subdir=file_path.is_dir(),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            with environment.begin() as transaction:
                cursor = transaction.cursor()
                for key, raw in cursor:
                    if key in {b"length", b"metadata"}:
                        continue
                    data = pickle.loads(raw)
                    z = _array(data, "atomic_numbers", "z")
                    pos = _array(data, "pos", "positions")
                    cell = _array(data, "cell", default=np.zeros((3, 3)))
                    pbc = _array(data, "pbc", default=[True, True, False])
                    labels: dict[str, Any] = {}
                    energy = _value(data, "y", "energy", "relaxed_energy")
                    if energy is not None:
                        labels["energy"] = float(np.asarray(energy).reshape(-1)[0])
                    forces = _value(data, "force", "forces")
                    if forces is not None:
                        labels["forces"] = np.asarray(forces)
                    sample_split = str(_value(data, "split") or split or "train")
                    yield MaterialSample(
                        id=f"{file_path.stem}:{key.decode(errors='replace')}",
                        material_type=self.material_type,
                        structure=StructureData(z, pos, cell=cell, pbc=pbc),
                        labels=labels,
                        metadata={
                            "sid": _value(data, "sid"),
                            "fid": _value(data, "fid"),
                            "tags": _value(data, "tags"),
                            "task": self.task,
                        },
                        provenance={"dataset": "OC20", "lmdb": str(file_path)},
                        split=sample_split,
                    )
            environment.close()


def _value(data: Any, *names: str) -> Any:
    for name in names:
        if isinstance(data, dict) and name in data:
            return data[name]
        if hasattr(data, name):
            return getattr(data, name)
    return None


def _array(data: Any, *names: str, default: Any = None) -> np.ndarray:
    value = _value(data, *names)
    if value is None:
        value = default
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _discover_lmdb(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.name in {"data.mdb", "lock.mdb"}:
            return path.parent
        if path.suffix.lower() == ".lmdb":
            return path
    if (path / "data.mdb").is_file():
        return path
    lmdb_files = sorted(path.rglob("*.lmdb")) if path.is_dir() else []
    if lmdb_files:
        return lmdb_files[0].parent
    data_files = sorted(path.rglob("data.mdb")) if path.is_dir() else []
    if data_files:
        return data_files[0].parent
    raise FileNotFoundError(f"no LMDB environment found in local OC20 input: {path}")
