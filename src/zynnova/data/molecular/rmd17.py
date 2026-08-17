from __future__ import annotations

import json
import os
import shutil
import tarfile
import urllib.error
import urllib.request
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ...structure import StructureData
from ..download import DownloadSpec
from ..exceptions import DownloadError
from ..local_input import find_local_files
from ..record import MaterialSample
from ..registry import DATASETS
from .base import MolecularDatasetSource

KCAL_MOL_TO_EV = 0.0433641153087705

_FIGSHARE_ARTICLE_API = "https://api.figshare.com/v2/articles/12672038"
_ARCHIVE_URLS = (
    "https://archive.materialscloud.org/records/"
    "pfffs-fff86/files/rmd17.tar.bz2?download=1",
    "https://figshare.com/ndownloader/files/23950376",
)
_ARCHIVE_MD5 = "cb1a927628d96f2e966025da4fb63d18"

_FILE_NAMES = {
    "aspirin": "rmd17_aspirin.npz",
    "azobenzene": "rmd17_azobenzene.npz",
    "benzene": "rmd17_benzene.npz",
    "ethanol": "rmd17_ethanol.npz",
    "malonaldehyde": "rmd17_malonaldehyde.npz",
    "naphthalene": "rmd17_naphthalene.npz",
    "paracetamol": "rmd17_paracetamol.npz",
    "salicylic": "rmd17_salicylic.npz",
    "toluene": "rmd17_toluene.npz",
    "uracil": "rmd17_uracil.npz",
}

_MOLECULE_ALIASES = {
    "salicylic acid": "salicylic",
    "salicylic_acid": "salicylic",
    "revised salicylic acid": "salicylic",
}


def _normalize_molecule(value: str) -> str:
    normalized = " ".join(value.strip().lower().replace("-", " ").split())
    if normalized.startswith("revised "):
        normalized = normalized[len("revised ") :]
    if normalized.startswith("rmd17 "):
        normalized = normalized[len("rmd17 ") :]
    normalized = _MOLECULE_ALIASES.get(normalized, normalized)
    if normalized not in _FILE_NAMES:
        supported = ", ".join(sorted(_FILE_NAMES))
        raise ValueError(
            f"unsupported revised-MD17 molecule {value!r}; choose one of: {supported}"
        )
    return normalized


def _select_indices(
    count: int,
    limit: int | None,
    *,
    selection: str,
    seed: int,
) -> np.ndarray:
    if count < 0:
        raise ValueError("count must be non-negative")
    if limit is None or limit >= count:
        return np.arange(count, dtype=np.int64)
    if limit < 0:
        raise ValueError("limit must be non-negative or None")
    if selection == "first":
        return np.arange(limit, dtype=np.int64)
    if selection == "uniform":
        return np.linspace(0, count - 1, limit, dtype=np.int64)
    if selection == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(count, size=limit, replace=False)).astype(np.int64)
    raise ValueError("selection must be 'first', 'uniform', or 'random'")


@DATASETS.register("rmd17", aliases=("md17", "revised_md17"))
class RevisedMD17DatasetSource(MolecularDatasetSource):
    """Revised MD17 source with ZynNova-owned download and NPZ parsing.

    The source intentionally does not instantiate ``torch_geometric.datasets.MD17``.
    Older PyG releases contain a Materials Cloud URL which now returns HTTP 404.
    ZynNova instead asks Figshare for the current per-molecule file URL and falls
    back to the complete official archive when direct-file metadata is unavailable.
    """

    name = "rmd17"
    homepage = "https://figshare.com/articles/dataset/12672038"
    citation = (
        "A. S. Christensen and O. A. von Lilienfeld, Machine Learning: "
        "Science and Technology 1, 045018 (2020)."
    )
    license = "CC0-1.0"

    def __init__(
        self,
        root: str | Path,
        *,
        molecule: str,
        convert_to_ev: bool = True,
        limit: int | None = None,
        selection: str = "random",
        seed: int = 0,
        local_file: str | Path | None = None,
        local_dir: str | Path | None = None,
        prefer_direct_download: bool = True,
        keep_archive: bool = False,
        metadata_timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        self.molecule = _normalize_molecule(molecule)
        self.file_name = _FILE_NAMES[self.molecule]
        self.convert_to_ev = bool(convert_to_ev)
        self.limit = limit
        self.selection = selection
        self.seed = int(seed)
        self.local_file = (
            None if local_file is None else Path(local_file).expanduser().resolve()
        )
        self._local_npz: Path | None = None
        self.prefer_direct_download = bool(prefer_direct_download)
        self.keep_archive = bool(keep_archive)
        self.metadata_timeout = float(metadata_timeout)
        super().__init__(
            root,
            local_file=local_file,
            local_dir=local_dir,
            **kwargs,
        )

    def download(self, *, force: bool = False) -> None:
        local = self.materialize_local_input(
            force=force,
            extract_subdir="rmd17-local",
        )
        if local is not None:
            candidates = find_local_files(local, (self.file_name,))
            if not candidates:
                raise FileNotFoundError(
                    f"local revised-MD17 input does not contain {self.file_name}: {local}"
                )
            self._local_npz = candidates[0]
            return

        destination = self.raw_dir / self.file_name
        if destination.is_file() and destination.stat().st_size > 0 and not force:
            return

        legacy = self._legacy_npz_path()
        if legacy is not None and not force:
            return

        direct_error: Exception | None = None
        if self.prefer_direct_download:
            try:
                metadata = self._figshare_file_metadata(self.file_name)
                checksum = metadata.get("computed_md5") or metadata.get("md5")
                self.download_manager.fetch(
                    DownloadSpec(
                        urls=(str(metadata["download_url"]),),
                        filename=self.file_name,
                        checksum=None if checksum is None else str(checksum),
                        checksum_algorithm="md5",
                        archive=None,
                        description=(
                            f"revised MD17 {self.molecule} trajectory from Figshare"
                        ),
                    ),
                    force=force,
                )
                return
            except (
                DownloadError,
                KeyError,
                OSError,
                RuntimeError,
                ValueError,
                urllib.error.HTTPError,
                urllib.error.URLError,
            ) as exc:
                direct_error = exc

        try:
            self._download_from_archive(destination, force=force)
        except Exception as archive_error:
            details = []
            if direct_error is not None:
                details.append(f"direct Figshare download failed: {direct_error}")
            details.append(f"official archive fallback failed: {archive_error}")
            raise DownloadError(
                "could not download revised MD17 data. "
                + "\n".join(details)
                + "\nYou may manually download the molecule NPZ and pass local_file=..."
            ) from archive_error

    def _figshare_file_metadata(self, filename: str) -> dict[str, Any]:
        request = urllib.request.Request(
            _FIGSHARE_ARTICLE_API,
            headers={
                "Accept": "application/json",
                "User-Agent": "ZynNova-data/0.5.2",
            },
        )
        with urllib.request.urlopen(
            request,
            timeout=self.metadata_timeout,
        ) as response:
            payload = json.load(response)
        files = payload.get("files", [])
        for item in files:
            if item.get("name") == filename and item.get("download_url"):
                return dict(item)
        available = ", ".join(
            sorted(str(item.get("name")) for item in files if item.get("name"))
        )
        raise RuntimeError(
            f"Figshare metadata does not contain {filename!r}; available files: {available}"
        )

    def _download_from_archive(self, destination: Path, *, force: bool) -> None:
        archive = self.download_manager.fetch(
            DownloadSpec(
                urls=_ARCHIVE_URLS,
                filename="rmd17.tar.bz2",
                checksum=_ARCHIVE_MD5,
                checksum_algorithm="md5",
                archive="tar.bz2",
                extract=False,
                description="complete revised MD17 archive",
            ),
            force=force,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with tarfile.open(archive, "r:bz2") as handle:
                member = next(
                    (
                        item
                        for item in handle.getmembers()
                        if item.isfile()
                        and Path(item.name).name == self.file_name
                        and "npz_data" in Path(item.name).parts
                    ),
                    None,
                )
                if member is None:
                    raise DownloadError(
                        f"{self.file_name} is missing from {archive.name}"
                    )
                source = handle.extractfile(member)
                if source is None:
                    raise DownloadError(f"cannot read {member.name} from archive")
                with source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
            if not self.keep_archive:
                archive.unlink(missing_ok=True)
                archive.with_suffix(archive.suffix + ".download.json").unlink(
                    missing_ok=True
                )

    def _legacy_npz_path(self) -> Path | None:
        candidates = (
            self.raw_dir / "rmd17" / "npz_data" / self.file_name,
            self.raw_dir / "_archive" / "rmd17" / "npz_data" / self.file_name,
            self.root / "raw" / "rmd17" / "npz_data" / self.file_name,
        )
        return next((path for path in candidates if path.is_file()), None)

    def _npz_path(self) -> Path:
        if self._local_npz is not None:
            return self._local_npz
        direct = self.raw_dir / self.file_name
        if direct.is_file():
            return direct
        legacy = self._legacy_npz_path()
        if legacy is not None:
            return legacy
        raise FileNotFoundError(
            f"revised-MD17 file is not prepared: expected {direct}. "
            "Create the source with download=True or pass local_file=..."
        )

    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        del split  # rMD17 does not define a single canonical split per sample.
        path = self._npz_path()
        factor = KCAL_MOL_TO_EV if self.convert_to_ev else 1.0

        with np.load(path, allow_pickle=False) as raw:
            required = {"nuclear_charges", "coords", "energies", "forces"}
            missing = required.difference(raw.files)
            if missing:
                raise ValueError(
                    f"invalid revised-MD17 NPZ {path}: missing keys {sorted(missing)}"
                )

            atomic_numbers = np.asarray(raw["nuclear_charges"], dtype=np.int64)
            coordinates = np.asarray(raw["coords"], dtype=np.float64)
            energies = np.asarray(raw["energies"], dtype=np.float64).reshape(-1)
            forces = np.asarray(raw["forces"], dtype=np.float64)

            if coordinates.ndim != 3 or coordinates.shape[1:] != (
                len(atomic_numbers),
                3,
            ):
                raise ValueError(
                    "rMD17 coords must have shape [samples, atoms, 3], got "
                    f"{coordinates.shape}"
                )
            if forces.shape != coordinates.shape:
                raise ValueError(
                    f"rMD17 forces shape {forces.shape} does not match coords "
                    f"shape {coordinates.shape}"
                )
            if len(energies) != len(coordinates):
                raise ValueError("rMD17 energies and coordinates have different lengths")

            if self.limit is None or self.limit > 1000:
                warnings.warn(
                    "The revised-MD17 authors recommend training on no more than "
                    "1000 samples because the trajectory frames are correlated.",
                    UserWarning,
                    stacklevel=2,
                )

            indices = _select_indices(
                len(coordinates),
                self.limit,
                selection=self.selection,
                seed=self.seed,
            )
            for source_index in indices.tolist():
                energy = float(energies[source_index]) * factor
                force = np.asarray(forces[source_index] * factor, dtype=np.float64)
                structure = StructureData(
                    atomic_numbers=atomic_numbers,
                    positions=coordinates[source_index],
                    source=str(path),
                    info={
                        "dataset": "revised MD17",
                        "molecule": self.molecule,
                        "source_index": source_index,
                    },
                )
                yield MaterialSample(
                    id=f"rmd17:{self.molecule}:{source_index}",
                    material_type=self.material_type,
                    structure=structure,
                    labels={"energy": energy, "forces": force},
                    metadata={
                        "molecule": self.molecule,
                        "energy_unit": (
                            "eV" if self.convert_to_ev else "kcal/mol"
                        ),
                        "force_unit": (
                            "eV/angstrom"
                            if self.convert_to_ev
                            else "kcal/mol/angstrom"
                        ),
                        "source_index": source_index,
                    },
                    provenance={
                        "dataset": "revised MD17",
                        "figshare_article_id": 12672038,
                        "doi": "10.6084/m9.figshare.12672038",
                        "file": self.file_name,
                        "index": source_index,
                    },
                )


__all__ = [
    "KCAL_MOL_TO_EV",
    "RevisedMD17DatasetSource",
]
