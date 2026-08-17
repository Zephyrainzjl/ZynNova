from __future__ import annotations

import bz2
import gzip
import json
import lzma
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..exceptions import DownloadError
from ..utils import atomic_write_json, file_checksum
from .spec import DownloadSpec


class DownloadManager:
    """Resumable downloader with mirrors, checksums and safe extraction."""

    def __init__(
        self,
        root: str | Path,
        *,
        timeout: float = 60.0,
        retries: int = 3,
        chunk_size: int = 1024 * 1024,
        user_agent: str = "ZynNova-data/0.4",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.chunk_size = int(chunk_size)
        self.user_agent = user_agent

    def fetch(self, spec: DownloadSpec, *, force: bool = False) -> Path:
        destination = self.root / spec.filename
        lock_path = destination.with_suffix(destination.suffix + ".lock")
        with _file_lock(lock_path):
            if destination.exists() and not force:
                self._verify(destination, spec)
            else:
                self._download_mirrors(spec, destination)
                self._verify(destination, spec)
            self._write_manifest(destination, spec)
            if spec.extract:
                return self.extract(destination, spec=spec, force=force)
            return destination

    def fetch_many(
        self,
        specs: list[DownloadSpec] | tuple[DownloadSpec, ...],
        *,
        force: bool = False,
    ) -> list[Path]:
        return [self.fetch(spec, force=force) for spec in specs]

    def _download_mirrors(self, spec: DownloadSpec, destination: Path) -> None:
        errors: list[str] = []
        for url in spec.urls:
            for attempt in range(self.retries + 1):
                try:
                    self._download_one(url, destination, headers=dict(spec.headers))
                    return
                except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
                    errors.append(f"{url} attempt {attempt + 1}: {exc}")
                    if attempt < self.retries:
                        time.sleep(min(2**attempt, 8))
        raise DownloadError("all download mirrors failed:\n" + "\n".join(errors))

    def _download_one(self, url: str, destination: Path, *, headers: dict[str, str]) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        headers.setdefault("User-Agent", self.user_agent)
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            status = getattr(response, "status", None)
            mode = "ab" if offset and status == 206 else "wb"
            if mode == "wb":
                offset = 0
            with partial.open(mode) as handle:
                while True:
                    chunk = response.read(self.chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
        os.replace(partial, destination)

    @staticmethod
    def _verify(path: Path, spec: DownloadSpec) -> None:
        if not path.exists() or path.stat().st_size == 0:
            raise DownloadError(f"downloaded file is missing or empty: {path}")
        if spec.checksum:
            actual = file_checksum(path, spec.checksum_algorithm)
            expected = spec.checksum.lower()
            if actual.lower() != expected:
                raise DownloadError(
                    f"checksum mismatch for {path.name}: expected {expected}, got {actual}"
                )

    def _write_manifest(self, path: Path, spec: DownloadSpec) -> None:
        payload = {
            "filename": path.name,
            "size": path.stat().st_size,
            "urls": list(spec.urls),
            "checksum": spec.checksum,
            "checksum_algorithm": spec.checksum_algorithm,
            "description": spec.description,
        }
        atomic_write_json(path.with_suffix(path.suffix + ".download.json"), payload)

    def extract(
        self,
        archive: str | Path,
        *,
        spec: DownloadSpec | None = None,
        force: bool = False,
    ) -> Path:
        source = Path(archive)
        archive_type = _archive_type(source, spec.archive if spec else "auto")
        subdir = spec.extract_subdir if spec else None
        destination = self.root / (subdir or _default_extract_name(source, archive_type))
        marker = destination / ".zynnova-extracted.json"
        if marker.exists() and not force:
            return destination
        if force and destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        if archive_type == "zip":
            with zipfile.ZipFile(source) as handle:
                _safe_zip_extract(handle, destination)
        elif archive_type in {"tar", "tar.gz", "tgz", "tar.bz2", "tar.xz"}:
            with tarfile.open(source, "r:*") as handle:
                _safe_tar_extract(handle, destination)
        elif archive_type in {"gz", "bz2", "xz"}:
            output = destination / _strip_compression_suffix(source.name)
            opener = {"gz": gzip.open, "bz2": bz2.open, "xz": lzma.open}[archive_type]
            with opener(source, "rb") as source_handle, output.open("wb") as target:
                shutil.copyfileobj(source_handle, target)
        else:
            raise DownloadError(f"unsupported archive type: {archive_type}")
        atomic_write_json(
            marker,
            {
                "source": str(source),
                "source_checksum": file_checksum(source),
                "archive_type": archive_type,
            },
        )
        return destination


@contextmanager
def _file_lock(path: Path, *, timeout: float = 600.0) -> Iterator[None]:
    start = time.monotonic()
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() - start > timeout:
                raise DownloadError(f"timed out waiting for download lock: {path}")
            time.sleep(0.2)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def _archive_type(path: Path, requested: str | None) -> str:
    if requested and requested != "auto":
        return requested.lower()
    name = path.name.lower()
    for suffix, kind in (
        (".tar.gz", "tar.gz"),
        (".tar.bz2", "tar.bz2"),
        (".tar.xz", "tar.xz"),
        (".tgz", "tgz"),
        (".zip", "zip"),
        (".tar", "tar"),
        (".gz", "gz"),
        (".bz2", "bz2"),
        (".xz", "xz"),
    ):
        if name.endswith(suffix):
            return kind
    raise DownloadError(f"cannot infer archive type from {path.name}")


def _default_extract_name(path: Path, archive_type: str) -> str:
    name = path.name
    suffixes = {
        "tar.gz": ".tar.gz",
        "tar.bz2": ".tar.bz2",
        "tar.xz": ".tar.xz",
        "tgz": ".tgz",
        "zip": ".zip",
        "tar": ".tar",
        "gz": ".gz",
        "bz2": ".bz2",
        "xz": ".xz",
    }
    suffix = suffixes[archive_type]
    return name[: -len(suffix)]


def _safe_zip_extract(handle: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in handle.infolist():
        target = (destination / member.filename).resolve()
        if root not in target.parents and target != root:
            raise DownloadError(f"unsafe path in zip archive: {member.filename}")
    handle.extractall(destination)


def _safe_tar_extract(handle: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in handle.getmembers():
        target = (destination / member.name).resolve()
        if root not in target.parents and target != root:
            raise DownloadError(f"unsafe path in tar archive: {member.name}")
        if member.issym() or member.islnk():
            raise DownloadError(f"links are not extracted from archives: {member.name}")
    handle.extractall(destination, filter="data")


def _strip_compression_suffix(name: str) -> str:
    for suffix in (".gz", ".bz2", ".xz"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name
