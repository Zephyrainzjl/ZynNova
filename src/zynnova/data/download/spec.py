from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    urls: tuple[str, ...]
    filename: str
    checksum: str | None = None
    checksum_algorithm: str = "sha256"
    archive: str | None = "auto"
    extract: bool = False
    extract_subdir: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.urls:
            raise ValueError("DownloadSpec.urls cannot be empty")
        if Path(self.filename).name != self.filename:
            raise ValueError("filename must be a basename")
