from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DatasetConfig:
    name: str
    root: str | Path = "data"
    split: str | None = None
    download: bool = True
    prepare: bool = True
    streaming: bool = False
    cache_size: int = 0
    options: dict[str, Any] = field(default_factory=dict)

    def resolved_root(self) -> Path:
        return Path(self.root).expanduser().resolve()


@dataclass(slots=True)
class LoaderConfig:
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    timeout: float = 0.0
    sampler: Any | None = None
    batch_sampler: Any | None = None
    collate_fn: Any | None = None
    worker_init_fn: Any | None = None
    seed: int = 0
