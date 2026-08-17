from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import LoaderConfig
from ..encoding import TaskCompiler
from ..record import MaterialSample
from ..source import DatasetSource
from ..transforms import Compose
from .collate import material_collate
from .dataset import MaterialDataset, StreamingMaterialDataset
from .splits import random_split_indices

@dataclass(slots=True)
class MaterialDataModule:
    source: DatasetSource | Sequence[MaterialSample]
    compiler: TaskCompiler
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    transforms: Compose | None = None
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_indices: dict[str, Any] | None = None
    streaming: bool = False

    _datasets: dict[str, Any] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )

    def setup(self) -> None:
        if self.streaming:
            for split in ("train", "valid", "test"):
                self._datasets[split] = StreamingMaterialDataset(
                    self.source,
                    self.compiler,
                    transforms=self.transforms,
                    split=split,
                )
            return
        samples = (
            list(self.source.iter_samples())
            if isinstance(self.source, DatasetSource)
            else list(self.source)
        )
        if self.split_indices is None:
            explicit = {name: [sample for sample in samples if sample.split == name] for name in (
                "train",
                "valid",
                "test",
            )}
            if any(explicit.values()):
                split_samples = explicit
            else:
                indices = random_split_indices(len(samples), self.ratios, seed=self.loader.seed)
                split_samples = {
                    name: [samples[int(index)] for index in values]
                    for name, values in indices.items()
                }
        else:
            split_samples = {
                name: [samples[int(index)] for index in values]
                for name, values in self.split_indices.items()
            }
        self._datasets = {
            name: MaterialDataset(
                subset,
                self.compiler,
                transforms=self.transforms,
                cache_size=0,
            )
            for name, subset in split_samples.items()
        }

    def dataset(self, split: str):
        if not self._datasets:
            self.setup()
        return self._datasets[split]

    def dataloader(self, split: str):
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise ImportError("PyTorch is required; install zynnova[data]") from exc
        dataset = self.dataset(split)
        shuffle = (
            self.loader.shuffle
            and split == "train"
            and not self.streaming
            and self.loader.sampler is None
            and self.loader.batch_sampler is None
        )
        generator = torch.Generator().manual_seed(self.loader.seed)
        kwargs: dict[str, Any] = {
            "batch_size": self.loader.batch_size,
            "shuffle": shuffle,
            "num_workers": self.loader.num_workers,
            "pin_memory": self.loader.pin_memory,
            "drop_last": self.loader.drop_last and split == "train",
            "persistent_workers": self.loader.persistent_workers and self.loader.num_workers > 0,
            "collate_fn": self.loader.collate_fn or material_collate,
            "generator": generator,
            "timeout": self.loader.timeout,
            "worker_init_fn": self.loader.worker_init_fn,
        }
        if self.loader.sampler is not None:
            kwargs["sampler"] = self.loader.sampler
        if self.loader.batch_sampler is not None:
            for key in ("batch_size", "shuffle", "sampler", "drop_last"):
                kwargs.pop(key, None)
            kwargs["batch_sampler"] = self.loader.batch_sampler
        if self.loader.prefetch_factor is not None and self.loader.num_workers > 0:
            kwargs["prefetch_factor"] = self.loader.prefetch_factor
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        return self.dataloader("train")

    def val_dataloader(self):
        return self.dataloader("valid")

    def test_dataloader(self):
        return self.dataloader("test")

    def load_all(self, split: str = "train") -> dict[str, Any]:
        dataset = self.dataset(split)
        return material_collate([dataset[index] for index in range(len(dataset))])
