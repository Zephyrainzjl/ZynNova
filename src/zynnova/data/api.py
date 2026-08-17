from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .config import DatasetConfig, LoaderConfig
from .encoding import TaskCompiler
from .pipeline import DataPipeline
from .record import MaterialSample
from .registry import DATASETS
from .schema import TaskSpec
from .source import DatasetSource
from .storage import load_dataset, save_dataset
from .torch import MaterialDataModule, MaterialDataset, StreamingMaterialDataset, material_collate

_BUILTINS_LOADED = False


def load_builtin_plugins() -> None:
    """Import built-in plugin modules once, without importing heavy dependencies."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from . import crystal as _crystal  # noqa: F401
    from . import molecular as _molecular  # noqa: F401
    from . import polymer as _polymer  # noqa: F401
    from . import special as _special  # noqa: F401

    _BUILTINS_LOADED = True


def list_datasets() -> tuple[str, ...]:
    load_builtin_plugins()
    return DATASETS.names()


def dataset_class(name: str):
    load_builtin_plugins()
    return DATASETS.get(name)


def create_dataset(
    name: str | DatasetConfig,
    *,
    root: str | Path | None = None,
    **options: Any,
) -> DatasetSource:
    """Create a registered public/local dataset plugin."""
    load_builtin_plugins()
    if isinstance(name, DatasetConfig):
        config = name
        kwargs = {
            "root": config.resolved_root(),
            "download": config.download,
            "prepare": config.prepare,
            **config.options,
            **options,
        }
        return DATASETS.create(config.name, **kwargs)
    if root is not None:
        options.setdefault("root", root)
    return DATASETS.create(name, **options)


def pipeline(
    source: DatasetSource | Sequence[MaterialSample] | Iterable[MaterialSample],
    **kwargs: Any,
) -> DataPipeline:
    return DataPipeline(source, **kwargs)


def make_torch_dataset(
    source: DatasetSource | Sequence[MaterialSample] | Iterable[MaterialSample],
    task: TaskSpec,
    *,
    representation_schema: Any | None = None,
    transforms: Any | None = None,
    split: str | None = None,
    streaming: bool = False,
    cache_size: int = 0,
    tensorize: bool = True,
):
    compiler = TaskCompiler(
        task,
        representation_schema=representation_schema,
        tensorize=tensorize,
    )
    if streaming:
        return StreamingMaterialDataset(source, compiler, transforms=transforms, split=split)
    samples = (
        list(source.iter_samples(split))
        if isinstance(source, DatasetSource)
        else list(source)
    )
    return MaterialDataset(
        samples,
        compiler,
        transforms=transforms,
        split=split,
        cache_size=cache_size,
    )


def make_datamodule(
    source: DatasetSource | Sequence[MaterialSample],
    task: TaskSpec,
    *,
    representation_schema: Any | None = None,
    loader: LoaderConfig | None = None,
    transforms: Any | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    streaming: bool = False,
) -> MaterialDataModule:
    return MaterialDataModule(
        source=source,
        compiler=TaskCompiler(task, representation_schema=representation_schema),
        loader=loader or LoaderConfig(),
        transforms=transforms,
        ratios=ratios,
        streaming=streaming,
    )


def make_dataloader(
    source: DatasetSource | Sequence[MaterialSample] | Iterable[MaterialSample],
    task: TaskSpec,
    *,
    representation_schema: Any | None = None,
    loader: LoaderConfig | None = None,
    transforms: Any | None = None,
    split: str | None = None,
    streaming: bool = False,
):
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("PyTorch is required; install zynnova[data]") from exc
    config = loader or LoaderConfig()
    dataset = make_torch_dataset(
        source,
        task,
        representation_schema=representation_schema,
        transforms=transforms,
        split=split,
        streaming=streaming,
    )
    kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "shuffle": (
            config.shuffle
            and not streaming
            and config.sampler is None
            and config.batch_sampler is None
        ),
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "drop_last": config.drop_last,
        "persistent_workers": config.persistent_workers and config.num_workers > 0,
        "collate_fn": config.collate_fn or material_collate,
        "generator": torch.Generator().manual_seed(config.seed),
        "timeout": config.timeout,
        "worker_init_fn": config.worker_init_fn,
    }
    if config.sampler is not None:
        kwargs["sampler"] = config.sampler
    if config.batch_sampler is not None:
        for key in ("batch_size", "shuffle", "sampler", "drop_last"):
            kwargs.pop(key, None)
        kwargs["batch_sampler"] = config.batch_sampler
    if config.prefetch_factor is not None and config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(dataset, **kwargs)


__all__ = [
    "load_builtin_plugins",
    "list_datasets",
    "dataset_class",
    "create_dataset",
    "pipeline",
    "make_torch_dataset",
    "make_datamodule",
    "make_dataloader",
    "save_dataset",
    "load_dataset",
]
