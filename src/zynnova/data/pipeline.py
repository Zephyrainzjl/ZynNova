from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import LoaderConfig
from .encoding import CompiledSample, TaskCompiler
from .record import MaterialSample, MaterialType
from .schema import TaskSpec
from .source import DatasetSource
from .storage import save_dataset
from .torch import MaterialDataModule, MaterialDataset, StreamingMaterialDataset
from .transforms import Compose


@dataclass(slots=True)
class DataPipeline:
    """Composable source -> transforms -> task -> Torch/storage pipeline."""

    source: DatasetSource | Sequence[MaterialSample] | Iterable[MaterialSample]
    transforms: Compose | None = None
    representation_schema: Any | None = None
    _materialized: list[MaterialSample] | None = field(default=None, init=False, repr=False)

    def iter_samples(self, split: str | None = None) -> Iterator[MaterialSample]:
        if self._materialized is not None:
            iterator: Iterable[MaterialSample] = self._materialized
        elif isinstance(self.source, DatasetSource):
            iterator = self.source.iter_samples(split)
        else:
            iterator = self.source
        for sample in iterator:
            if split is not None and sample.split != split:
                continue
            if self.transforms is not None:
                sample = self.transforms(sample)
                if sample is None:
                    continue
            yield sample

    def materialize(self, split: str | None = None) -> list[MaterialSample]:
        result = list(self.iter_samples(split))
        if split is None:
            self._materialized = result
        return result

    def fit_representation_schema(
        self,
        *,
        max_nodes: int | None = None,
        max_atoms: int | None = None,
        max_tokens: int = 512,
    ) -> Any:
        records = [
            sample.structure
            for sample in self.iter_samples()
            if sample.material_type is MaterialType.POLYMER and sample.structure is not None
        ]
        if not records:
            raise ValueError("representation schema fitting requires polymer records")
        from ..structure.polymer import RepresentationSchema

        kwargs: dict[str, Any] = {"max_tokens": max_tokens}
        if max_nodes is not None:
            kwargs["max_nodes"] = max_nodes
        if max_atoms is not None:
            kwargs["max_atoms"] = max_atoms
        self.representation_schema = RepresentationSchema.fit(records, **kwargs)
        return self.representation_schema

    def compiler(
        self,
        task: TaskSpec,
        *,
        tensorize: bool = True,
        device: str | None = None,
    ) -> TaskCompiler:
        return TaskCompiler(
            task,
            representation_schema=self.representation_schema,
            tensorize=tensorize,
            device=device,
        )

    def compile(
        self,
        task: TaskSpec,
        *,
        split: str | None = None,
        tensorize: bool = True,
        device: str | None = None,
    ) -> Iterator[CompiledSample]:
        compiler = self.compiler(task, tensorize=tensorize, device=device)
        for sample in self.iter_samples(split):
            compiled = compiler(sample)
            if compiled is not None:
                yield compiled

    def torch_dataset(
        self,
        task: TaskSpec,
        *,
        split: str | None = None,
        streaming: bool = False,
        tensorize: bool = True,
        cache_size: int = 0,
    ):
        compiler = self.compiler(task, tensorize=tensorize)
        if streaming:
            if isinstance(self.source, DatasetSource):
                return StreamingMaterialDataset(
                    self.source,
                    compiler,
                    transforms=self.transforms,
                    split=split,
                )
            return StreamingMaterialDataset(
                self.source,
                compiler,
                transforms=self.transforms,
                split=split,
            )
        return MaterialDataset(
            self.materialize(split),
            compiler,
            cache_size=cache_size,
        )

    def datamodule(
        self,
        task: TaskSpec,
        *,
        loader: LoaderConfig | None = None,
        streaming: bool = False,
        ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> MaterialDataModule:
        source: DatasetSource | Sequence[MaterialSample]
        if isinstance(self.source, DatasetSource) and self.transforms is None:
            source = self.source
        else:
            source = self.materialize()
        return MaterialDataModule(
            source=source,
            compiler=self.compiler(task),
            loader=loader or LoaderConfig(),
            transforms=self.transforms if source is self.source else None,
            ratios=ratios,
            streaming=streaming,
        )

    def save(
        self,
        path: str | Path,
        *,
        format: str | None = None,
        overwrite: bool = False,
        split: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        return save_dataset(
            self.iter_samples(split),
            path,
            format=format,
            overwrite=overwrite,
            metadata=metadata,
        )
