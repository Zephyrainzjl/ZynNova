"""Extensible materials dataset, preprocessing and PyTorch input layer."""

from .api import (
    create_dataset,
    dataset_class,
    list_datasets,
    load_builtin_plugins,
    make_dataloader,
    make_datamodule,
    make_torch_dataset,
    pipeline,
)
from .catalog import DatasetInfo, dataset_catalog
from .config import DatasetConfig, LoaderConfig
from .download import DownloadManager, DownloadSpec
from .local_input import LocalDatasetInput
from .encoding import CompiledSample, TaskCompiler, compile_sample, encode_structure
from .pipeline import DataPipeline
from .record import MaterialSample, MaterialType
from .registry import DATASETS, ENCODERS, STORAGE_FORMATS, TRANSFORMS, Registry
from .schema import (
    FieldLevel,
    FieldRole,
    FieldSpec,
    MissingPolicy,
    StructureEncodingSpec,
    TaskKind,
    TaskSpec,
)
from .source import DatasetSource, SequenceSource
from .statistics import FieldStatistics, fit_field_statistics, standardization_pipeline
from .validation import DatasetIssue, DatasetReport, validate_dataset
from .storage import PreparedDataset, load_dataset, save_dataset
from .torch import (
    MaterialDataModule,
    MaterialDataset,
    StreamingMaterialDataset,
    material_collate,
    random_split_indices,
)
from .transforms import *
from .transforms import __all__ as _transform_all

load_builtin_plugins()

__all__ = [
    "MaterialSample",
    "MaterialType",
    "FieldRole",
    "FieldLevel",
    "FieldSpec",
    "MissingPolicy",
    "TaskKind",
    "TaskSpec",
    "StructureEncodingSpec",
    "DatasetSource",
    "FieldStatistics",
    "fit_field_statistics",
    "standardization_pipeline",
    "DatasetIssue",
    "DatasetReport",
    "validate_dataset",
    "SequenceSource",
    "DatasetConfig",
    "LoaderConfig",
    "DownloadSpec",
    "DownloadManager",
    "LocalDatasetInput",
    "CompiledSample",
    "TaskCompiler",
    "compile_sample",
    "encode_structure",
    "MaterialDataset",
    "StreamingMaterialDataset",
    "MaterialDataModule",
    "material_collate",
    "random_split_indices",
    "DataPipeline",
    "PreparedDataset",
    "save_dataset",
    "load_dataset",
    "create_dataset",
    "dataset_class",
    "list_datasets",
    "dataset_catalog",
    "DatasetInfo",
    "make_torch_dataset",
    "make_datamodule",
    "make_dataloader",
    "pipeline",
    "Registry",
    "DATASETS",
    "TRANSFORMS",
    "ENCODERS",
    "STORAGE_FORMATS",
    *_transform_all,
]
