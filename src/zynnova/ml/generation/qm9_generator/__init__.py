"""Property-conditioned three-dimensional molecular generation on QM9."""

from ...registry import MODELS
from .config import (
    QM9GeneratorConfig,
    QM9GeneratorDataConfig,
    QM9GeneratorModelConfig,
    QM9GeneratorSamplingConfig,
    QM9GeneratorTrainConfig,
    QM9_PROPERTY_UNITS,
)
from .data import (
    QM9GeneratorDataModule,
    QM9PropertyDataset,
    prepare_qm9_generator_data,
)
from .model import QM9ConditionalGenerator
from .normalizer import QM9PropertyNormalizer
from .sampler import (
    GeneratedMolecule,
    QM9GenerationResult,
    composition_to_atomic_numbers,
    generate_qm9_candidates,
    generate_qm9_molecule,
    load_qm9_generator,
    save_qm9_generation,
)
from .trainer import train_qm9_generator
from .validation import GeometryReport, analyze_generated_structure, infer_bonds


@MODELS.register(
    "generation",
    "qm9_generator",
    description=(
        "Property-conditioned E(3)-equivariant QM9 coordinate generator with "
        "candidate ranking and geometry validation"
    ),
)
def create_qm9_generator(
    config: QM9GeneratorModelConfig | None = None,
) -> QM9ConditionalGenerator:
    return QM9ConditionalGenerator(config)


__all__ = [
    "GeneratedMolecule",
    "GeometryReport",
    "QM9ConditionalGenerator",
    "QM9GenerationResult",
    "QM9GeneratorConfig",
    "QM9GeneratorDataConfig",
    "QM9GeneratorDataModule",
    "QM9GeneratorModelConfig",
    "QM9GeneratorSamplingConfig",
    "QM9GeneratorTrainConfig",
    "QM9PropertyDataset",
    "QM9PropertyNormalizer",
    "QM9_PROPERTY_UNITS",
    "analyze_generated_structure",
    "composition_to_atomic_numbers",
    "create_qm9_generator",
    "generate_qm9_candidates",
    "generate_qm9_molecule",
    "infer_bonds",
    "load_qm9_generator",
    "prepare_qm9_generator_data",
    "save_qm9_generation",
    "train_qm9_generator",
]
