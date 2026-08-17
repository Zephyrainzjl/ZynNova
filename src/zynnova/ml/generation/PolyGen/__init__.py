"""Property-conditioned masked discrete-flow generation for energy-storage polymers."""

from ...registry import MODELS
from .config import (
    PolyGenConfig,
    PolyGenDataConfig,
    PolyGenModelConfig,
    PolyGenSamplingConfig,
    PolyGenTrainConfig,
)
from .data import (
    PolyGenDataModule,
    PolymerFlowDataset,
    polymer_flow_collate,
    prepare_poly_gen_data,
)
from .model import PolymerMaskedFlow
from .representation import (
    GenerationRepresentation,
    decode_polymer_generation_sequence,
    encode_polymer_generation_sequence,
    polymer_selfies_to_psmiles,
    psmiles_to_polymer_selfies,
)
from .sampler import (
    GeneratedPolymer,
    LoadedPolyGenerator,
    PolyGenerationResult,
    generate_polymers,
    load_poly_generator,
    save_generation,
)
from .theory import (
    PAPER_IRRADIATION_MECHANISMS,
    IrradiationMechanism,
    assess_paper_mechanisms,
)
from .trainer import corrupt_discrete_flow, train_poly_gen
from .validation import PolymerValidityReport, validate_generated_polymer


@MODELS.register(
    "generation",
    "poly_gen",
    description=(
        "Classifier-free property-conditioned masked discrete flow over "
        "Polymer-SELFIES with validity-preserving decoding, predictor reranking "
        "and physics filters"
    ),
)
def create_poly_gen(config: PolyGenModelConfig) -> PolymerMaskedFlow:
    return PolymerMaskedFlow(config)


__all__ = [
    "GeneratedPolymer",
    "GenerationRepresentation",
    "LoadedPolyGenerator",
    "PAPER_IRRADIATION_MECHANISMS",
    "PolyGenConfig",
    "PolyGenDataConfig",
    "PolyGenDataModule",
    "PolyGenModelConfig",
    "PolyGenSamplingConfig",
    "PolyGenTrainConfig",
    "PolyGenerationResult",
    "PolymerFlowDataset",
    "PolymerMaskedFlow",
    "PolymerValidityReport",
    "IrradiationMechanism",
    "assess_paper_mechanisms",
    "corrupt_discrete_flow",
    "create_poly_gen",
    "decode_polymer_generation_sequence",
    "encode_polymer_generation_sequence",
    "generate_polymers",
    "load_poly_generator",
    "polymer_flow_collate",
    "prepare_poly_gen_data",
    "polymer_selfies_to_psmiles",
    "psmiles_to_polymer_selfies",
    "save_generation",
    "train_poly_gen",
    "validate_generated_polymer",
]
