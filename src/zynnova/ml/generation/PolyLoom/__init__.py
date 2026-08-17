"""Self-conditioned discrete-flow generation for energy-storage polymers."""

from ...registry import MODELS
from ..PolyGen.representation import (
    GenerationRepresentation,
    decode_polymer_generation_sequence,
    encode_polymer_generation_sequence,
    polymer_selfies_to_psmiles,
    psmiles_to_polymer_selfies,
)
from ..PolyGen.theory import (
    PAPER_IRRADIATION_MECHANISMS,
    IrradiationMechanism,
    assess_paper_mechanisms,
)
from ..PolyGen.validation import PolymerValidityReport, validate_generated_polymer
from .config import (
    PolyLoomConfig,
    PolyLoomDataConfig,
    PolyLoomModelConfig,
    PolyLoomSamplingConfig,
    PolyLoomTrainConfig,
)
from .data import (
    PolyLoomDataModule,
    PolyLoomDataset,
    polymer_flow_collate,
    prepare_poly_loom_data,
)
from .model import PolyLoomNetwork, log_snr_time_embedding
from .objectives import cosine_corrupt, polyloom_losses
from .sampler import (
    LoadedPolyLoom,
    PolyLoomGeneratedPolymer,
    PolyLoomGenerationResult,
    generate_poly_loom,
    load_poly_loom,
    save_poly_loom_generation,
)
from .trainer import train_poly_loom


@MODELS.register(
    "generation",
    "poly_loom",
    description=(
        "Self-conditioned discrete flow with classifier-free multi-property "
        "guidance, endpoint control, independent PolyPrism reranking and "
        "chemistry validation"
    ),
)
def create_poly_loom(
    config: PolyLoomModelConfig | None = None,
) -> PolyLoomNetwork:
    return PolyLoomNetwork(config)


__all__ = [
    "GenerationRepresentation",
    "IrradiationMechanism",
    "LoadedPolyLoom",
    "PAPER_IRRADIATION_MECHANISMS",
    "PolyLoomConfig",
    "PolyLoomDataConfig",
    "PolyLoomDataModule",
    "PolyLoomDataset",
    "PolyLoomGeneratedPolymer",
    "PolyLoomGenerationResult",
    "PolyLoomModelConfig",
    "PolyLoomNetwork",
    "PolyLoomSamplingConfig",
    "PolyLoomTrainConfig",
    "PolymerValidityReport",
    "assess_paper_mechanisms",
    "cosine_corrupt",
    "create_poly_loom",
    "decode_polymer_generation_sequence",
    "encode_polymer_generation_sequence",
    "generate_poly_loom",
    "load_poly_loom",
    "log_snr_time_embedding",
    "polyloom_losses",
    "polymer_flow_collate",
    "polymer_selfies_to_psmiles",
    "prepare_poly_loom_data",
    "psmiles_to_polymer_selfies",
    "save_poly_loom_generation",
    "train_poly_loom",
    "validate_generated_polymer",
]
