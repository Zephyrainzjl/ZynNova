"""Multiview, multi-fidelity prediction for energy-storage polymers."""

from ...registry import MODELS
from .config import (
    PolyPrismConfig,
    PolyPrismDataConfig,
    PolyPrismModelConfig,
    PolyPrismTrainConfig,
)
from .data import (
    PolyPrismDataModule,
    PolyPrismDataset,
    infer_fidelity_index,
    poly_prism_collate,
    prepare_poly_prism_data,
)
from .model import PolyPrismNetwork, SparseExpertBlock
from .objectives import balanced_student_t_nll, evidential_regularizer
from .predictor import (
    LoadedPolyPrism,
    PolyPrismPrediction,
    load_poly_prism,
    predict_one_poly_prism,
    predict_poly_prism,
)
from .trainer import train_poly_prism


@MODELS.register(
    "prediction",
    "poly_prism",
    description=(
        "Property-query multiview Transformer with sparse experts, multi-fidelity "
        "conditioning and decomposed evidential uncertainty"
    ),
)
def create_poly_prism(
    config: PolyPrismModelConfig | None = None,
) -> PolyPrismNetwork:
    return PolyPrismNetwork(config)


__all__ = [
    "LoadedPolyPrism",
    "PolyPrismConfig",
    "PolyPrismDataConfig",
    "PolyPrismDataModule",
    "PolyPrismDataset",
    "PolyPrismModelConfig",
    "PolyPrismNetwork",
    "PolyPrismPrediction",
    "PolyPrismTrainConfig",
    "SparseExpertBlock",
    "balanced_student_t_nll",
    "create_poly_prism",
    "evidential_regularizer",
    "infer_fidelity_index",
    "load_poly_prism",
    "poly_prism_collate",
    "predict_one_poly_prism",
    "predict_poly_prism",
    "prepare_poly_prism_data",
    "train_poly_prism",
]
