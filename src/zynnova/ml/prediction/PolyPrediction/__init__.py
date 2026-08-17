"""Physics-aware multiview prediction for energy-storage polymers."""

from ...registry import MODELS
from .calibration import ConformalCalibrator
from .config import (
    ENERGY_STORAGE_CONDITIONS,
    ENERGY_STORAGE_PROPERTIES,
    PolyPredictionConfig,
    PolyPredictionDataConfig,
    PolyPredictionModelConfig,
    PolyPredictionTrainConfig,
    PropertySpec,
)
from .data import (
    PolymerPropertyDataset,
    PolyPredictionDataModule,
    polymer_property_collate,
    prepare_poly_prediction_data,
)
from .model import PolyPredictionNetwork
from .physics import high_entropy_report, physics_consistency_loss
from .predictor import (
    LoadedPolyPredictor,
    PolymerPrediction,
    load_poly_predictor,
    predict_polymer,
    predict_polymers,
)
from .screening import PropertyConstraint, ScreenedPolymer, screen_predictions
from .trainer import train_poly_prediction


@MODELS.register(
    "prediction",
    "poly_prediction",
    description=(
        "Multiview PSMILES/graph Transformer with heteroscedastic uncertainty, "
        "condition-aware energy-storage targets and physics consistency"
    ),
)
def create_poly_prediction(
    config: PolyPredictionModelConfig,
) -> PolyPredictionNetwork:
    return PolyPredictionNetwork(config)


__all__ = [
    "ConformalCalibrator",
    "ENERGY_STORAGE_CONDITIONS",
    "ENERGY_STORAGE_PROPERTIES",
    "LoadedPolyPredictor",
    "PolyPredictionConfig",
    "PolyPredictionDataConfig",
    "PolyPredictionDataModule",
    "PolyPredictionModelConfig",
    "PolyPredictionNetwork",
    "PolyPredictionTrainConfig",
    "PolymerPrediction",
    "PolymerPropertyDataset",
    "PropertyConstraint",
    "PropertySpec",
    "ScreenedPolymer",
    "create_poly_prediction",
    "high_entropy_report",
    "load_poly_predictor",
    "physics_consistency_loss",
    "polymer_property_collate",
    "predict_polymer",
    "predict_polymers",
    "prepare_poly_prediction_data",
    "screen_predictions",
    "train_poly_prediction",
]
