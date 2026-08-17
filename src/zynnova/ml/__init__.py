"""Machine-learning namespace for ZynNova."""
"""Built-in machine-learning models for ZynNova.

Model implementations live inside ``zynnova.ml``. Training data, checkpoints,
configuration snapshots, generated structures and exported models are always
written to an external :class:`MLWorkspace`.
"""

from .common import TrainingResult
from .generation import *
from .generation import __all__ as _generation_all
from .mlff import *
from .mlff import __all__ as _mlff_all
from .prediction import *
from .prediction import __all__ as _prediction_all
from .registry import MODELS, ModelEntry, ModelRegistry
from .workspace import MLWorkspace, RunPaths


def create_model(category: str, name: str, **kwargs):
    return MODELS.create(category, name, **kwargs)


def list_models(category: str | None = None) -> tuple[str, ...]:
    return MODELS.names(category)


__all__ = [
    "MLWorkspace",
    "MODELS",
    "ModelEntry",
    "ModelRegistry",
    "RunPaths",
    "TrainingResult",
    "create_model",
    "list_models",
    *_mlff_all,
    *_generation_all,
    *_prediction_all,
]
