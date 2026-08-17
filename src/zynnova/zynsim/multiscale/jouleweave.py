"""JouleWeave adapter with externally injected models/calculators."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ..exceptions import PropertyResolutionError
from .properties import MaterialProperty, PropertyRequest, convert_property


StructureFactory = Callable[[float, float], Any]
PropertyExtractor = Callable[[Any, Any, PropertyRequest], MaterialProperty]


@dataclass(slots=True)
class JouleWeavePropertyProvider:
    """Resolve atomistic properties without loading a checkpoint inside zynsim.

    ``potential`` must already be a JouleWeave model or ASE-compatible
    calculator. The application is responsible for loading/fine-tuning it and
    selecting device/dtype before injection.
    """

    potential: Any
    structure_factory: StructureFactory
    output_directory: str | Path = "zynsim-jouleweave"
    device: str = "auto"
    dtype: str = "float32"
    compile_model: bool = False
    custom_extractors: Mapping[str, PropertyExtractor] = field(default_factory=dict)
    strain: float = 0.005
    _elastic_cache: dict[tuple[int, int], Any] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if isinstance(self.potential, (str, Path)):
            raise TypeError(
                "JouleWeavePropertyProvider does not load checkpoints. Load the model or "
                "calculator in application code and inject the resulting object."
            )
        if not callable(self.structure_factory):
            raise TypeError("structure_factory must be callable")
        self.output_directory = Path(self.output_directory)
        self.custom_extractors = dict(self.custom_extractors)

    def resolve(self, request: PropertyRequest) -> MaterialProperty | None:
        structure = self.structure_factory(request.soc, request.temperature_K)
        if request.name in self.custom_extractors:
            result = self.custom_extractors[request.name](
                self.potential, structure, request
            )
            return convert_property(result, request.unit)
        supported = {
            "elastic_stiffness",
            "bulk_modulus",
            "shear_modulus",
            "young_modulus",
            "poisson_ratio",
        }
        if request.name not in supported:
            return None
        result = self._elasticity(structure, request)
        mappings: dict[str, tuple[Any, str]] = {
            "elastic_stiffness": (result.stiffness_GPa, "GPa"),
            "bulk_modulus": (result.bulk_modulus_hill_GPa, "GPa"),
            "shear_modulus": (result.shear_modulus_hill_GPa, "GPa"),
            "young_modulus": (result.young_modulus_hill_GPa, "GPa"),
            "poisson_ratio": (result.poisson_ratio_hill, "1"),
        }
        value, unit = mappings[request.name]
        return convert_property(
            MaterialProperty(
                name=request.name,
                value=np.asarray(value).copy() if np.asarray(value).ndim else float(value),
                unit=unit,
                source=f"JouleWeave:{type(self.potential).__name__}",
                soc=request.soc,
                temperature_K=request.temperature_K,
                metadata={
                    "strain_amplitude": self.strain,
                    "model_loading": "external",
                },
            ),
            request.unit,
        )

    def _elasticity(self, structure: Any, request: PropertyRequest) -> Any:
        key = (round(request.soc * 10000), round(request.temperature_K * 10))
        if key in self._elastic_cache:
            return self._elastic_cache[key]
        try:
            from zynnova.ml.mlff.jouleweave import JouleWeaveMaterials
        except ImportError as exc:
            raise PropertyResolutionError(
                "JouleWeave integration requires zynnova[mlff]"
            ) from exc
        facade = JouleWeaveMaterials(
            self.potential,
            device=self.device,
            dtype=self.dtype,
            compile_model=self.compile_model,
        )
        target = self.output_directory / (
            f"soc-{request.soc:.5f}-T-{request.temperature_K:.2f}"
        )
        result = facade.elastic(
            structure,
            strain=self.strain,
            output_directory=target,
        )
        self._elastic_cache[key] = result
        return result


__all__ = [
    "JouleWeavePropertyProvider",
    "PropertyExtractor",
    "StructureFactory",
]
