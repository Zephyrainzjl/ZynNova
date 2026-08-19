"""Thin, source-locked adapter around the upstream equivariant runtime.

No tensor-product, spherical-harmonic, symmetric-contraction, neighbour-list,
or GPU conversion implementation is duplicated here.  ZIVAR consumes the
upstream scalar energy and the official invariant node descriptors.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._deps import require_e3nn, require_mace, require_torch, upstream_warning_guard
from .config import SUPPORTED_MACE_SERIES, BackboneConfig

torch = require_torch()
nn = torch.nn


@dataclass(frozen=True, slots=True)
class BackboneSourceContract:
    package: str = "mace-torch"
    reviewed_version: str = SUPPORTED_MACE_SERIES
    model_class: str = "ScaleShiftMACE"
    first_interaction: str = "RealAgnosticInteractionBlock"
    later_interaction: str = "RealAgnosticResidualInteractionBlock"
    invariant_extractor: str = "mace.modules.utils.extract_invariant"
    gpu_converter: str = "mace.cli.convert_e3nn_cueq.run"
    oeq_converter: str = "mace.cli.convert_e3nn_oeq.run"
    hybrid_converter: str = "mace.cli.convert_e3nn_hybrid.run"
    lammps_converter: str = "mace.cli.create_lammps_model"


SOURCE_CONTRACT = BackboneSourceContract()


def installed_mace_version() -> str:
    require_mace()
    try:
        return importlib.metadata.version("mace-torch")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("mace is importable but mace-torch metadata is unavailable") from exc


def verify_mace_runtime(*, exact: bool = False) -> str:
    version = installed_mace_version()
    reviewed = tuple(int(value) for value in SUPPORTED_MACE_SERIES.split("."))
    matched = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if matched is None:
        raise RuntimeError(f"cannot parse installed mace-torch version {version!r}")
    numeric = tuple(int(value) for value in matched.groups())
    if exact and numeric != reviewed:
        raise RuntimeError(
            f"ZIVAR was reviewed against mace-torch {SUPPORTED_MACE_SERIES}; found {version}"
        )
    if numeric[:2] != reviewed[:2] or numeric < reviewed:
        raise RuntimeError(
            f"ZIVAR requires mace-torch >= {SUPPORTED_MACE_SERIES}, < 0.4; found {version}"
        )
    return version


def _mace_attribute(module_name: str, attribute: str) -> Any:
    root = require_mace()
    value = getattr(getattr(root, "modules", None), attribute, None)
    if value is not None:
        return value
    return getattr(importlib.import_module(module_name), attribute)


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _convert_backend(backbone: Any, backend: str, device: Any) -> Any:
    modules = {
        "cueq": "mace.cli.convert_e3nn_cueq",
        "oeq": "mace.cli.convert_e3nn_oeq",
        "hybrid": "mace.cli.convert_e3nn_hybrid",
    }
    converter = importlib.import_module(modules[backend]).run
    signature = inspect.signature(converter)
    kwargs: dict[str, Any] = {}
    if "device" in signature.parameters:
        kwargs["device"] = str(torch.device(device))
    converted = converter(backbone, **kwargs)
    converted.zivar_acceleration_backend = backend
    return converted


def _converter_architecture_error(config: BackboneConfig, backend: str) -> str | None:
    correlations = config.correlations
    if len(set(correlations)) != 1:
        return (
            "the reviewed MACE converters reconstruct correlation from the first "
            "layer and cannot preserve nonuniform per-layer values"
        )
    correlation = correlations[0]
    if backend in {"cueq", "hybrid"} and correlation not in {2, 3}:
        return "the MACE 0.3.16 CuEq contraction converter supports correlation 2 or 3"
    if backend == "oeq" and config.num_interactions < 2:
        return "the MACE 0.3.16 OEQ converter assumes at least two interactions"
    return None


def _resolve_backend(requested: str, device: Any, config: BackboneConfig) -> str:
    if requested == "e3nn" or torch.device(device).type != "cuda":
        if requested in {"cueq", "oeq", "hybrid"}:
            raise ValueError(f"backend={requested!r} requires a CUDA device")
        return "e3nn"
    cueq = _module_available("cuequivariance_torch") and _module_available(
        "cuequivariance_ops_torch"
    )
    oeq = _module_available("openequivariance")
    if requested == "auto":
        candidates = []
        if cueq and oeq:
            candidates.append("hybrid")
        if cueq:
            candidates.append("cueq")
        if oeq:
            candidates.append("oeq")
        candidates.append("e3nn")
        return next(
            candidate
            for candidate in candidates
            if candidate == "e3nn"
            or _converter_architecture_error(config, candidate) is None
        )
    if requested in {"cueq", "hybrid"} and not cueq:
        raise ImportError(
            f"backend={requested!r} requires cuequivariance-torch and the "
            "CUDA-major-matched cuequivariance-ops-torch package"
        )
    if requested in {"oeq", "hybrid"} and not oeq:
        raise ImportError(f"backend={requested!r} requires openequivariance")
    incompatibility = _converter_architecture_error(config, requested)
    if incompatibility is not None:
        raise ValueError(
            f"backend={requested!r} is unsafe for this architecture: {incompatibility}"
        )
    return requested


def build_reference_backbone(config: BackboneConfig, *, device: Any = "cpu") -> Any:
    """Build the reviewed upstream energy model and optionally convert it to CuEq."""

    with upstream_warning_guard():
        verify_mace_runtime()
        require_e3nn()
        o3 = importlib.import_module("e3nn.o3")
        functional = torch.nn.functional
        scale_shift_cls = _mace_attribute("mace.modules.models", "ScaleShiftMACE")
        first_cls = _mace_attribute(
            "mace.modules.blocks", "RealAgnosticInteractionBlock"
        )
        residual_cls = _mace_attribute(
            "mace.modules.blocks", "RealAgnosticResidualInteractionBlock"
        )
    atomic_energies = np.asarray(
        config.atomic_energies_eV
        if config.atomic_energies_eV is not None
        else np.zeros(len(config.atomic_numbers)),
        dtype=np.float64,
    )
    kwargs = {
        "r_max": float(config.cutoff_A),
        "num_bessel": int(config.num_bessel),
        "num_polynomial_cutoff": int(config.cutoff_polynomial_order),
        "max_ell": int(config.max_ell),
        "interaction_cls": residual_cls,
        "interaction_cls_first": first_cls,
        "num_interactions": int(config.num_interactions),
        "num_elements": len(config.atomic_numbers),
        "hidden_irreps": o3.Irreps(config.resolved_hidden_irreps),
        "MLP_irreps": o3.Irreps(config.mlp_irreps),
        "atomic_energies": atomic_energies,
        "avg_num_neighbors": float(config.average_num_neighbors),
        "atomic_numbers": tuple(config.atomic_numbers),
        "correlation": config.correlations,
        "gate": functional.silu,
        "pair_repulsion": bool(config.pair_repulsion),
        "apply_cutoff": True,
        "use_reduced_cg": bool(config.use_reduced_cg),
        "use_edge_irreps_first": bool(config.use_edge_irreps_first),
        "radial_MLP": list(config.radial_mlp),
        "radial_type": "bessel",
        "heads": ["Default"],
        "atomic_inter_scale": 1.0,
        "atomic_inter_shift": 0.0,
    }
    signature = inspect.signature(scale_shift_cls.__init__)
    if not any(
        item.kind is inspect.Parameter.VAR_KEYWORD
        for item in signature.parameters.values()
    ):
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    with upstream_warning_guard():
        backbone = scale_shift_cls(**kwargs).to(device)
        backend = _resolve_backend(config.backend, device, config)
        if backend != "e3nn":
            backbone = _convert_backend(backbone, backend, device).to(device)
        else:
            backbone.zivar_acceleration_backend = "e3nn"
    return backbone


def _forward_kwargs(module: Any) -> dict[str, bool]:
    parameters = inspect.signature(module.forward).parameters
    candidates = {
        "training": module.training,
        "compute_force": False,
        "compute_forces": False,
        "compute_virials": False,
        "compute_stress": False,
        "compute_displacement": False,
        "compute_hessian": False,
    }
    return {key: value for key, value in candidates.items() if key in parameters}


class ReferenceBackboneAdapter(nn.Module):
    """Stable feature/energy surface over the reviewed upstream model API."""

    def __init__(self, backbone: Any) -> None:
        super().__init__()
        if not hasattr(backbone, "products") or not hasattr(backbone, "num_interactions"):
            raise TypeError("backbone is not a compatible upstream equivariant model")
        self.model = backbone
        first_product = backbone.products[0]
        linear = getattr(first_product, "linear", None)
        irreps_out = getattr(linear, "irreps_out", None)
        if irreps_out is None:
            raise TypeError("cannot resolve product irreps from the upstream model")
        require_e3nn()
        o3 = importlib.import_module("e3nn.o3")
        self.irreps_out = o3.Irreps(str(irreps_out))
        self.lmax = int(self.irreps_out.lmax)
        families = [(int(mul), int(ir.l)) for mul, ir in self.irreps_out]
        if [ell for _, ell in families] != list(range(self.lmax + 1)) or len(
            {mul for mul, _ in families}
        ) != 1:
            raise ValueError(
                "the upstream invariant extractor requires one complete, "
                "equal-multiplicity l=0..L hidden family"
            )
        divisor = (self.lmax + 1) ** 2
        if self.irreps_out.dim % divisor:
            raise ValueError("upstream product irreps do not have the expected layout")
        self.features_per_layer = int(self.irreps_out.dim // divisor)
        self.num_interactions = int(backbone.num_interactions)
        self.invariant_dim = self.features_per_layer * self.num_interactions

    @property
    def atomic_numbers(self) -> tuple[int, ...]:
        values = self.model.atomic_numbers
        if hasattr(values, "detach"):
            values = values.detach().cpu().tolist()
        return tuple(int(value) for value in values)

    @property
    def cutoff_A(self) -> float:
        value = self.model.r_max
        return float(value.detach().cpu()) if hasattr(value, "detach") else float(value)

    def _extract_invariants(self, node_features: Any) -> Any:
        try:
            extractor = importlib.import_module("mace.modules.utils").extract_invariant
        except (ImportError, AttributeError):
            try:
                extractor = importlib.import_module("mace.tools.utils").extract_invariant
            except (ImportError, AttributeError):
                extractor = None
        if extractor is not None:
            result = extractor(
                node_features,
                num_layers=self.num_interactions,
                num_features=self.features_per_layer,
                l_max=self.lmax,
            )
        else:
            # Exact layout fallback used by the reviewed upstream descriptor code:
            # each non-final layer is a complete l-family; the final layer is 0e.
            parts = []
            offset = 0
            complete_width = self.features_per_layer * (self.lmax + 1) ** 2
            for layer in range(self.num_interactions):
                width = (
                    self.features_per_layer
                    if layer == self.num_interactions - 1
                    else complete_width
                )
                parts.append(node_features[:, offset : offset + self.features_per_layer])
                offset += width
            result = torch.cat(parts, dim=-1)
        if result.shape[-1] != self.invariant_dim:
            raise RuntimeError(
                f"upstream invariant descriptor width drifted: {result.shape[-1]} "
                f"!= {self.invariant_dim}"
            )
        return result

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        output = self.model(data, **_forward_kwargs(self.model))
        if not isinstance(output, dict) or output.get("node_feats") is None:
            raise RuntimeError("upstream model did not return node_feats")
        return {
            "energy": output["energy"],
            "node_energy": output.get("node_energy"),
            "invariant_features": self._extract_invariants(output["node_feats"]),
            "raw_output": output,
        }


__all__ = [
    "BackboneSourceContract",
    "ReferenceBackboneAdapter",
    "SOURCE_CONTRACT",
    "build_reference_backbone",
    "installed_mace_version",
    "verify_mace_runtime",
]
