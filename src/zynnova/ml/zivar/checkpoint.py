"""Self-describing, versioned ZIVAR checkpoints."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._deps import require_torch
from .backbone import installed_mace_version
from .config import (
    ARCHITECTURE_REVISION,
    LEGACY_ARCHITECTURE_REVISIONS,
    NUMERICS_REVISION,
    ZIVAR_VERSION,
    ZIVARConfig,
)
from .model import build_zivar

torch = require_torch()
CHECKPOINT_SCHEMA = "zivar-variational-checkpoint-0.2.1"
LEGACY_CHECKPOINT_SCHEMAS: tuple[str, ...] = (
    "zivar-electrospin-checkpoint-0.1.0",
)


@dataclass(frozen=True, slots=True)
class LegacyBackboneImportReport:
    """Audit record for a deliberately limited legacy-weight import.

    A legacy electro-spin checkpoint cannot be upgraded into the variational
    model.  The only transferable state is the registered local backbone; all
    electronic, magnetic, QEq/SCF and output-head state is deliberately left
    freshly initialized and therefore requires a new training run.
    """

    source_checkpoint: str
    source_schema: str
    source_architecture_revision: str
    target_architecture_revision: str
    imported_keys: tuple[str, ...]
    ignored_legacy_keys: tuple[str, ...]
    imported_tensor_count: int
    imported_value_count: int
    requires_retraining: bool = field(default=True, init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda", [])
    if cuda_state:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_state)
    random.setstate(tuple(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )


def save_zivar(
    model: Any,
    path: str | Path,
    *,
    optimizer: Any | None = None,
    trainer: Any | None = None,
    epoch: int = 0,
    metrics: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
    release_evidence: str | Path | dict[str, Any] | None = None,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(model.config, ZIVARConfig):
        raise TypeError("model does not expose a ZIVARConfig")
    if trainer is not None:
        if getattr(trainer, "model", None) is not model:
            raise ValueError(
                "checkpoint trainer belongs to a different model instance"
            )
        trainer_optimizer = getattr(trainer, "optimizer", None)
        if optimizer is not None and optimizer is not trainer_optimizer:
            raise ValueError(
                "explicit optimizer differs from the checkpoint trainer optimizer"
            )
        optimizer = trainer_optimizer
        trainer_epoch = int(getattr(trainer, "epoch", 0))
        if epoch not in {0, trainer_epoch}:
            raise ValueError("explicit epoch differs from the checkpoint trainer epoch")
        epoch = trainer_epoch
    if callable(getattr(model, "seal_backbone", None)):
        model.seal_backbone()
    validation = None
    evidence_payload = None
    if release_evidence is not None:
        from .maturity import assert_release_ready

        validation = assert_release_ready(
            Path(__file__).resolve().parent, release_evidence
        ).to_dict()
        evidence_payload = (
            dict(release_evidence)
            if isinstance(release_evidence, dict)
            else json.loads(Path(release_evidence).read_text(encoding="utf-8"))
        )
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "version": ZIVAR_VERSION,
        "architecture_revision": ARCHITECTURE_REVISION,
        "numerics_revision": NUMERICS_REVISION,
        "mace_torch_version": (
            installed_mace_version()
            if model.config.backbone.kind in {"mace", "zephyr"}
            else None
        ),
        "backbone_manifest": model.backbone_manifest,
        "config": model.config.to_dict(),
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "training_state": None if trainer is None else trainer.state_dict(),
        "rng_state": _rng_state(),
        "epoch": int(epoch),
        "metrics": dict(metrics or {}),
        "metadata": dict(metadata or {}),
        "release_validation": validation,
        "release_evidence": evidence_payload,
        "capabilities": {
            "variational_q_p_Q_m": model.config.electronic.method == "variational",
            "legacy_polar_multipole_density": model.config.electronic.method == "polar",
            "spin_lattice_hamiltonian": model.config.spin.mode == "spin_lattice",
            "formal_oxidation_supervision": (
                model.config.electronic.oxidation.enabled
                and model.config.electronic.oxidation.label_source.lower() != "none"
            ),
            "production_validated": validation is not None,
        },
    }
    torch.save(payload, destination)
    return destination


def _validate_backbone_manifest(
    stored: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    identity_keys = (
        "contract",
        "kind",
        "architecture",
        "implementation",
        "invariant_dim",
        "atomic_numbers",
        "cutoff_A",
    )
    for key in identity_keys:
        if key not in stored:
            raise ValueError(f"checkpoint backbone manifest is missing {key!r}")
        source = stored[key]
        target = expected.get(key)
        if key == "atomic_numbers":
            source, target = tuple(source), tuple(target)
        if source != target:
            raise ValueError(f"checkpoint backbone {key} is incompatible")


def _validate_current_checkpoint(payload: Any) -> Mapping[str, Any]:
    """Validate the complete identity of a version-2 checkpoint.

    A schema name alone is not an architecture contract.  Version,
    architecture, numerics, configuration and tensor payload are all required
    before either inference loading or training-state restoration may proceed.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("ZIVAR checkpoint payload must be a mapping")
    schema = payload.get("schema")
    if schema in LEGACY_CHECKPOINT_SCHEMAS:
        raise ValueError(
            "legacy fixed-depth electro-spin checkpoints are not physically "
            "equivalent to the variational SCF architecture; migrate only "
            "compatible backbone weights and retrain all q/p/Q/m heads"
        )
    if schema != CHECKPOINT_SCHEMA:
        raise ValueError("not a supported current ZIVAR checkpoint")
    if payload.get("version") != ZIVAR_VERSION:
        raise ValueError("checkpoint package version is incompatible")
    if payload.get("architecture_revision") != ARCHITECTURE_REVISION:
        raise ValueError("checkpoint architecture revision is incompatible")
    if payload.get("numerics_revision") != NUMERICS_REVISION:
        raise ValueError(
            "checkpoint numerics revision is incompatible; start a new "
            "training run or use an explicit migration"
        )
    if not isinstance(payload.get("config"), Mapping):
        raise ValueError("checkpoint does not contain a valid configuration")
    if not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("checkpoint does not contain a state dictionary")
    return payload


def _legacy_target_config(payload: Mapping[str, Any]) -> ZIVARConfig:
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("legacy checkpoint does not contain a valid configuration")
    legacy = ZIVARConfig.from_dict(raw_config)
    return ZIVARConfig(
        backbone=legacy.backbone,
        dft_level=legacy.dft_level,
        charge_label_scheme=legacy.charge_label_scheme,
        spin_label_scheme=legacy.spin_label_scheme,
        oxidation_label_scheme=legacy.oxidation_label_scheme,
        energy_reference_scheme=legacy.energy_reference_scheme,
    )


def import_legacy_backbone(
    path: str | Path,
    *,
    target_config: ZIVARConfig | None = None,
    device: str | Any = "cpu",
    dtype: str | Any | None = None,
) -> tuple[Any, LegacyBackboneImportReport]:
    """Import only compatible backbone tensors from a known legacy checkpoint.

    This is intentionally not a checkpoint migration: the returned version-2
    model has freshly initialized variational and magnetic parameters and must
    be retrained.  Unknown schemas/revisions, incomplete backbones and shape
    changes fail closed instead of falling back to a partial load.
    """

    source_path = Path(path).expanduser().resolve()
    payload = torch.load(source_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("legacy checkpoint payload must be a mapping")
    schema = payload.get("schema")
    if schema not in LEGACY_CHECKPOINT_SCHEMAS:
        raise ValueError("backbone import accepts only a known legacy schema")
    source_revision = payload.get("architecture_revision")
    if source_revision not in LEGACY_ARCHITECTURE_REVISIONS:
        raise ValueError("legacy checkpoint architecture revision is incompatible")
    if payload.get("version") != "0.1.0":
        raise ValueError("legacy checkpoint package version is incompatible")

    config = target_config or _legacy_target_config(payload)
    if config.electronic.method != "variational":
        raise ValueError("legacy backbone import requires a variational target config")
    model = build_zivar(config, device="cpu")

    stored_manifest = payload.get("backbone_manifest")
    if not isinstance(stored_manifest, Mapping):
        raise ValueError("legacy checkpoint requires a backbone manifest")
    _validate_backbone_manifest(stored_manifest, model.backbone_manifest)

    source_state = payload.get("state_dict")
    if not isinstance(source_state, Mapping):
        raise ValueError("legacy checkpoint does not contain a state dictionary")
    source_backbone = {
        str(name): value
        for name, value in source_state.items()
        if isinstance(name, str) and name.startswith("backbone.")
    }
    target_state = model.state_dict()
    target_backbone_keys = {
        name for name in target_state if name.startswith("backbone.")
    }
    source_backbone_keys = set(source_backbone)
    missing = sorted(target_backbone_keys - source_backbone_keys)
    unexpected = sorted(source_backbone_keys - target_backbone_keys)
    if missing or unexpected:
        raise ValueError(
            "legacy backbone key set is incompatible: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    if not source_backbone_keys:
        raise ValueError("legacy checkpoint contains no backbone tensors")

    resolved_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    if resolved_dtype is None:
        resolved_dtype = next(
            (
                value.dtype
                for value in source_backbone.values()
                if torch.is_tensor(value) and value.is_floating_point()
            ),
            None,
        )
    if resolved_dtype is not None:
        model = model.to(dtype=resolved_dtype)
        target_state = model.state_dict()

    migrated_state = dict(target_state)
    imported_value_count = 0
    for name in sorted(source_backbone_keys):
        source = source_backbone[name]
        target = target_state[name]
        if not torch.is_tensor(source):
            raise ValueError(f"legacy backbone entry {name!r} is not a tensor")
        if source.shape != target.shape:
            raise ValueError(
                f"legacy backbone tensor {name!r} has shape {tuple(source.shape)!r}; "
                f"expected {tuple(target.shape)!r}"
            )
        if source.is_floating_point() != target.is_floating_point():
            raise ValueError(f"legacy backbone tensor {name!r} has incompatible dtype")
        migrated_state[name] = source.to(dtype=target.dtype)
        imported_value_count += source.numel()
    model.load_state_dict(migrated_state, strict=True)

    ignored = tuple(
        sorted(
            str(name)
            for name in source_state
            if not (isinstance(name, str) and name.startswith("backbone."))
        )
    )
    report = LegacyBackboneImportReport(
        source_checkpoint=str(source_path),
        source_schema=str(schema),
        source_architecture_revision=str(source_revision),
        target_architecture_revision=ARCHITECTURE_REVISION,
        imported_keys=tuple(sorted(source_backbone_keys)),
        ignored_legacy_keys=ignored,
        imported_tensor_count=len(source_backbone_keys),
        imported_value_count=imported_value_count,
    )
    model._legacy_backbone_import = report.to_dict()
    model._requires_retraining = True
    model = model.to(device=device, dtype=resolved_dtype).train().seal_backbone()
    return model, report


def load_zivar(
    path: str | Path,
    *,
    device: str | Any = "cpu",
    dtype: str | Any | None = None,
    strict: bool = True,
) -> Any:
    payload = torch.load(
        Path(path).expanduser().resolve(), map_location="cpu", weights_only=True
    )
    payload = _validate_current_checkpoint(payload)
    resolved_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    if resolved_dtype is None:
        resolved_dtype = next(
            (
                value.dtype
                for value in payload["state_dict"].values()
                if torch.is_tensor(value) and value.is_floating_point()
            ),
            None,
        )
    model = build_zivar(ZIVARConfig.from_dict(payload["config"]), device="cpu")
    if resolved_dtype is not None:
        model = model.to(dtype=resolved_dtype)
    stored_manifest = payload.get("backbone_manifest")
    if not isinstance(stored_manifest, Mapping):
        raise ValueError("checkpoint requires a backbone manifest")
    _validate_backbone_manifest(stored_manifest, model.backbone_manifest)
    model.load_state_dict(payload["state_dict"], strict=strict)
    model._checkpoint_capabilities = dict(payload.get("capabilities") or {})
    model._release_validation = payload.get("release_validation")
    model._release_evidence = payload.get("release_evidence")
    model._checkpoint_training_state = payload.get("training_state")
    return model.to(device=device, dtype=resolved_dtype).eval().seal_backbone()


def restore_training_state(
    trainer: Any,
    path: str | Path,
    *,
    restore_rng: bool = True,
) -> None:
    """Restore optimizer/scheduler/scaler/counters for an exact training resume."""

    payload = torch.load(
        Path(path).expanduser().resolve(), map_location="cpu", weights_only=True
    )
    payload = _validate_current_checkpoint(payload)
    model = getattr(trainer, "model", None)
    if model is None or not isinstance(getattr(model, "config", None), ZIVARConfig):
        raise TypeError("training resume requires a trainer with a ZIVAR model")
    if model.config.to_dict() != dict(payload["config"]):
        raise ValueError("trainer model configuration differs from the checkpoint")
    stored_manifest = payload.get("backbone_manifest")
    if not isinstance(stored_manifest, Mapping):
        raise ValueError("checkpoint requires a backbone manifest")
    _validate_backbone_manifest(stored_manifest, model.backbone_manifest)
    current_state = model.state_dict()
    stored_state = payload["state_dict"]
    if set(current_state) != set(stored_state):
        raise ValueError("trainer model state keys differ from the checkpoint")
    for name, current in current_state.items():
        stored = stored_state[name]
        if not torch.is_tensor(stored) or current.shape != stored.shape:
            raise ValueError(f"trainer model tensor {name!r} is incompatible")
        if current.dtype != stored.dtype or not torch.equal(
            current.detach().cpu(), stored.detach().cpu()
        ):
            raise ValueError(
                "trainer model weights differ from the checkpoint; load the "
                "checkpoint model before restoring optimizer state"
            )
    state = payload.get("training_state")
    if state is None:
        raise ValueError("checkpoint does not contain trainer state")
    trainer.load_state_dict(state)
    if restore_rng:
        rng = payload.get("rng_state")
        if rng is None:
            raise ValueError("checkpoint does not contain RNG state")
        _restore_rng_state(rng)


def inspect_zivar_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return {
        key: payload.get(key)
        for key in (
            "schema",
            "version",
            "architecture_revision",
            "numerics_revision",
            "mace_torch_version",
            "backbone_manifest",
            "epoch",
            "metrics",
            "metadata",
            "capabilities",
            "release_validation",
            "release_evidence",
        )
    }


__all__ = [
    "CHECKPOINT_SCHEMA",
    "LEGACY_CHECKPOINT_SCHEMAS",
    "LegacyBackboneImportReport",
    "import_legacy_backbone",
    "inspect_zivar_checkpoint",
    "load_zivar",
    "restore_training_state",
    "save_zivar",
]
