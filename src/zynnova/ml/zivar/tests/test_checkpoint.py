from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from zynnova.ml.zivar.checkpoint import (
    CHECKPOINT_SCHEMA,
    inspect_zivar_checkpoint,
    load_zivar,
    save_zivar,
)
from zynnova.ml.zivar.config import (
    ARCHITECTURE_REVISION,
    NUMERICS_REVISION,
    ZIVAR_VERSION,
    ZIVARConfig,
)
from zynnova.ml.zivar.model import build_zivar


def test_convolution_checkpoint_does_not_require_mace(tmp_path: object) -> None:
    config = ZIVARConfig.convolution(
        dft_level="test",
        backbone__atomic_numbers=(3, 8),
        backbone__channels=8,
        backbone__num_interactions=1,
        backbone__num_bessel=2,
        backbone__radial_mlp=(8,),
        electronic__hidden=(8,),
        electronic__polarization_updates=0,
        electronic__oxidation__enabled=False,
    )
    model = build_zivar(config)
    path = save_zivar(model, tmp_path / "model.pt")
    inspected = inspect_zivar_checkpoint(path)
    assert inspected["numerics_revision"] == "variational-scf-pme.1"
    assert inspected["capabilities"]["production_validated"] is False
    restored = load_zivar(path)
    assert restored.backbone_manifest == model.backbone_manifest


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", "999.0.0", "package version"),
        ("architecture_revision", "legacy-in-current-schema", "architecture"),
        ("numerics_revision", "unknown-numerics", "numerics"),
    ],
)
def test_current_loader_rejects_forged_identity(
    tmp_path: object, field: str, value: str, message: str
) -> None:
    config = ZIVARConfig.convolution(
        dft_level="identity-test",
        backbone__atomic_numbers=(3, 8),
        backbone__channels=8,
        backbone__num_interactions=1,
        backbone__num_bessel=2,
        backbone__radial_mlp=(8,),
        electronic__hidden=(8,),
        electronic__oxidation__enabled=False,
    )
    path = save_zivar(build_zivar(config), tmp_path / "valid.pt")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == CHECKPOINT_SCHEMA
    assert payload["version"] == ZIVAR_VERSION
    assert payload["architecture_revision"] == ARCHITECTURE_REVISION
    assert payload["numerics_revision"] == NUMERICS_REVISION
    payload[field] = value
    forged = tmp_path / f"forged-{field}.pt"
    torch.save(payload, forged)
    with pytest.raises(ValueError, match=message):
        load_zivar(forged)
