from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from zynnova.ml.zivar.checkpoint import (
    LEGACY_CHECKPOINT_SCHEMAS,
    LegacyBackboneImportReport,
    import_legacy_backbone,
    load_zivar,
)
from zynnova.ml.zivar.config import LEGACY_ARCHITECTURE_REVISIONS, ZIVARConfig
from zynnova.ml.zivar.model import build_zivar


def _legacy_config() -> ZIVARConfig:
    return ZIVARConfig.direct_heads(
        dft_level="legacy-test",
        backbone__atomic_numbers=(3, 8),
        backbone__kind="convolution",
        backbone__channels=8,
        backbone__max_ell=0,
        backbone__correlation=1,
        backbone__num_interactions=1,
        backbone__num_bessel=2,
        backbone__radial_mlp=(8,),
        electronic__hidden=(8,),
        electronic__oxidation__enabled=False,
        spin__hidden=(8,),
    )


def _legacy_payload() -> dict[str, object]:
    config = _legacy_config()
    model = build_zivar(config)
    state = {}
    for name, value in model.state_dict().items():
        if name.startswith("backbone.") and value.is_floating_point():
            state[name] = torch.full_like(value, 0.125)
        elif not name.startswith("backbone.") and value.is_floating_point():
            state[name] = torch.full_like(value, 77.0)
        else:
            state[name] = value.detach().clone()
    return {
        "schema": LEGACY_CHECKPOINT_SCHEMAS[0],
        "version": "0.1.0",
        "architecture_revision": LEGACY_ARCHITECTURE_REVISIONS[0],
        "config": config.to_dict(),
        "backbone_manifest": model.backbone_manifest,
        "state_dict": state,
    }


def _save(payload: dict[str, object], path: Path) -> Path:
    torch.save(payload, path)
    return path


def test_imports_only_exact_backbone_and_marks_retraining(tmp_path: Path) -> None:
    payload = _legacy_payload()
    path = _save(payload, tmp_path / "legacy.pt")

    model, report = import_legacy_backbone(path)

    assert isinstance(report, LegacyBackboneImportReport)
    assert report.requires_retraining is True
    assert model._requires_retraining is True
    assert model.config.electronic.method == "variational"
    assert report.imported_keys
    assert all(name.startswith("backbone.") for name in report.imported_keys)
    assert any(name.startswith("magnetic.") for name in report.ignored_legacy_keys)
    assert any(name.startswith("electronic.") for name in report.ignored_legacy_keys)
    assert report.to_dict()["requires_retraining"] is True

    source_state = payload["state_dict"]
    assert isinstance(source_state, dict)
    target_state = model.state_dict()
    for name in report.imported_keys:
        torch.testing.assert_close(target_state[name], source_state[name])
    magnetic_name = next(
        name
        for name, value in target_state.items()
        if name.startswith("magnetic.") and value.is_floating_point()
    )
    assert not torch.all(target_state[magnetic_name] == 77.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "zivar-unknown-checkpoint", "known legacy schema"),
        ("architecture_revision", "zivar-unknown-legacy", "revision"),
        ("version", "999.0.0", "package version"),
    ],
)
def test_import_fails_closed_for_unknown_legacy_identity(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    payload = _legacy_payload()
    payload[field] = value
    path = _save(payload, tmp_path / f"unknown-{field}.pt")
    with pytest.raises(ValueError, match=message):
        import_legacy_backbone(path)


def test_import_rejects_missing_or_reshaped_backbone_state(tmp_path: Path) -> None:
    missing_payload = _legacy_payload()
    missing_state = missing_payload["state_dict"]
    assert isinstance(missing_state, dict)
    missing_key = next(name for name in missing_state if name.startswith("backbone."))
    del missing_state[missing_key]
    with pytest.raises(ValueError, match="key set"):
        import_legacy_backbone(_save(missing_payload, tmp_path / "missing.pt"))

    shaped_payload = _legacy_payload()
    shaped_state = shaped_payload["state_dict"]
    assert isinstance(shaped_state, dict)
    shaped_key = next(
        name
        for name, value in shaped_state.items()
        if name.startswith("backbone.") and value.ndim >= 1 and value.shape[0] > 1
    )
    shaped_state[shaped_key] = shaped_state[shaped_key][:-1]
    with pytest.raises(ValueError, match="shape"):
        import_legacy_backbone(_save(shaped_payload, tmp_path / "shape.pt"))


def test_normal_loader_still_rejects_legacy_checkpoint(tmp_path: Path) -> None:
    path = _save(_legacy_payload(), tmp_path / "legacy.pt")
    with pytest.raises(ValueError, match="not physically equivalent"):
        load_zivar(path)
