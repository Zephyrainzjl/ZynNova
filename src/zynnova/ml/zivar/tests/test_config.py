from __future__ import annotations

from dataclasses import replace

import pytest

from zynnova.ml.zivar.config import (
    ARCHITECTURE_REVISION,
    NUMERICS_REVISION,
    ElectronicConfig,
    ElectrostaticsConfig,
    OxidationConfig,
    SpinConfig,
    ZIVARConfig,
)


def test_public_version_and_new_architecture_revision() -> None:
    config = ZIVARConfig.balanced(dft_level="PBE")
    assert config.to_dict()["dft_level"] == "PBE"
    assert ARCHITECTURE_REVISION == "zivar-variational-electrospin-2"
    assert "variational" in ARCHITECTURE_REVISION
    assert NUMERICS_REVISION == "variational-scf-pme.1"


@pytest.mark.parametrize(
    "method", ["variational", "polar", "direct", "fukui_auxiliary", "qeq"]
)
def test_all_stable_electronic_methods_roundtrip(method: str) -> None:
    if method == "qeq":
        config = ZIVARConfig.qeq(backbone__kind="convolution", backbone__max_ell=0)
    elif method == "direct":
        config = ZIVARConfig.direct_heads(
            backbone__kind="convolution", backbone__max_ell=0
        )
    elif method == "variational":
        config = ZIVARConfig.convolution()
    else:
        config = ZIVARConfig.convolution(
            electronic__method=method,
            electronic__polarization_updates=1,
            electronic__energy_coupling=("full" if method == "polar" else "learned"),
        )
    restored = ZIVARConfig.from_dict(config.to_dict())
    assert restored == config
    assert restored.electronic.method == method
    assert restored.backbone.kind == "convolution"
    assert restored.backbone.max_ell == 0


def test_polar_requires_a_feedforward_update() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ElectronicConfig(method="polar", polarization_updates=0)


def test_variational_path_uses_convergence_not_fixed_depth() -> None:
    assert ZIVARConfig.production().electronic.polarization_updates == 0
    with pytest.raises(ValueError, match="SCF convergence"):
        ElectronicConfig(method="variational", polarization_updates=1)


def test_pme_order_is_never_silently_downgraded() -> None:
    assert ElectrostaticsConfig(interpolation_order=6).interpolation_order == 6
    with pytest.raises(ValueError, match="2, 4 or 6"):
        ElectrostaticsConfig(interpolation_order=5)


def test_production_physics_cannot_be_silently_disabled() -> None:
    with pytest.raises(ValueError, match="energy_coupling='full'"):
        ElectronicConfig(method="polar", energy_coupling="learned")
    with pytest.raises(ValueError, match="requires explicit spin vectors"):
        SpinConfig(mode="spin_lattice", require_spin_input=False)
    with pytest.raises(ValueError, match="requires induced_magnetization=True"):
        ZIVARConfig(spin=replace(SpinConfig(), induced_magnetization=False))
    with pytest.raises(ValueError, match="cannot run beside"):
        ZIVARConfig(
            spin=SpinConfig(
                mode="magnitude_auxiliary",
                require_spin_input=False,
            )
        )
    with pytest.raises(ValueError, match="fixed_charge"):
        ZIVARConfig(
            electronic=replace(ElectronicConfig(), boundary_mode="fixed_potential")
        )
    with pytest.raises(ValueError, match="formal-label source"):
        OxidationConfig(enabled=True, label_source="rounded_charge")
    with pytest.raises(ValueError, match="do not implement explicit electrostatic"):
        ElectronicConfig(
            method="direct",
            energy_coupling="full",
            density_lmax=0,
            potential_lmax=0,
            polarization_updates=0,
        )


def test_convolution_preset_is_independent_of_angular_capacity() -> None:
    config = ZIVARConfig.convolution()
    assert config.backbone.max_ell == 0
    assert config.electronic.density_lmax == 2
    with pytest.raises(ValueError, match="max_ell"):
        replace(config.backbone, max_ell=5)
