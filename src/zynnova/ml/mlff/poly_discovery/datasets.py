from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .features import FunctionalGroupFeaturizer
from .schema import Observation

DatasetRole = Literal[
    "language_pretraining",
    "property_learning",
    "electronic_mechanism",
    "atomistic_dynamics",
    "focused_experiment",
]


@dataclass(frozen=True, slots=True)
class PublicPolymerDataset:
    name: str
    role: DatasetRole
    modalities: tuple[str, ...]
    properties: tuple[str, ...]
    homepage: str
    citation: str
    license: str
    zynnova_adapter: str | None
    access_note: str


PUBLIC_POLYMER_DATASETS: tuple[PublicPolymerDataset, ...] = (
    PublicPolymerDataset(
        name="TransPolymer",
        role="property_learning",
        modalities=("pSMILES", "tabular properties"),
        properties=(
            "glass transition",
            "density",
            "dielectric constant",
            "band gap",
            "multiple benchmark targets",
        ),
        homepage="https://github.com/ChangwenXu98/TransPolymer",
        citation="https://doi.org/10.1038/s41524-023-01016-5",
        license="MIT repository license; verify each bundled source dataset",
        zynnova_adapter="transpolymer",
        access_note=(
            "Directly supported by ZynNova. Select a CSV and map columns explicitly; "
            "do not merge unlike measurement conditions as one label."
        ),
    ),
    PublicPolymerDataset(
        name="PI1M",
        role="language_pretraining",
        modalities=("about one million polymer SMILES",),
        properties=("unlabelled chemical space",),
        homepage="https://figshare.com/articles/dataset/12483473",
        citation="https://doi.org/10.1021/acs.jcim.0c00726",
        license="CC BY 4.0 for the Figshare release",
        zynnova_adapter=None,
        access_note=(
            "Use for representation pretraining and novelty, not as experimental "
            "evidence for a physical mechanism."
        ),
    ),
    PublicPolymerDataset(
        name="polyVERSE",
        role="property_learning",
        modalities=("pSMILES", "virtual polymers", "curated property tables"),
        properties=("electronic", "thermal", "mechanical", "sustainability"),
        homepage="https://github.com/Ramprasad-Group/polyVERSE",
        citation="https://doi.org/10.5281/zenodo.13352644",
        license="Dataset-specific; inspect the selected release",
        zynnova_adapter="polymer_table",
        access_note="Download a selected CSV and load it through polymer_table.",
    ),
    PublicPolymerDataset(
        name="Huan polymer DFT dataset",
        role="electronic_mechanism",
        modalities=("periodic structures", "uniform DFT calculations"),
        properties=("GGA band gap", "HSE06 band gap", "dielectric response"),
        homepage="https://doi.org/10.1038/sdata.2016.12",
        citation="https://doi.org/10.1038/sdata.2016.12",
        license="See Dryad/NOMAD data record terms",
        zynnova_adapter="polymer_table",
        access_note=(
            "Preferred public source for matched band-edge/dielectric mechanism work. "
            "Preserve the DFT method as a fidelity/environment field."
        ),
    ),
    PublicPolymerDataset(
        name="RadonPy database",
        role="atomistic_dynamics",
        modalities=("amorphous cells", "all-atom MD", "computed properties"),
        properties=(
            "density",
            "heat capacity",
            "thermal conductivity",
            "refractive index",
            "mechanical properties",
        ),
        homepage="https://github.com/RadonPy/RadonPy",
        citation="https://doi.org/10.1038/s41524-022-00906-4",
        license="BSD-3-Clause software; database release terms may differ",
        zynnova_adapter="polymer_table",
        access_note=(
            "Use as simulation evidence and for cross-engine validation, not as an "
            "experimental substitute."
        ),
    ),
    PublicPolymerDataset(
        name="High-entropy ferroelectric polymer source data",
        role="focused_experiment",
        modalities=("P-E loops", "dielectric", "structural", "irradiation conditions"),
        properties=(
            "recoverable energy density",
            "efficiency",
            "polarization",
            "high-entropy bond distribution",
        ),
        homepage="https://www.nature.com/articles/s41563-025-02211-z",
        citation="https://doi.org/10.1038/s41563-025-02211-z",
        license="Article/source-data terms",
        zynnova_adapter="polymer_table",
        access_note=(
            "A focused intervention dataset. It is valuable for mechanism tests but too "
            "chemically narrow to establish broad polymer-space generalization alone."
        ),
    ),
    PublicPolymerDataset(
        name="Crosslinked ferroelectric polymer source data",
        role="focused_experiment",
        modalities=("crosslink series", "DFT conformer scans", "piezoelectric response"),
        properties=(
            "d33",
            "crosslink density",
            "torsional heterogeneity",
            "phase energy landscape",
        ),
        homepage="https://www.nature.com/articles/s41467-026-69998-6",
        citation="https://doi.org/10.1038/s41467-026-69998-6",
        license="CC BY-NC-ND 4.0 article; inspect source-data terms",
        zynnova_adapter="polymer_table",
        access_note=(
            "Use crosslinker and composition as separate environments to test the "
            "flattened-energy-landscape pathway."
        ),
    ),
)


def list_public_polymer_datasets(
    *,
    role: DatasetRole | None = None,
) -> tuple[PublicPolymerDataset, ...]:
    if role is None:
        return PUBLIC_POLYMER_DATASETS
    return tuple(dataset for dataset in PUBLIC_POLYMER_DATASETS if dataset.role == role)


def public_dataset_plan(
    objective: str = "energy_storage",
) -> tuple[PublicPolymerDataset, ...]:
    """Return a fidelity-aware dataset order for a discovery campaign."""

    normalized = objective.strip().lower()
    if normalized in {"bandgap", "band_gap", "electronic"}:
        preferred = (
            "Huan polymer DFT dataset",
            "TransPolymer",
            "polyVERSE",
            "PI1M",
        )
    elif normalized in {"piezoelectric", "d33", "crosslink"}:
        preferred = (
            "Crosslinked ferroelectric polymer source data",
            "RadonPy database",
            "TransPolymer",
            "PI1M",
        )
    else:
        preferred = (
            "High-entropy ferroelectric polymer source data",
            "Huan polymer DFT dataset",
            "RadonPy database",
            "TransPolymer",
            "polyVERSE",
            "PI1M",
        )
    lookup = {dataset.name: dataset for dataset in PUBLIC_POLYMER_DATASETS}
    return tuple(lookup[name] for name in preferred)


def load_zynnova_polymer_dataset(
    dataset: str,
    *,
    root: str | Path,
    limit: int | None = None,
    **dataset_kwargs: Any,
) -> list[Any]:
    """Load a public/local dataset through ZynNova's existing dataset registry."""

    from ....data import create_dataset

    source = create_dataset(dataset, root=root, **dataset_kwargs)
    samples = source.materialize()
    return samples if limit is None else samples[: int(limit)]


def observations_from_material_samples(
    samples: Iterable[Any],
    *,
    target_names: Sequence[str] | None = None,
    target_map: Mapping[str, str] | None = None,
    mediator_names: Sequence[str] = (),
    environment_field: str | None = None,
    fidelity_field: str | None = None,
    featurizer: FunctionalGroupFeaturizer | None = None,
) -> list[Observation]:
    """Convert ZynNova ``MaterialSample`` rows into leakage-safe observations."""

    from ...polymer_utils import extract_psmiles

    featurizer = featurizer or FunctionalGroupFeaturizer()
    target_map = dict(target_map or {})
    requested = tuple(target_names or target_map or ())
    target_sources = {
        target_map.get(name, name)
        for name in requested
    }
    result: list[Observation] = []
    for sample in samples:
        psmiles = extract_psmiles(sample)
        vector = featurizer.transform(psmiles)
        labels = dict(getattr(sample, "labels", {}))
        if requested:
            targets = {
                name: labels.get(target_map.get(name, name))
                for name in requested
                if labels.get(target_map.get(name, name)) is not None
            }
        else:
            targets = {
                str(name): value
                for name, value in labels.items()
                if _is_numeric(value) and name not in mediator_names
            }
        mediators = {
            name: float(labels[name])
            for name in mediator_names
            if name in labels and _is_numeric(labels[name])
        }
        metadata = dict(getattr(sample, "metadata", {}))
        provenance = dict(getattr(sample, "provenance", {}))
        supplied_features = {
            str(name): float(value)
            for name, value in dict(getattr(sample, "features", {})).items()
            if _is_numeric(value)
            and name not in target_sources
            and name not in mediator_names
        }
        environment = _resolve_field(sample, environment_field)
        if environment is None:
            environment = provenance.get("dataset") or metadata.get("source_file") or "unknown"
        fidelity = _resolve_field(sample, fidelity_field)
        if fidelity is None:
            fidelity = metadata.get("fidelity") or provenance.get("fidelity") or "unknown"
        result.append(
            Observation(
                sample_id=str(sample.id),
                features={**vector.values, **supplied_features},
                targets={name: float(value) for name, value in targets.items()},
                mediators=mediators,
                conditions={
                    str(name): float(value)
                    for name, value in dict(getattr(sample, "conditions", {})).items()
                    if _is_numeric(value)
                },
                environment=str(environment),
                fidelity=str(fidelity),
                provenance={
                    **provenance,
                    "psmiles": psmiles,
                    "exact_chemistry_features": vector.exact_chemistry,
                    "feature_warnings": list(vector.warnings),
                },
            )
        )
    return result


def _resolve_field(sample: Any, field: str | None) -> Any | None:
    if not field:
        return None
    getter = getattr(sample, "get", None)
    return getter(field) if callable(getter) else None


def _is_numeric(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


__all__ = [
    "PUBLIC_POLYMER_DATASETS",
    "DatasetRole",
    "PublicPolymerDataset",
    "list_public_polymer_datasets",
    "load_zynnova_polymer_dataset",
    "observations_from_material_samples",
    "public_dataset_plan",
]
