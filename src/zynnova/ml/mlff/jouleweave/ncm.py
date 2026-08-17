from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ....dynamics.adapters import to_ase_atoms

LI_SUBLATTICE = 1
TM_SUBLATTICE = 2
O_SUBLATTICE = 3
SITE_ROLE_ARRAY = "jouleweave_site_role"


@dataclass(slots=True)
class NCMEnumerationConfig:
    """Finite, deterministic enumeration controls for layered NCM90."""

    ni_fraction: float = 0.90
    co_fraction: float = 0.05
    mn_fraction: float = 0.05
    antisite_pairs: tuple[int, ...] = (0, 1)
    decorate_transition_metals: bool = True
    max_structures_per_x: int = 128
    composition_tolerance: float = 0.025
    seed: int = 42

    def __post_init__(self) -> None:
        fractions = (self.ni_fraction, self.co_fraction, self.mn_fraction)
        if any(value < 0 for value in fractions):
            raise ValueError("NCM fractions cannot be negative")
        if not math.isclose(sum(fractions), 1.0, abs_tol=1.0e-10):
            raise ValueError("Ni/Co/Mn fractions must sum to 1")
        if self.max_structures_per_x < 1:
            raise ValueError("max_structures_per_x must be positive")
        if self.composition_tolerance < 0:
            raise ValueError("composition_tolerance cannot be negative")
        if any(int(value) < 0 for value in self.antisite_pairs):
            raise ValueError("antisite pair counts cannot be negative")
        self.antisite_pairs = tuple(sorted({int(value) for value in self.antisite_pairs}))


@dataclass(slots=True)
class NCMConfiguration:
    atoms: Any
    x_li: float
    n_li: int
    n_transition_metals: int
    vacancy_site_indices: tuple[int, ...]
    antisite_li_site_indices: tuple[int, ...]
    antisite_ni_site_indices: tuple[int, ...]
    transition_metal_counts: dict[str, int]
    fingerprint: tuple[Any, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def antisite_pair_count(self) -> int:
        return len(self.antisite_li_site_indices)


def _sample_combinations(
    values: Sequence[int],
    count: int,
    limit: int,
    rng: np.random.Generator,
) -> list[tuple[int, ...]]:
    values = tuple(int(value) for value in values)
    if count < 0 or count > len(values):
        return []
    total = math.comb(len(values), count)
    if total <= limit:
        return list(itertools.combinations(values, count))
    chosen: set[tuple[int, ...]] = set()
    while len(chosen) < limit:
        selection = tuple(
            sorted(int(value) for value in rng.choice(values, size=count, replace=False).tolist())
        )
        chosen.add(selection)
    return sorted(chosen)


def _largest_remainder_counts(
    total: int,
    fractions: Sequence[float],
) -> tuple[int, ...]:
    raw = np.asarray(fractions, dtype=float) * int(total)
    counts = np.floor(raw).astype(int)
    remaining = int(total) - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    for index in order[:remaining]:
        counts[int(index)] += 1
    return tuple(int(value) for value in counts)


def ncm_mixing_statistics(atoms: Any) -> dict[str, float | int]:
    """Count Li/Ni antisites using immutable parent-sublattice labels."""

    if SITE_ROLE_ARRAY not in atoms.arrays:
        raise ValueError(f"structure does not contain the {SITE_ROLE_ARRAY!r} site-role array")
    roles = np.asarray(atoms.arrays[SITE_ROLE_ARRAY], dtype=int)
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    ni_on_li = int(np.sum((roles == LI_SUBLATTICE) & (numbers == 28)))
    li_on_tm = int(np.sum((roles == TM_SUBLATTICE) & (numbers == 3)))
    occupied_li_sites = int(np.sum(roles == LI_SUBLATTICE))
    tm_sites = int(np.sum(roles == TM_SUBLATTICE))
    original_li_sites = int(atoms.info.get("jouleweave_original_li_site_count", occupied_li_sites))
    return {
        "ni_on_li_sites": ni_on_li,
        "li_on_tm_sites": li_on_tm,
        "paired_antisites": min(ni_on_li, li_on_tm),
        "antisite_balance_error": abs(ni_on_li - li_on_tm),
        "ni_on_li_fraction_of_original_sites": (
            ni_on_li / original_li_sites if original_li_sites else 0.0
        ),
        "li_on_tm_fraction": li_on_tm / tm_sites if tm_sites else 0.0,
    }


class NCMCompositionEnumerator:
    """Generate LixNCM90 vacancies and composition-conserving Li/Ni antisites.

    The input must be a fully lithiated layered parent with one explicit Li site
    per transition-metal site. A parent supercell with 20, 40, ... TM sites
    represents the default 90:5:5 composition exactly.
    """

    def __init__(
        self,
        parent: Any,
        config: NCMEnumerationConfig | None = None,
    ) -> None:
        self.config = config or NCMEnumerationConfig()
        self.parent = to_ase_atoms(parent).copy()
        if not bool(np.all(np.asarray(self.parent.pbc, dtype=bool))):
            raise ValueError("NCM enumeration requires a fully periodic parent")
        numbers = np.asarray(self.parent.get_atomic_numbers(), dtype=int)
        self.li_sites = tuple(int(value) for value in np.flatnonzero(numbers == 3))
        self.tm_sites = tuple(
            int(value) for value in np.flatnonzero(np.isin(numbers, (25, 27, 28)))
        )
        self.oxygen_sites = tuple(int(value) for value in np.flatnonzero(numbers == 8))
        if not self.li_sites or not self.tm_sites or not self.oxygen_sites:
            raise ValueError("parent must contain Li, O, and Ni/Co/Mn sites")
        if len(self.li_sites) != len(self.tm_sites):
            raise ValueError("fully lithiated layered parent must have one Li site per TM site")
        roles = np.zeros(len(self.parent), dtype=np.int8)
        roles[list(self.li_sites)] = LI_SUBLATTICE
        roles[list(self.tm_sites)] = TM_SUBLATTICE
        roles[list(self.oxygen_sites)] = O_SUBLATTICE
        if np.any(roles == 0):
            raise ValueError("NCM90 parent may contain only Li, O, Ni, Co, and Mn")
        self.parent.set_array(SITE_ROLE_ARRAY, roles)
        self.parent.info["jouleweave_original_li_site_count"] = len(self.li_sites)
        self._rng = np.random.default_rng(self.config.seed)

    def _target_counts(self) -> dict[str, int]:
        counts = _largest_remainder_counts(
            len(self.tm_sites),
            (
                self.config.ni_fraction,
                self.config.co_fraction,
                self.config.mn_fraction,
            ),
        )
        actual = np.asarray(counts, dtype=float) / len(self.tm_sites)
        desired = np.asarray(
            (
                self.config.ni_fraction,
                self.config.co_fraction,
                self.config.mn_fraction,
            )
        )
        if float(np.max(np.abs(actual - desired))) > self.config.composition_tolerance:
            raise ValueError(
                "parent supercell cannot represent the requested NCM ratio within "
                f"tolerance {self.config.composition_tolerance}; use a supercell "
                "with at least 20 TM sites for exact NCM90 (90:5:5)"
            )
        return {"Ni": counts[0], "Co": counts[1], "Mn": counts[2]}

    def _tm_decorations(self, limit: int) -> list[dict[int, str]]:
        target = self._target_counts()
        current_symbols = self.parent.get_chemical_symbols()
        if not self.config.decorate_transition_metals:
            current = {
                symbol: sum(current_symbols[index] == symbol for index in self.tm_sites)
                for symbol in ("Ni", "Co", "Mn")
            }
            if current != target:
                raise ValueError(
                    f"parent TM composition {current} does not match requested {target}"
                )
            return [{index: current_symbols[index] for index in self.tm_sites}]

        co_choices = _sample_combinations(
            self.tm_sites,
            target["Co"],
            max(limit, 1),
            self._rng,
        )
        decorations: list[dict[int, str]] = []
        for co_sites in co_choices:
            remaining = tuple(index for index in self.tm_sites if index not in co_sites)
            mn_choices = _sample_combinations(
                remaining,
                target["Mn"],
                max(limit - len(decorations), 1),
                self._rng,
            )
            for mn_sites in mn_choices:
                co_set = set(co_sites)
                mn_set = set(mn_sites)
                decorations.append(
                    {
                        index: ("Co" if index in co_set else "Mn" if index in mn_set else "Ni")
                        for index in self.tm_sites
                    }
                )
                if len(decorations) >= limit:
                    return decorations
        return decorations

    @staticmethod
    def _fingerprint(atoms: Any) -> tuple[Any, ...]:
        scaled = np.mod(np.asarray(atoms.get_scaled_positions(wrap=True)), 1.0)
        records = sorted(
            (
                int(z),
                int(role),
                *(float(value) for value in np.round(position, 8)),
            )
            for z, role, position in zip(
                atoms.get_atomic_numbers(),
                atoms.arrays[SITE_ROLE_ARRAY],
                scaled,
                strict=True,
            )
        )
        return tuple(records)

    def enumerate(
        self,
        li_fractions: Iterable[float],
        *,
        antisite_pairs: Sequence[int] | None = None,
        max_structures_per_x: int | None = None,
    ) -> list[NCMConfiguration]:
        pair_counts = (
            self.config.antisite_pairs
            if antisite_pairs is None
            else tuple(sorted({int(value) for value in antisite_pairs}))
        )
        if any(value < 0 for value in pair_counts):
            raise ValueError("antisite pair counts cannot be negative")
        limit = int(max_structures_per_x or self.config.max_structures_per_x)
        if limit < 1:
            raise ValueError("max_structures_per_x must be positive")
        decorations = self._tm_decorations(limit)
        target_counts = self._target_counts()
        configurations: list[NCMConfiguration] = []

        for requested_x in li_fractions:
            x_value = float(requested_x)
            if not 0.0 <= x_value <= 1.0:
                raise ValueError("Li fraction x must lie in [0, 1]")
            n_li = int(round(x_value * len(self.tm_sites)))
            actual_x = n_li / len(self.tm_sites)
            vacancy_count = len(self.li_sites) - n_li
            vacancy_choices = _sample_combinations(
                self.li_sites,
                vacancy_count,
                limit,
                self._rng,
            )
            seen: set[tuple[Any, ...]] = set()
            produced = 0
            for decoration in decorations:
                ni_sites = tuple(index for index, symbol in decoration.items() if symbol == "Ni")
                for vacancies in vacancy_choices:
                    occupied_li = tuple(index for index in self.li_sites if index not in vacancies)
                    for pair_count in pair_counts:
                        if pair_count > min(len(occupied_li), len(ni_sites)):
                            continue
                        li_choices = _sample_combinations(
                            occupied_li,
                            pair_count,
                            limit,
                            self._rng,
                        )
                        ni_choices = _sample_combinations(
                            ni_sites,
                            pair_count,
                            limit,
                            self._rng,
                        )
                        for li_antisites, ni_antisites in itertools.product(
                            li_choices,
                            ni_choices,
                        ):
                            atoms = self.parent.copy()
                            symbols = atoms.get_chemical_symbols()
                            for index, symbol in decoration.items():
                                symbols[index] = symbol
                            for index in li_antisites:
                                symbols[index] = "Ni"
                            for index in ni_antisites:
                                symbols[index] = "Li"
                            atoms.set_chemical_symbols(symbols)
                            metadata = {
                                "x_li": actual_x,
                                "requested_x_li": x_value,
                                "n_li": n_li,
                                "n_transition_metals": len(self.tm_sites),
                                "vacancy_site_indices": tuple(vacancies),
                                "antisite_li_site_indices": tuple(li_antisites),
                                "antisite_ni_site_indices": tuple(ni_antisites),
                                "transition_metal_counts": dict(target_counts),
                            }
                            atoms.info["jouleweave_ncm"] = metadata
                            atoms.info["jouleweave_original_li_site_count"] = len(self.li_sites)
                            if vacancies:
                                del atoms[list(sorted(vacancies, reverse=True))]
                            fingerprint = self._fingerprint(atoms)
                            if fingerprint in seen:
                                continue
                            seen.add(fingerprint)
                            configurations.append(
                                NCMConfiguration(
                                    atoms=atoms,
                                    x_li=actual_x,
                                    n_li=n_li,
                                    n_transition_metals=len(self.tm_sites),
                                    vacancy_site_indices=tuple(vacancies),
                                    antisite_li_site_indices=tuple(li_antisites),
                                    antisite_ni_site_indices=tuple(ni_antisites),
                                    transition_metal_counts=dict(target_counts),
                                    fingerprint=fingerprint,
                                    metadata=metadata,
                                )
                            )
                            produced += 1
                            if produced >= limit:
                                break
                        if produced >= limit:
                            break
                    if produced >= limit:
                        break
                if produced >= limit:
                    break
        return configurations


__all__ = [
    "LI_SUBLATTICE",
    "NCMCompositionEnumerator",
    "NCMConfiguration",
    "NCMEnumerationConfig",
    "O_SUBLATTICE",
    "SITE_ROLE_ARRAY",
    "TM_SUBLATTICE",
    "ncm_mixing_statistics",
]
