from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_SYMBOLS = {
    3: "Li",
    8: "O",
    25: "Mn",
    27: "Co",
    28: "Ni",
}


@dataclass(slots=True)
class ChargePartitionLabels:
    """Atom-resolved charge labels with explicit partitioning provenance."""

    net_charges_e: np.ndarray
    scheme: str
    source: Path
    electron_populations_e: np.ndarray | None = None
    positions_A: np.ndarray | None = None
    atomic_volumes_A3: np.ndarray | None = None
    total_charge_e: float | None = None
    charge_sum_residual_e: float | None = None

    def __post_init__(self) -> None:
        self.net_charges_e = np.asarray(self.net_charges_e, dtype=float).reshape(-1)
        self.source = Path(self.source)
        if not np.all(np.isfinite(self.net_charges_e)):
            raise ValueError("partition charges must be finite")
        for name in ("electron_populations_e", "atomic_volumes_A3"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value, dtype=float).reshape(-1)
                if value.shape != self.net_charges_e.shape:
                    raise ValueError(f"{name} must contain one value per atom")
                setattr(self, name, value)
        if self.positions_A is not None:
            self.positions_A = np.asarray(self.positions_A, dtype=float)
            if self.positions_A.shape != (len(self.net_charges_e), 3):
                raise ValueError("positions_A must have shape [atoms, 3]")


def _reference_electron_array(
    reference_electrons: Mapping[int, float] | Sequence[float] | np.ndarray,
    atom_count: int,
    atomic_numbers: Sequence[int] | np.ndarray | None,
) -> np.ndarray:
    if isinstance(reference_electrons, Mapping):
        if atomic_numbers is None:
            raise ValueError("atomic_numbers are required when reference_electrons is a mapping")
        numbers = np.asarray(atomic_numbers, dtype=int).reshape(-1)
        if numbers.shape != (atom_count,):
            raise ValueError("atomic_numbers must contain one value per ACF entry")
        missing = sorted({int(z) for z in numbers if int(z) not in reference_electrons})
        if missing:
            raise ValueError(
                "missing pseudopotential reference-electron counts for Z="
                + ", ".join(map(str, missing))
            )
        return np.asarray(
            [float(reference_electrons[int(z)]) for z in numbers],
            dtype=float,
        )
    values = np.asarray(reference_electrons, dtype=float).reshape(-1)
    if values.shape != (atom_count,):
        raise ValueError(
            "reference_electrons must contain one value per atom; "
            f"got {values.shape} for {atom_count} atoms"
        )
    return values


def read_bader_acf(
    path: str | Path,
    *,
    reference_electrons: Mapping[int, float] | Sequence[float] | np.ndarray,
    atomic_numbers: Sequence[int] | np.ndarray | None = None,
    total_charge_e: float | None = None,
) -> ChargePartitionLabels:
    """Read ``ACF.dat`` and convert basin populations to net atomic charges.

    ``ACF.dat`` stores integrated electron populations. Net charges require the
    exact valence-electron count of the pseudopotential used to create the charge
    density (or an explicit per-atom all-electron reference); atomic number is
    deliberately not guessed.
    """

    source = Path(path)
    rows: list[list[float]] = []
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 7:
            continue
        try:
            index = int(fields[0])
            values = [float(value) for value in fields[1:7]]
        except ValueError:
            continue
        if index != len(rows) + 1:
            raise ValueError(
                f"non-contiguous atom index in {source}: expected {len(rows) + 1}, found {index}"
            )
        rows.append(values)
    if not rows:
        raise ValueError(f"no atom rows found in Bader ACF file: {source}")
    table = np.asarray(rows, dtype=float)
    populations = table[:, 3]
    references = _reference_electron_array(
        reference_electrons,
        len(rows),
        atomic_numbers,
    )
    charges = references - populations
    residual = None
    if total_charge_e is not None:
        residual = float(charges.sum() - float(total_charge_e))
    return ChargePartitionLabels(
        net_charges_e=charges,
        electron_populations_e=populations,
        positions_A=table[:, :3],
        atomic_volumes_A3=table[:, 5],
        scheme="bader",
        source=source,
        total_charge_e=None if total_charge_e is None else float(total_charge_e),
        charge_sum_residual_e=residual,
    )


def read_ddec_charges(
    path: str | Path,
    *,
    scheme: str = "ddec6",
    total_charge_e: float | None = None,
) -> ChargePartitionLabels:
    """Read a Chargemol DDEC net-atomic-charge XYZ file."""

    source = Path(path)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"empty DDEC file: {source}")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("DDEC charge file must start with an XYZ atom count") from exc
    charges: list[float] = []
    positions: list[list[float]] = []
    for line in lines[2:]:
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            xyz = [float(value) for value in fields[1:4]]
            charge = float(fields[-1])
        except ValueError:
            continue
        positions.append(xyz)
        charges.append(charge)
        if len(charges) == atom_count:
            break
    if len(charges) != atom_count:
        raise ValueError(f"expected {atom_count} DDEC atom rows in {source}, found {len(charges)}")
    charge_array = np.asarray(charges, dtype=float)
    residual = None
    if total_charge_e is not None:
        residual = float(charge_array.sum() - float(total_charge_e))
    return ChargePartitionLabels(
        net_charges_e=charge_array,
        positions_A=np.asarray(positions, dtype=float),
        scheme=str(scheme).lower(),
        source=source,
        total_charge_e=None if total_charge_e is None else float(total_charge_e),
        charge_sum_residual_e=residual,
    )


def attach_partition_labels(
    atoms: Any,
    labels: ChargePartitionLabels,
    *,
    array_name: str = "jouleweave_charge",
    set_initial_charges: bool = False,
) -> Any:
    """Attach validated partition charges to an ASE ``Atoms`` copy."""

    if len(atoms) != len(labels.net_charges_e):
        raise ValueError("atom count does not match the charge-label file")
    output = atoms.copy()
    output.set_array(array_name, labels.net_charges_e.copy())
    output.info["jouleweave_charge_scheme"] = labels.scheme
    output.info["jouleweave_charge_source"] = str(labels.source)
    if set_initial_charges:
        output.set_initial_charges(labels.net_charges_e)
    return output


class BaderRunner:
    """Run the Henkelman-group ``bader`` executable without shell expansion."""

    def __init__(self, executable: str | Path = "bader") -> None:
        resolved = shutil.which(str(executable))
        if resolved is None:
            candidate = Path(executable)
            if not candidate.is_file():
                raise FileNotFoundError(f"Bader executable not found: {executable!s}")
            resolved = str(candidate.resolve())
        self.executable = resolved

    def run(
        self,
        charge_density: str | Path,
        *,
        reference_density: str | Path | None = None,
        output_directory: str | Path = "bader-analysis",
        reference_electrons: Mapping[int, float] | Sequence[float] | np.ndarray,
        atomic_numbers: Sequence[int] | np.ndarray | None = None,
        total_charge_e: float | None = None,
    ) -> ChargePartitionLabels:
        density = Path(charge_density).expanduser().resolve()
        if not density.is_file():
            raise FileNotFoundError(density)
        output = Path(output_directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        command = [self.executable, str(density)]
        if reference_density is not None:
            reference = Path(reference_density).expanduser().resolve()
            if not reference.is_file():
                raise FileNotFoundError(reference)
            command.extend(["-ref", str(reference)])
        completed = subprocess.run(
            command,
            cwd=output,
            check=False,
            capture_output=True,
            text=True,
        )
        (output / "bader.stdout").write_text(completed.stdout, encoding="utf-8")
        (output / "bader.stderr").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(
                f"Bader exited with status {completed.returncode}; see {output / 'bader.stderr'}"
            )
        return read_bader_acf(
            output / "ACF.dat",
            reference_electrons=reference_electrons,
            atomic_numbers=atomic_numbers,
            total_charge_e=total_charge_e,
        )


@dataclass(slots=True)
class ChargeOxidationCalibrator:
    """Element- and scheme-specific Gaussian calibration of charge vs valence."""

    scheme: str
    means: dict[tuple[int, int], float]
    standard_deviations: dict[tuple[int, int], float]
    counts: dict[tuple[int, int], int]
    minimum_standard_deviation_e: float = 0.05

    @classmethod
    def fit(
        cls,
        atomic_numbers: Sequence[int] | np.ndarray,
        charges_e: Sequence[float] | np.ndarray,
        oxidation_states: Sequence[int] | np.ndarray,
        *,
        scheme: str,
        minimum_standard_deviation_e: float = 0.05,
    ) -> ChargeOxidationCalibrator:
        numbers = np.asarray(atomic_numbers, dtype=int).reshape(-1)
        charges = np.asarray(charges_e, dtype=float).reshape(-1)
        states = np.asarray(oxidation_states, dtype=int).reshape(-1)
        if not (numbers.shape == charges.shape == states.shape):
            raise ValueError("atomic_numbers, charges, and oxidation_states must align")
        means: dict[tuple[int, int], float] = {}
        deviations: dict[tuple[int, int], float] = {}
        counts: dict[tuple[int, int], int] = {}
        for key in sorted({(int(z), int(state)) for z, state in zip(numbers, states, strict=True)}):
            mask = (numbers == key[0]) & (states == key[1])
            values = charges[mask]
            means[key] = float(values.mean())
            deviations[key] = max(
                float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                float(minimum_standard_deviation_e),
            )
            counts[key] = int(len(values))
        return cls(
            scheme=str(scheme).lower(),
            means=means,
            standard_deviations=deviations,
            counts=counts,
            minimum_standard_deviation_e=float(minimum_standard_deviation_e),
        )

    def log_likelihood(self, atomic_number: int, state: int, charge_e: float) -> float:
        key = (int(atomic_number), int(state))
        if key not in self.means:
            return 0.0
        sigma = max(
            self.standard_deviations[key],
            self.minimum_standard_deviation_e,
        )
        normalized = (float(charge_e) - self.means[key]) / sigma
        return -0.5 * normalized * normalized - math.log(sigma)


@dataclass(slots=True)
class OxidationStateAssignment:
    states: np.ndarray
    labels: tuple[str, ...]
    site_confidence: np.ndarray
    total_oxidation_state: int
    target_total_charge: int
    score: float
    second_best_score: float | None
    ambiguity_gap: float
    combinatorially_unique: bool
    is_unique: bool
    assumptions: tuple[str, ...] = ()

    def ni_states(self, atomic_numbers: Sequence[int] | np.ndarray) -> np.ndarray:
        numbers = np.asarray(atomic_numbers, dtype=int).reshape(-1)
        return self.states[numbers == 28]


class OxidationStateResolver:
    """Charge-neutral global MAP assignment with explicit ambiguity reporting."""

    def __init__(
        self,
        *,
        minimum_assignment_gap: float = 2.0,
        minimum_site_confidence: float = 0.50,
        charge_likelihood_weight: float = 1.0,
    ) -> None:
        self.minimum_assignment_gap = float(minimum_assignment_gap)
        self.minimum_site_confidence = float(minimum_site_confidence)
        self.charge_likelihood_weight = float(charge_likelihood_weight)

    @staticmethod
    def _label(atomic_number: int, state: int) -> str:
        symbol = _SYMBOLS.get(int(atomic_number), f"Z{int(atomic_number)}")
        if state > 0:
            return f"{symbol}{state}+"
        if state < 0:
            return f"{symbol}{abs(state)}-"
        return f"{symbol}0"

    def resolve(
        self,
        atomic_numbers: Sequence[int] | np.ndarray,
        probabilities: Sequence[Sequence[float]] | np.ndarray,
        state_values: Sequence[int] | np.ndarray,
        *,
        target_total_charge: int = 0,
        allowed_states: Mapping[int, Sequence[int]] | None = None,
        fixed_states: Mapping[int, int] | None = None,
        partition_charges_e: Sequence[float] | np.ndarray | None = None,
        calibrator: ChargeOxidationCalibrator | None = None,
        assumptions: Sequence[str] = (),
    ) -> OxidationStateAssignment:
        numbers = np.asarray(atomic_numbers, dtype=int).reshape(-1)
        probability = np.asarray(probabilities, dtype=float)
        values = np.asarray(state_values, dtype=int).reshape(-1)
        if probability.shape != (len(numbers), len(values)):
            raise ValueError("probabilities must have shape [atoms, oxidation-state classes]")
        if np.any(probability < 0) or not np.all(np.isfinite(probability)):
            raise ValueError("oxidation probabilities must be finite and non-negative")
        row_sum = probability.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            raise ValueError("each atom needs at least one non-zero state probability")
        probability = probability / row_sum
        allowed = {
            int(z): tuple(int(v) for v in states) for z, states in (allowed_states or {}).items()
        }
        fixed = {int(index): int(value) for index, value in (fixed_states or {}).items()}
        charges = (
            None
            if partition_charges_e is None
            else np.asarray(partition_charges_e, dtype=float).reshape(-1)
        )
        if charges is not None and charges.shape != numbers.shape:
            raise ValueError("partition_charges_e must contain one value per atom")
        if calibrator is not None and charges is None:
            raise ValueError("a charge calibrator requires partition_charges_e")

        options: list[list[tuple[int, float, float]]] = []
        for index, atomic_number in enumerate(numbers):
            candidate_states = (
                (fixed[index],)
                if index in fixed
                else allowed.get(int(atomic_number), tuple(int(v) for v in values))
            )
            atom_options: list[tuple[int, float, float]] = []
            for state in candidate_states:
                matches = np.flatnonzero(values == state)
                if not len(matches):
                    raise ValueError(f"state {state} is not represented by the model classes")
                site_probability = float(probability[index, int(matches[0])])
                score = math.log(max(site_probability, 1.0e-12))
                if calibrator is not None:
                    score += self.charge_likelihood_weight * calibrator.log_likelihood(
                        int(atomic_number),
                        state,
                        float(charges[index]),
                    )
                atom_options.append((state, score, site_probability))
            if not atom_options:
                raise ValueError(f"no allowed oxidation states for atom {index}")
            options.append(atom_options)

        remaining_min = np.zeros(len(options) + 1, dtype=int)
        remaining_max = np.zeros(len(options) + 1, dtype=int)
        for index in range(len(options) - 1, -1, -1):
            states = [item[0] for item in options[index]]
            remaining_min[index] = remaining_min[index + 1] + min(states)
            remaining_max[index] = remaining_max[index + 1] + max(states)

        # Each record is (score, previous_sum, previous_rank, chosen_state).
        layers: list[dict[int, list[tuple[float, int | None, int | None, int | None]]]] = [
            {0: [(0.0, None, None, None)]}
        ]
        target = int(target_total_charge)
        for atom_index, atom_options in enumerate(options):
            next_layer: dict[
                int,
                list[tuple[float, int | None, int | None, int | None]],
            ] = {}
            for previous_sum, records in layers[-1].items():
                for previous_rank, record in enumerate(records):
                    for state, state_score, _probability in atom_options:
                        current_sum = previous_sum + state
                        if not (
                            current_sum + remaining_min[atom_index + 1]
                            <= target
                            <= current_sum + remaining_max[atom_index + 1]
                        ):
                            continue
                        candidate = (
                            record[0] + state_score,
                            previous_sum,
                            previous_rank,
                            state,
                        )
                        bucket = next_layer.setdefault(current_sum, [])
                        bucket.append(candidate)
                        bucket.sort(key=lambda item: item[0], reverse=True)
                        del bucket[2:]
            if not next_layer:
                raise ValueError(
                    "no charge-neutral oxidation-state assignment satisfies the "
                    "requested fixed/allowed-state constraints"
                )
            layers.append(next_layer)
        final_records = layers[-1].get(target)
        if not final_records:
            raise ValueError(f"no oxidation-state assignment sums to target charge {target}")

        def reconstruct(rank: int) -> np.ndarray:
            assignment = np.zeros(len(numbers), dtype=int)
            current_sum = target
            current_rank = rank
            for atom_index in range(len(numbers) - 1, -1, -1):
                record = layers[atom_index + 1][current_sum][current_rank]
                assignment[atom_index] = int(record[3])
                current_sum = int(record[1])
                current_rank = int(record[2])
            return assignment

        best = reconstruct(0)
        best_score = float(final_records[0][0])
        second_score = float(final_records[1][0]) if len(final_records) > 1 else None
        gap = math.inf if second_score is None else best_score - second_score
        confidence = np.empty(len(numbers), dtype=float)
        for index, state in enumerate(best):
            state_probability = {
                candidate_state: site_probability
                for candidate_state, _score, site_probability in options[index]
            }
            denominator = sum(state_probability.values())
            confidence[index] = state_probability[int(state)] / max(denominator, 1.0e-12)
        combinatorially_unique = second_score is None
        is_unique = bool(
            (combinatorially_unique or gap >= self.minimum_assignment_gap)
            and np.all(confidence >= self.minimum_site_confidence)
        )
        return OxidationStateAssignment(
            states=best,
            labels=tuple(
                self._label(int(z), int(state)) for z, state in zip(numbers, best, strict=True)
            ),
            site_confidence=confidence,
            total_oxidation_state=int(best.sum()),
            target_total_charge=target,
            score=best_score,
            second_best_score=second_score,
            ambiguity_gap=float(gap),
            combinatorially_unique=combinatorially_unique,
            is_unique=is_unique,
            assumptions=tuple(str(value) for value in assumptions),
        )

    def resolve_ncm(
        self,
        atomic_numbers: Sequence[int] | np.ndarray,
        probabilities: Sequence[Sequence[float]] | np.ndarray,
        state_values: Sequence[int] | np.ndarray,
        *,
        target_total_charge: int = 0,
        oxygen_redox: bool = False,
        variable_co_mn: bool = False,
        partition_charges_e: Sequence[float] | np.ndarray | None = None,
        calibrator: ChargeOxidationCalibrator | None = None,
    ) -> OxidationStateAssignment:
        """Resolve NCM valences under an explicit, inspectable chemistry prior."""

        allowed: dict[int, tuple[int, ...]] = {
            3: (1,),
            8: (-2, -1) if oxygen_redox else (-2,),
            28: (2, 3, 4),
            27: (2, 3, 4) if variable_co_mn else (3,),
            25: (2, 3, 4) if variable_co_mn else (4,),
        }
        assumptions = [
            "Li constrained to +1",
            "Ni allowed in {+2,+3,+4}",
            ("oxygen redox allowed in {-2,-1}" if oxygen_redox else "oxygen constrained to -2"),
            (
                "Co/Mn allowed in {+2,+3,+4}"
                if variable_co_mn
                else "Co constrained to +3 and Mn to +4"
            ),
        ]
        return self.resolve(
            atomic_numbers,
            probabilities,
            state_values,
            target_total_charge=target_total_charge,
            allowed_states=allowed,
            partition_charges_e=partition_charges_e,
            calibrator=calibrator,
            assumptions=assumptions,
        )


__all__ = [
    "BaderRunner",
    "ChargeOxidationCalibrator",
    "ChargePartitionLabels",
    "OxidationStateAssignment",
    "OxidationStateResolver",
    "attach_partition_labels",
    "read_bader_acf",
    "read_ddec_charges",
]
