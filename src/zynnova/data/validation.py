from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .record import MaterialSample


@dataclass(slots=True)
class DatasetIssue:
    sample_id: str
    severity: str
    message: str
    field: str | None = None


@dataclass(slots=True)
class DatasetReport:
    count: int = 0
    material_types: Counter[str] = field(default_factory=Counter)
    splits: Counter[str] = field(default_factory=Counter)
    duplicate_ids: list[str] = field(default_factory=list)
    missing_fields: Counter[str] = field(default_factory=Counter)
    issues: list[DatasetIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "material_types": dict(self.material_types),
            "splits": dict(self.splits),
            "duplicate_ids": list(self.duplicate_ids),
            "missing_fields": dict(self.missing_fields),
            "valid": self.valid,
            "issues": [asdict(issue) for issue in self.issues],
        }


def validate_dataset(
    samples: Iterable[MaterialSample],
    *,
    required_fields: Sequence[str] = (),
    require_structure: bool = False,
    check_finite: bool = True,
    max_issues: int = 1000,
) -> DatasetReport:
    report = DatasetReport()
    seen: set[str] = set()
    for sample in samples:
        report.count += 1
        report.material_types[sample.material_type.value] += 1
        report.splits[sample.split or "unspecified"] += 1
        if sample.id in seen:
            report.duplicate_ids.append(sample.id)
            _issue(report, sample.id, "error", "duplicate sample id", max_issues=max_issues)
        seen.add(sample.id)
        if require_structure and sample.structure is None:
            _issue(
                report,
                sample.id,
                "error",
                "missing structure",
                field="structure",
                max_issues=max_issues,
            )
        for path in required_fields:
            value = sample.get(path)
            if value is None:
                report.missing_fields[path] += 1
                _issue(
                    report,
                    sample.id,
                    "error",
                    "required field is missing",
                    field=path,
                    max_issues=max_issues,
                )
        if check_finite:
            for root in ("features", "labels", "conditions"):
                _check_mapping_finite(sample.id, root, getattr(sample, root), report, max_issues)
    return report


def _check_mapping_finite(
    sample_id: str,
    prefix: str,
    mapping: dict[str, Any],
    report: DatasetReport,
    max_issues: int,
) -> None:
    for name, value in mapping.items():
        path = f"{prefix}.{name}"
        if isinstance(value, dict):
            _check_mapping_finite(sample_id, path, value, report, max_issues)
            continue
        if value is None or isinstance(value, (str, bytes, bool)):
            continue
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            continue
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            _issue(
                report,
                sample_id,
                "warning",
                "field contains NaN or Inf",
                field=path,
                max_issues=max_issues,
            )


def _issue(
    report: DatasetReport,
    sample_id: str,
    severity: str,
    message: str,
    *,
    field: str | None = None,
    max_issues: int,
) -> None:
    if len(report.issues) < max_issues:
        report.issues.append(DatasetIssue(sample_id, severity, message, field))
