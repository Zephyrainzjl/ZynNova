"""Command-line diagnostics and deployment entry point."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def _runtime(_: argparse.Namespace) -> int:
    from .backbone import SOURCE_CONTRACT, verify_mace_runtime

    print(
        json.dumps(
            {"mace_torch": verify_mace_runtime(), "contract": asdict(SOURCE_CONTRACT)},
            indent=2,
        )
    )
    return 0


def _inspect(arguments: argparse.Namespace) -> int:
    from .checkpoint import inspect_zivar_checkpoint

    print(json.dumps(inspect_zivar_checkpoint(arguments.checkpoint), indent=2, default=str))
    return 0


def _export(arguments: argparse.Namespace) -> int:
    from .lammps import export_zivar_lammps_bundle

    element_map = tuple(int(value) for value in arguments.elements.split(","))
    bundle = export_zivar_lammps_bundle(
        arguments.checkpoint,
        arguments.directory,
        element_map,
        data_file=arguments.data_file,
        run_steps=arguments.steps,
    )
    print(bundle.directory)
    return 0


def _export_local(arguments: argparse.Namespace) -> int:
    from .checkpoint import load_zivar
    from .lammps import export_local_backbone_mliap

    model = load_zivar(
        arguments.checkpoint,
        device=arguments.device,
        dtype=arguments.dtype,
    )
    print(export_local_backbone_mliap(model, arguments.directory))
    return 0


def _audit(arguments: argparse.Namespace) -> int:
    from .maturity import assess_maturity, evidence_template

    package_root = Path(__file__).resolve().parent
    if arguments.template is not None:
        arguments.template.write_text(
            json.dumps(evidence_template(package_root), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    report = assess_maturity(package_root, arguments.evidence)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.production_ready else 2


def _run_gate(arguments: argparse.Namespace) -> int:
    from .gates import run_gate

    result = run_gate(arguments.name, arguments.artifact)
    payload = result.to_dict()
    payload["evidence_record"] = result.evidence_record()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.status == "pass" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zivar")
    commands = parser.add_subparsers(dest="command", required=True)
    runtime = commands.add_parser("check-runtime", help="verify reviewed upstream runtime")
    runtime.set_defaults(handler=_runtime)
    inspect = commands.add_parser("inspect", help="inspect a checkpoint")
    inspect.add_argument("checkpoint", type=Path)
    inspect.set_defaults(handler=_inspect)
    export = commands.add_parser(
        "export-lammps", help="export the full stable global-model bundle"
    )
    export.add_argument("checkpoint", type=Path)
    export.add_argument("directory", type=Path)
    export.add_argument("--elements", required=True, help="atomic numbers by LAMMPS type")
    export.add_argument("--data-file", default="system.data")
    export.add_argument("--steps", type=int, default=0)
    export.set_defaults(handler=_export)
    local = commands.add_parser(
        "export-local-mliap",
        help="export the local MACE backbone only with the official converter",
    )
    local.add_argument("checkpoint", type=Path)
    local.add_argument("directory", type=Path)
    local.add_argument("--device", default="cuda")
    local.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    local.set_defaults(handler=_export_local)
    audit = commands.add_parser(
        "audit-release", help="run fail-closed source and target maturity gates"
    )
    audit.add_argument("--evidence", type=Path)
    audit.add_argument("--template", type=Path)
    audit.set_defaults(handler=_audit)
    from .gates import FIXED_GATE_REGISTRY

    gate = commands.add_parser(
        "run-gate",
        help="run one fixed maturity gate and write its hash-bound artifact",
    )
    gate.add_argument("name", choices=tuple(FIXED_GATE_REGISTRY))
    gate.add_argument("--artifact", required=True, type=Path)
    gate.set_defaults(handler=_run_gate)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
