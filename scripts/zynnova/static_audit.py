"""Run a conservative static audit over ZynNova-owned Python and metadata files."""

from __future__ import annotations

import argparse
import ast
import json
import re
import tomllib
from pathlib import Path
from typing import Iterable, Sequence


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return ""


def _python_files(root: Path) -> list[Path]:
    package = root / "src/zynnova"
    scan_roots = (
        package / "core",
        package / "geometry",
        package / "zynform",
        package / "zynmorph",
        package / "zynvista",
        package / "zynvox",
        root / "examples/zynnova",
        root / "scripts/zynnova",
        root / "tests/zynnova",
    )
    files = {path for directory in scan_roots for path in directory.rglob("*.py")}
    files.update({package / "__init__.py", package / "__main__.py", package / "cli.py"})
    return sorted(files)



def _locations(items: Iterable[tuple[Path, int]], root: Path) -> list[dict[str, object]]:
    return [
        {"file": path.relative_to(root).as_posix(), "line": line}
        for path, line in items
    ]


def audit(root: Path) -> dict[str, object]:
    syntax_errors: list[dict[str, object]] = []
    shell_true: list[tuple[Path, int]] = []
    os_system: list[tuple[Path, int]] = []
    popen: list[tuple[Path, int]] = []
    builtin_eval: list[tuple[Path, int]] = []
    builtin_exec: list[tuple[Path, int]] = []
    unsafe_pickle: list[tuple[Path, int]] = []
    absolute_paths: list[dict[str, object]] = []
    files = _python_files(root)
    line_count = 0

    for path in files:
        text = path.read_text(encoding="utf-8")
        line_count += len(text.splitlines())
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            syntax_errors.append(
                {
                    "file": path.relative_to(root).as_posix(),
                    "line": exc.lineno,
                    "message": exc.msg,
                }
            )
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _dotted_name(node.func)
                location = (path, node.lineno)
                if name == "os.system":
                    os_system.append(location)
                if name in {"subprocess.Popen", "Popen"}:
                    popen.append(location)
                if (
                    isinstance(node.func, ast.Name) and node.func.id == "eval"
                ) or name == "builtins.eval":
                    builtin_eval.append(location)
                if (
                    isinstance(node.func, ast.Name) and node.func.id == "exec"
                ) or name == "builtins.exec":
                    builtin_exec.append(location)
                if name in {"pickle.load", "pickle.loads"}:
                    unsafe_pickle.append(location)
                if any(
                    keyword.arg == "shell"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in node.keywords
                ):
                    shell_true.append(location)
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and re.match(r"^(?:/home/|/mnt/|[A-Za-z]:\\\\)", node.value)
            ):
                absolute_paths.append(
                    {
                        "file": path.relative_to(root).as_posix(),
                        "line": getattr(node, "lineno", 0),
                        "value": node.value[:160],
                    }
                )

    source_lock = json.loads(
        (root / "src/zynnova/SOURCE_LOCK.json").read_text(encoding="utf-8")
    )
    source_ids = [str(item["id"]) for item in source_lock["sources"]]
    repositories = [str(item["repository"]) for item in source_lock["sources"]]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"].get("optional-dependencies", {})

    registered_extras: list[str] = []
    extension_dirs = (
        root / "src/zynnova/core",
        root / "src/zynnova/geometry",
        root / "src/zynnova/zynform",
        root / "src/zynnova/zynmorph",
        root / "src/zynnova/zynvista",
        root / "src/zynnova/zynvox",
    )
    for directory in extension_dirs:
        for path in directory.rglob("*.py"):
            for match in re.finditer(r"extras=\(([^)]*)\)", path.read_text(), re.S):
                registered_extras.extend(
                    re.findall(r"[\"'](zynnova-[^\"']+)[\"']", match.group(1))
                )
    unknown_extras = sorted(set(registered_extras) - set(extras))

    report: dict[str, object] = {
        "schema": "zynnova.static-audit/1.0",
        "observed_at": "2026-08-17",
        "python_files": len(files),
        "python_lines": line_count,
        "syntax_errors": syntax_errors,
        "shell_true": _locations(shell_true, root),
        "os_system": _locations(os_system, root),
        "subprocess_popen": _locations(popen, root),
        "builtin_eval_calls": _locations(builtin_eval, root),
        "builtin_exec_calls": _locations(builtin_exec, root),
        "unsafe_pickle_load_calls": _locations(unsafe_pickle, root),
        "hardcoded_user_absolute_paths": absolute_paths,
        "source_lock": {
            "source_count": len(source_ids),
            "unique_ids": len(source_ids) == len(set(source_ids)),
            "all_https": all(item.startswith("https://") for item in repositories),
            "ids": source_ids,
        },
        "pyproject": {
            "parsed": True,
            "zynnova_optional_groups": sorted(
                key for key in extras if key.startswith("zynnova")
            ),
            "unknown_registered_extras": unknown_extras,
        },
    }
    findings = (
        syntax_errors,
        shell_true,
        os_system,
        popen,
        builtin_eval,
        builtin_exec,
        unsafe_pickle,
        absolute_paths,
        unknown_extras,
    )
    lock_result = report["source_lock"]
    assert isinstance(lock_result, dict)
    report["passed"] = bool(
        not any(findings)
        and lock_result["unique_ids"]
        and lock_result["all_https"]
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/zynnova_static_audit.json"),
    )
    args = parser.parse_args(argv)
    root = _root()
    report = audit(root)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
