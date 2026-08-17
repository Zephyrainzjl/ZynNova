from __future__ import annotations

import json

from zynnova import __version__
from zynnova.cli import build_parser, main


def test_cli_status_writes_json(tmp_path, capsys) -> None:
    output = tmp_path / "status.json"
    assert main(["status", "--output", str(output)]) == 0
    captured = capsys.readouterr()
    assert "zynnova_version" in captured.out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["zynnova_version"] == __version__


def test_cli_exposes_all_isolated_subcommands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for command in ("morph", "scene", "object", "voice", "tts", "voice-ui"):
        assert command in help_text
