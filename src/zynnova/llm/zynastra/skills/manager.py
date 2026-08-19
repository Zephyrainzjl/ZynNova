"""Portable skill folders: SKILL.md instructions plus optional JSON metadata."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    directory: Path
    instructions: str
    metadata: dict[str, object]


class SkillManager:
    def __init__(self, roots: Iterable[str | Path] = ()) -> None:
        self.roots = tuple(Path(p).expanduser().resolve() for p in roots)
        self._skills: dict[str, Skill] = {}

    def discover(self) -> tuple[Skill, ...]:
        found: dict[str, Skill] = {}
        for root in self.roots:
            if not root.exists(): continue
            candidates = [root] if (root / "SKILL.md").is_file() else [p for p in root.iterdir() if p.is_dir()]
            for directory in candidates:
                skill_file = directory / "SKILL.md"
                if not skill_file.is_file(): continue
                metadata: dict[str, object] = {}
                manifest = directory / "manifest.json"
                if manifest.is_file():
                    metadata = json.loads(manifest.read_text(encoding="utf-8"))
                name = str(metadata.get("name") or directory.name)
                found[name] = Skill(name, directory, skill_file.read_text(encoding="utf-8"), metadata)
        self._skills = found
        return tuple(found.values())

    def get(self, name: str) -> Skill:
        if not self._skills: self.discover()
        return self._skills[name]

    def prompt_fragment(self) -> str:
        if not self._skills: self.discover()
        if not self._skills: return ""
        chunks = ["\n# Installed ZynAstra skills"]
        for skill in self._skills.values():
            chunks.append(f"\n## {skill.name}\n{skill.instructions}")
        return "\n".join(chunks)


__all__ = ["Skill", "SkillManager"]
