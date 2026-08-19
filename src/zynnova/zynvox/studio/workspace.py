"""External workspace for datasets, engines, checkpoints and generated audio."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VoiceWorkspace:
    root: Path
    def __init__(self, root: str | Path | None = None) -> None:
        base=root or os.environ.get("ZYNNOVA_VOICE_WORKSPACE") or os.environ.get("ZYNNOVA_WORKSPACE","~/.zynnova")
        root_path=Path(base).expanduser().resolve()
        if root_path.name != "zynvox": root_path=root_path/"zynvox"
        object.__setattr__(self,"root",root_path)
    @property
    def voices(self)->Path: return self.root/"voices"
    @property
    def datasets(self)->Path: return self.root/"datasets"
    @property
    def models(self)->Path: return self.root/"models"
    @property
    def engines(self)->Path: return self.root/"engines"
    @property
    def runs(self)->Path: return self.root/"runs"
    @property
    def cache(self)->Path: return self.root/"cache"
    def ensure(self)->"VoiceWorkspace":
        for p in (self.root,self.voices,self.datasets,self.models,self.engines,self.runs,self.cache): p.mkdir(parents=True,exist_ok=True)
        return self


__all__=["VoiceWorkspace"]
