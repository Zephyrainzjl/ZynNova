"""Synchronous, reproducible training orchestration for external acoustic engines."""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

from .engine import VoiceEngineProfile
from .workspace import VoiceWorkspace


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    dataset_manifest: Path
    run_name: str
    stages: tuple[str,...] = ("prepare-text","ssl-features","semantic","acoustic")
    stage_commands: Mapping[str,tuple[str,...]] = field(default_factory=dict)
    batch_size: int = 4
    epochs_semantic: int = 15
    epochs_acoustic: int = 8
    precision: str = "bf16"
    device: str = "cuda"
    extra: Mapping[str,object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_directory: Path
    model_directory: Path
    elapsed_s: float
    stage_logs: Mapping[str,Path]


def train_voice_model(config: TrainingConfig, engine: VoiceEngineProfile, workspace: VoiceWorkspace) -> TrainingResult:
    manifest=Path(config.dataset_manifest).expanduser().resolve()
    if not manifest.is_file(): raise FileNotFoundError(manifest)
    workspace.ensure(); run=workspace.runs/config.run_name; run.mkdir(parents=True,exist_ok=False)
    model_dir=workspace.models/config.run_name; model_dir.mkdir(parents=True,exist_ok=True)
    job={**asdict(config),"dataset_manifest":str(manifest),"model_directory":str(model_dir),"engine_root":str(engine.root)}
    job_path=run/"training.json"; job_path.write_text(json.dumps(job,default=str,indent=2),encoding="utf-8")
    env=os.environ.copy(); env.update(engine.env); env["ZYNNOVA_TRAINING_JOB"]=str(job_path)
    logs={}; started=time.perf_counter()
    for stage in config.stages:
        command=config.stage_commands.get(stage)
        if not command:
            # Standard driver contract: external repo may expose one module for all stages.
            command=(engine.python,"-m","zynnova_voice_driver.train","--stage",stage,"--job",str(job_path))
        log=run/f"{stage}.log"; logs[stage]=log
        proc=subprocess.run(list(command),cwd=engine.root,env=env,text=True,capture_output=True)
        log.write_text(proc.stdout+"\n--- STDERR ---\n"+proc.stderr,encoding="utf-8")
        if proc.returncode: raise RuntimeError(f"training stage {stage!r} failed; see {log}")
    return TrainingResult(run,model_dir,time.perf_counter()-started,logs)


__all__=["TrainingConfig","TrainingResult","train_voice_model"]
