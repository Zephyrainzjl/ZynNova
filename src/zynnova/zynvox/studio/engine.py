"""Pluggable local engine contracts.

The package does not vendor third-party checkpoints or repositories.  A local engine
is installed in an external workspace and is invoked through this stable contract.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Protocol

from ..policy import enforce_consent_record
from ..schema import VoiceConfig, VoiceRequest
from ..tts_pipeline import run_speech_synthesis
from ..tts_schema import TTSConfig, TTSRequest
from .types import GenerationRequest, GenerationResult, VoiceProfile


class VoiceEngine(Protocol):
    name: str
    def synthesize(self, request: GenerationRequest, profile: VoiceProfile, output: Path) -> GenerationResult: ...
    def convert(self, source: Path, profile: VoiceProfile, output: Path, **options: object) -> GenerationResult: ...


@dataclass(frozen=True, slots=True)
class VoiceEngineProfile:
    """Command driver for GPT-SoVITS-class or future local engines."""
    name: str
    root: Path
    python: str = sys.executable
    infer_module: str | None = None
    vc_module: str | None = None
    infer_command: tuple[str, ...] = ()
    vc_command: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    output_contract: str = "json-job"  # engine reads job JSON and writes requested output

    def __post_init__(self)->None:
        object.__setattr__(self,"root",Path(self.root).expanduser().resolve())
        object.__setattr__(self,"env",dict(self.env))


class CommandVoiceEngine:
    """Invoke a local repository without exposing its private API to ZynNova callers."""
    def __init__(self, profile: VoiceEngineProfile) -> None:
        self.profile=profile; self.name=profile.name
        if not profile.root.is_dir(): raise FileNotFoundError(profile.root)

    def _base(self, kind: str) -> list[str]:
        module=self.profile.infer_module if kind=="tts" else self.profile.vc_module
        command=self.profile.infer_command if kind=="tts" else self.profile.vc_command
        if command: return [str(x) for x in command]
        if module: return [self.profile.python,"-m",module]
        raise RuntimeError(f"engine {self.name!r} has no {kind} command/module")

    def _run(self, kind: str, payload: dict[str,object], output: Path) -> GenerationResult:
        output.parent.mkdir(parents=True,exist_ok=True)
        job=output.with_suffix(output.suffix+".job.json")
        payload={**payload,"output":str(output.resolve()),"contract":"zynnova-zynvox-studio-v1"}
        job.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
        cmd=[*self._base(kind),"--zynnova-job",str(job)]
        import os
        env=os.environ.copy(); env.update(self.profile.env)
        started=time.perf_counter()
        proc=subprocess.run(cmd,cwd=self.profile.root,env=env,text=True,capture_output=True)
        elapsed=time.perf_counter()-started
        if proc.returncode:
            raise RuntimeError(f"voice engine failed ({proc.returncode}): {proc.stderr[-4000:]}")
        if not output.is_file():
            raise RuntimeError(f"voice engine reported success but did not create {output}")
        meta={"stdout":proc.stdout[-4000:],"job":str(job),"command":cmd}
        return GenerationResult(output,self.name,str(payload.get("model") or "") or None,elapsed,metadata=meta)

    def synthesize(self, request: GenerationRequest, profile: VoiceProfile, output: Path) -> GenerationResult:
        enforce_consent_record(profile.consent)
        payload={"task":"tts","request":asdict(request),"voice":{"id":profile.voice_id,"reference_audio":str(profile.reference_audio),"reference_text":profile.reference_text,"language":profile.language},"model":request.model or profile.model}
        return self._run("tts",payload,output)

    def convert(self, source: Path, profile: VoiceProfile, output: Path, **options: object) -> GenerationResult:
        enforce_consent_record(profile.consent)
        source=Path(source).resolve()
        if not source.is_file(): raise FileNotFoundError(source)
        return self._run("vc",{"task":"voice-conversion","source":str(source),"voice":{"id":profile.voice_id,"reference_audio":str(profile.reference_audio)},"options":options,"model":profile.model},output)


class LegacyZynVoxEngine:
    """Adapter that exposes existing ZynVox backends through the new Studio API."""
    name="zynnova-legacy"
    def __init__(self, *, tts_backend: str="auto", vc_backend: str="auto", backend_options: Mapping[str,object]|None=None) -> None:
        self.tts_backend=tts_backend; self.vc_backend=vc_backend; self.backend_options=dict(backend_options or {})
    def synthesize(self, request: GenerationRequest, profile: VoiceProfile, output: Path) -> GenerationResult:
        started=time.perf_counter()
        req=TTSRequest(text=request.text,target_reference=profile.reference_audio,consent=profile.consent,backend=self.tts_backend,output_name=output.stem,language=request.language.upper(),reference_transcript=profile.reference_text,streaming=request.streaming)
        opts={**self.backend_options,"seed":request.seed,"top_k":request.top_k,"top_p":request.top_p,"temperature":request.temperature,"speed":request.speed,"repetition_penalty":request.repetition_penalty,"batch_size":request.batch_size,"parallel_infer":request.parallel_infer,**request.extra}
        result=run_speech_synthesis(req,TTSConfig(output_directory=str(output.parent.parent),benchmark=False,backend_options=opts))
        output.parent.mkdir(parents=True,exist_ok=True)
        if result.output_audio.resolve()!=output.resolve():
            import shutil; shutil.copy2(result.output_audio,output)
        return GenerationResult(output,self.name,profile.model,time.perf_counter()-started,metadata={"manifest":str(result.manifest_path)})
    def convert(self, source: Path, profile: VoiceProfile, output: Path, **options: object) -> GenerationResult:
        from ..pipeline import run_voice_conversion
        started=time.perf_counter()
        req=VoiceRequest(source_audio=source,target_reference=profile.reference_audio,consent=profile.consent,backend=self.vc_backend,output_name=output.stem)
        result=run_voice_conversion(req,VoiceConfig(output_directory=str(output.parent.parent),benchmark=False,backend_options={**self.backend_options,**options}))
        output.parent.mkdir(parents=True,exist_ok=True)
        if result.output_audio.resolve()!=output.resolve():
            import shutil; shutil.copy2(result.output_audio,output)
        return GenerationResult(output,self.name,profile.model,time.perf_counter()-started,metadata={"manifest":str(result.manifest_path)})


__all__=["CommandVoiceEngine","LegacyZynVoxEngine","VoiceEngine","VoiceEngineProfile"]
