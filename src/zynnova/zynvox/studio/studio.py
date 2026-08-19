"""High-level first-party ZynVox Studio API."""
from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from ..policy import enforce_consent_record
from ..schema import ConsentBasis, ConsentRecord
from .engine import LegacyZynVoxEngine, VoiceEngine
from .types import GenerationRequest, GenerationResult, VoiceProfile
from .workspace import VoiceWorkspace


class ZynVoxStudio:
    def __init__(self, workspace: str | VoiceWorkspace | None = None, *, engine: VoiceEngine | None = None) -> None:
        self.workspace=workspace if isinstance(workspace,VoiceWorkspace) else VoiceWorkspace(workspace)
        self.workspace.ensure(); self.engine=engine or LegacyZynVoxEngine()

    def _voice_dir(self, voice_id: str)->Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+",voice_id): raise ValueError("invalid voice_id")
        return self.workspace.voices/voice_id

    def enroll_voice(self, voice_id: str, reference_audio: str|Path, consent: ConsentRecord, *, reference_text: str|None=None, language: str="auto", model: str|None=None, metadata: Mapping[str,object]|None=None, copy_audio: bool=True) -> VoiceProfile:
        policy=enforce_consent_record(consent); source=Path(reference_audio).expanduser().resolve()
        if not source.is_file(): raise FileNotFoundError(source)
        directory=self._voice_dir(voice_id); directory.mkdir(parents=True,exist_ok=True)
        target=directory/("reference"+source.suffix.lower()) if copy_audio else source
        if copy_audio: shutil.copy2(source,target)
        record={"voice_id":voice_id,"reference_audio":str(target),"reference_text":reference_text,"language":language,"model":model,"metadata":dict(metadata or {}),"consent":{"confirmed":consent.confirmed,"basis":consent.basis.value,"purpose":consent.purpose,"record_id":consent.record_id,"recorded_at":consent.recorded_at,"evidence":str(consent.evidence) if consent.evidence else None},"policy":asdict(policy)}
        (directory/"voice.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
        return self.get_voice(voice_id)

    def get_voice(self, voice_id: str)->VoiceProfile:
        path=self._voice_dir(voice_id)/"voice.json"
        if not path.is_file(): raise KeyError(f"unknown voice: {voice_id}")
        obj=json.loads(path.read_text(encoding="utf-8")); c=obj["consent"]
        consent=ConsentRecord(bool(c["confirmed"]),ConsentBasis(c["basis"]),c["purpose"],record_id=c["record_id"],recorded_at=c["recorded_at"],evidence=Path(c["evidence"]) if c.get("evidence") else None)
        return VoiceProfile(obj["voice_id"],Path(obj["reference_audio"]),consent,obj.get("reference_text"),obj.get("language","auto"),obj.get("model"),obj.get("metadata") or {})

    def list_voices(self)->tuple[str,...]:
        return tuple(sorted(p.name for p in self.workspace.voices.iterdir() if (p/"voice.json").is_file()))

    def synthesize(self, request: GenerationRequest)->GenerationResult:
        profile=self.get_voice(request.voice_id); run=self.workspace.runs/f"tts-{uuid.uuid4().hex}"
        output=run/"exports"/(Path(request.output_name).stem+".wav")
        return self.engine.synthesize(request,profile,output)

    def voice_convert(self, source_audio: str|Path, voice_id: str, *, output_name: str="converted", **options: object)->GenerationResult:
        profile=self.get_voice(voice_id); run=self.workspace.runs/f"vc-{uuid.uuid4().hex}"
        output=run/"exports"/(Path(output_name).stem+".wav")
        return self.engine.convert(Path(source_audio),profile,output,**options)


__all__=["ZynVoxStudio"]
