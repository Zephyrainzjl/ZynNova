"""Dataset slicing, normalization and optional Whisper transcription."""
from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .workspace import VoiceWorkspace


@dataclass(frozen=True, slots=True)
class DatasetPrepareConfig:
    dataset_name: str
    input_audio: tuple[Path,...]
    language: str = "auto"
    sample_rate: int = 32000
    min_segment_s: float = 1.0
    max_segment_s: float = 15.0
    silence_db: float = -42.0
    pad_s: float = 0.15
    transcribe: bool = True
    whisper_model: str = "large-v3"
    whisper_device: str = "auto"


def _read(path: Path, sample_rate: int):
    try: import soundfile as sf
    except ImportError as exc: raise RuntimeError("install zynnova[voice-studio]") from exc
    audio,sr=sf.read(path,dtype="float32",always_2d=False)
    if audio.ndim>1: audio=audio.mean(axis=1)
    if sr!=sample_rate:
        try: from scipy.signal import resample_poly
        except ImportError as exc: raise RuntimeError("scipy is required for resampling") from exc
        import math as _m; g=_m.gcd(sr,sample_rate); audio=resample_poly(audio,sample_rate//g,sr//g).astype(np.float32); sr=sample_rate
    peak=float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak>0: audio=np.clip(audio/peak*0.95,-1,1)
    return audio,sr


def _segments(audio: np.ndarray, sr: int, cfg: DatasetPrepareConfig):
    frame=max(1,int(0.03*sr)); hop=max(1,int(0.01*sr))
    if len(audio)<=frame: return [(0,len(audio))] if len(audio)>=int(cfg.min_segment_s*sr) else []
    starts=np.arange(0,max(1,len(audio)-frame+1),hop)
    rms=np.array([np.sqrt(np.mean(audio[s:s+frame]**2)+1e-12) for s in starts])
    db=20*np.log10(rms+1e-9); voiced=db>=cfg.silence_db
    intervals=[]; active=None
    for i,v in enumerate(voiced):
        pos=int(starts[i])
        if v and active is None: active=pos
        if active is not None and (not v or i==len(voiced)-1):
            end=min(len(audio),pos+frame); pad=int(cfg.pad_s*sr); a=max(0,active-pad); b=min(len(audio),end+pad)
            if b-a>=int(cfg.min_segment_s*sr): intervals.append((a,b))
            active=None
    out=[]; maxn=int(cfg.max_segment_s*sr)
    for a,b in intervals:
        while b-a>maxn:
            out.append((a,a+maxn)); a+=maxn
        if b-a>=int(cfg.min_segment_s*sr): out.append((a,b))
    return out


def prepare_dataset(config: DatasetPrepareConfig, workspace: VoiceWorkspace) -> Path:
    """Create WAV clips + ``manifest.csv`` suitable for GPT/SoVITS-class training drivers."""
    try: import soundfile as sf
    except ImportError as exc: raise RuntimeError("install zynnova[voice-studio]") from exc
    workspace.ensure(); target=workspace.datasets/config.dataset_name
    if target.exists(): shutil.rmtree(target)
    clips=target/"wavs"; clips.mkdir(parents=True)
    rows=[]
    for source in config.input_audio:
        source=Path(source).expanduser().resolve()
        if not source.is_file(): raise FileNotFoundError(source)
        audio,sr=_read(source,config.sample_rate)
        for idx,(a,b) in enumerate(_segments(audio,sr,config)):
            path=clips/f"{source.stem}-{idx:05d}.wav"; sf.write(path,audio[a:b],sr,subtype="PCM_16")
            rows.append([str(path),config.language,""])
    if config.transcribe and rows:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc: raise RuntimeError("install zynnova[voice-studio-asr] for transcription") from exc
        device="cuda" if config.whisper_device=="auto" else config.whisper_device
        try: model=WhisperModel(config.whisper_model,device=device,compute_type="float16" if device=="cuda" else "int8")
        except Exception:
            model=WhisperModel(config.whisper_model,device="cpu",compute_type="int8")
        for row in rows:
            segs,info=model.transcribe(row[0],language=None if config.language.lower()=="auto" else config.language)
            row[2]="".join(seg.text for seg in segs).strip(); row[1]=getattr(info,"language",row[1]) or row[1]
    manifest=target/"manifest.csv"
    with manifest.open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f); writer.writerow(["audio","language","text"]); writer.writerows(rows)
    return manifest


__all__=["DatasetPrepareConfig","prepare_dataset"]
