"""External image/multiview/text-to-object generation contract."""
from __future__ import annotations
import json, os, subprocess, sys, time
from dataclasses import asdict,dataclass,field
from pathlib import Path
from typing import Mapping

@dataclass(frozen=True,slots=True)
class ObjectEngineProfile:
    name:str; root:Path; python:str=sys.executable; module:str|None=None; command:tuple[str,...]=(); env:Mapping[str,str]=field(default_factory=dict); capabilities:tuple[str,...]=("image","multiview","text","mesh","pbr")
    def __post_init__(self): object.__setattr__(self,"root",Path(self.root).expanduser().resolve()); object.__setattr__(self,"env",dict(self.env))

@dataclass(frozen=True,slots=True)
class GenerativeObjectRequest:
    images:tuple[Path,...]=(); prompt:str|None=None; model:str|None=None; seed:int=0; texture_mode:str="pbr"; topology_mode:str="production"; target_extent_m:float|None=None; options:Mapping[str,object]=field(default_factory=dict)
    def __post_init__(self):
        images=tuple(Path(p).expanduser().resolve() for p in self.images)
        if not images and not self.prompt: raise ValueError("object generation needs image(s) or prompt")
        for p in images:
            if not p.is_file(): raise FileNotFoundError(p)
        if self.target_extent_m is not None and self.target_extent_m<=0: raise ValueError("target_extent_m must be positive")
        object.__setattr__(self,"images",images); object.__setattr__(self,"options",dict(self.options))

@dataclass(frozen=True,slots=True)
class ObjectAssetBundle:
    directory:Path; assets:Mapping[str,Path]; engine:str; model:str|None; elapsed_s:float; metadata:Mapping[str,object]=field(default_factory=dict)

class CommandObjectEngine:
    def __init__(self,profile:ObjectEngineProfile)->None:
        self.profile=profile
        if not profile.root.is_dir(): raise FileNotFoundError(profile.root)
    def run(self,request:GenerativeObjectRequest,output_dir:str|Path)->ObjectAssetBundle:
        out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=False); job=out/"job.json"; result=out/"result.json"
        payload={"contract":"zynnova-zynform-v1","request":{**asdict(request),"images":[str(p) for p in request.images]},"output_dir":str(out),"result_json":str(result)}; job.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
        if self.profile.command: cmd=[*self.profile.command,"--zynnova-job",str(job)]
        elif self.profile.module: cmd=[self.profile.python,"-m",self.profile.module,"--zynnova-job",str(job)]
        else: raise RuntimeError(f"object engine {self.profile.name!r} has no command/module")
        env=os.environ.copy(); env.update(self.profile.env); started=time.perf_counter(); proc=subprocess.run(cmd,cwd=self.profile.root,env=env,text=True,capture_output=True); elapsed=time.perf_counter()-started
        (out/"engine.log").write_text(proc.stdout+"\n--- STDERR ---\n"+proc.stderr,encoding="utf-8")
        if proc.returncode: raise RuntimeError(f"object engine failed ({proc.returncode}); see {out/'engine.log'}")
        if not result.is_file(): raise RuntimeError(f"object engine did not create {result}")
        obj=json.loads(result.read_text(encoding="utf-8")); assets={str(k):(out/Path(v) if not Path(v).is_absolute() else Path(v)) for k,v in (obj.get("assets") or {}).items()}
        if "mesh" not in assets: raise RuntimeError("object engine contract requires a 'mesh' asset")
        bad={k:str(v) for k,v in assets.items() if not v.is_file()}
        if bad: raise RuntimeError(f"object result references missing assets: {bad}")
        return ObjectAssetBundle(out,assets,self.profile.name,request.model,elapsed,obj.get("metadata") or {})

__all__=["CommandObjectEngine","GenerativeObjectRequest","ObjectAssetBundle","ObjectEngineProfile"]
