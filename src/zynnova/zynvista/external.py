"""Stable contract for fast-moving external world/reconstruction engines."""
from __future__ import annotations
import importlib, json, os, subprocess, sys, time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True,slots=True)
class SceneEngineProfile:
    name: str
    root: Path
    python: str = sys.executable
    module: str | None = None
    command: tuple[str,...] = ()
    env: Mapping[str,str] = field(default_factory=dict)
    capabilities: tuple[str,...] = ("image","multiview","video","text","mesh","3dgs")
    def __post_init__(self):
        object.__setattr__(self,"root",Path(self.root).expanduser().resolve()); object.__setattr__(self,"env",dict(self.env))


@dataclass(frozen=True,slots=True)
class GenerativeSceneRequest:
    prompt: str | None = None
    images: tuple[Path,...] = ()
    video: Path | None = None
    model: str | None = None
    seed: int = 0
    geometry_mode: str = "mesh+3dgs"
    texture_mode: str = "pbr"
    target_extent_m: float | None = None
    options: Mapping[str,object] = field(default_factory=dict)
    def __post_init__(self):
        imgs=tuple(Path(p).expanduser().resolve() for p in self.images)
        if not self.prompt and not imgs and self.video is None: raise ValueError("scene generation needs text, images, or video")
        for p in imgs:
            if not p.is_file(): raise FileNotFoundError(p)
        video=None if self.video is None else Path(self.video).expanduser().resolve()
        if video is not None and not video.is_file(): raise FileNotFoundError(video)
        if self.target_extent_m is not None and self.target_extent_m<=0: raise ValueError("target_extent_m must be positive")
        object.__setattr__(self,"images",imgs); object.__setattr__(self,"video",video); object.__setattr__(self,"options",dict(self.options))


@dataclass(frozen=True,slots=True)
class SceneAssetBundle:
    directory: Path
    assets: Mapping[str,Path]
    engine: str
    model: str | None
    elapsed_s: float
    metadata: Mapping[str,object] = field(default_factory=dict)
    def require(self,*roles:str)->None:
        missing=[role for role in roles if role not in self.assets or not Path(self.assets[role]).is_file()]
        if missing: raise RuntimeError(f"scene engine is missing required assets: {missing}")


class CommandSceneEngine:
    """Engine reads one JSON job and writes ``result.json`` plus assets into output_dir."""
    def __init__(self,profile:SceneEngineProfile)->None:
        self.profile=profile
        if not profile.root.is_dir(): raise FileNotFoundError(profile.root)
    def run(self,request:GenerativeSceneRequest,output_dir:str|Path)->SceneAssetBundle:
        out=Path(output_dir).expanduser().resolve(); out.mkdir(parents=True,exist_ok=False)
        job=out/"job.json"; result_path=out/"result.json"
        payload={"contract":"zynnova-zynvista-v1","request":{**asdict(request),"images":[str(p) for p in request.images],"video":str(request.video) if request.video else None},"output_dir":str(out),"result_json":str(result_path)}
        job.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
        if self.profile.command: cmd=[*self.profile.command,"--zynnova-job",str(job)]
        elif self.profile.module: cmd=[self.profile.python,"-m",self.profile.module,"--zynnova-job",str(job)]
        else: raise RuntimeError(f"scene engine {self.profile.name!r} has no command/module")
        env=os.environ.copy(); env.update(self.profile.env); started=time.perf_counter()
        proc=subprocess.run(cmd,cwd=self.profile.root,env=env,text=True,capture_output=True); elapsed=time.perf_counter()-started
        (out/"engine.log").write_text(proc.stdout+"\n--- STDERR ---\n"+proc.stderr,encoding="utf-8")
        if proc.returncode: raise RuntimeError(f"scene engine failed ({proc.returncode}); see {out/'engine.log'}")
        if not result_path.is_file(): raise RuntimeError(f"scene engine did not create {result_path}")
        obj=json.loads(result_path.read_text(encoding="utf-8")); assets={str(k):(out/Path(v) if not Path(v).is_absolute() else Path(v)) for k,v in (obj.get("assets") or {}).items()}
        bad={k:str(v) for k,v in assets.items() if not v.is_file()}
        if bad: raise RuntimeError(f"scene result references missing assets: {bad}")
        return SceneAssetBundle(out,assets,self.profile.name,request.model,elapsed,obj.get("metadata") or {})


class PythonSceneEngine:
    """Call ``module:function(request_dict, output_dir)`` from an external checkout."""
    def __init__(self,name:str,callable_path:str,root:str|Path|None=None)->None:
        self.name=name; self.callable_path=callable_path; self.root=None if root is None else Path(root).resolve()
    def run(self,request:GenerativeSceneRequest,output_dir:str|Path)->SceneAssetBundle:
        import sys
        if self.root is not None and str(self.root) not in sys.path: sys.path.insert(0,str(self.root))
        module,name=self.callable_path.split(":",1); fn=getattr(importlib.import_module(module),name); out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=False)
        started=time.perf_counter(); result=fn({**asdict(request),"images":[str(p) for p in request.images],"video":str(request.video) if request.video else None},str(out)); elapsed=time.perf_counter()-started
        if not isinstance(result,Mapping): raise TypeError("external scene callable must return a mapping of asset roles to paths")
        assets={str(k):Path(v).resolve() for k,v in result.items()}; return SceneAssetBundle(out,assets,self.name,request.model,elapsed)

__all__=["CommandSceneEngine","GenerativeSceneRequest","PythonSceneEngine","SceneAssetBundle","SceneEngineProfile"]
