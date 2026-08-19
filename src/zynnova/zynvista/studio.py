"""High-level world/scene studio combining existing metric reconstruction with modern external generators."""
from __future__ import annotations
import json, shutil, uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .external import GenerativeSceneRequest, SceneAssetBundle
from .model_hub import scene_workspace
from .pipeline import SceneResult, run_scene
from .schema import SceneConfig, SceneRequest


class GenerativeSceneEngine(Protocol):
    def run(self,request:GenerativeSceneRequest,output_dir:str|Path)->SceneAssetBundle: ...


class SceneStudio:
    def __init__(self,workspace:str|Path|None=None)->None:
        self.workspace=scene_workspace(workspace); (self.workspace/"runs").mkdir(parents=True,exist_ok=True); self._engines:dict[str,GenerativeSceneEngine]={}
    def register_engine(self,name:str,engine:GenerativeSceneEngine,*,replace:bool=False)->None:
        if name in self._engines and not replace: raise KeyError(name)
        self._engines[name]=engine
    def engines(self)->tuple[str,...]: return tuple(sorted(self._engines))
    def reconstruct(self,request:SceneRequest,config:SceneConfig|None=None)->SceneResult:
        return run_scene(request,config)
    def generate(self,request:GenerativeSceneRequest,*,engine:str)->SceneAssetBundle:
        if engine not in self._engines: raise KeyError(f"unknown scene studio engine {engine!r}; registered={sorted(self._engines)}")
        run=self.workspace/"runs"/f"world-{uuid.uuid4().hex}"; bundle=self._engines[engine].run(request,run/"engine")
        export=run/"exports"; export.mkdir(parents=True)
        copied={}
        for role,source in bundle.assets.items():
            target=export/f"{role}{source.suffix}"; shutil.copy2(source,target); copied[role]=target
        manifest={"format":1,"workflow":"zynnova.zynvista.studio.generate","engine":engine,"model":request.model,"assets":{k:str(v) for k,v in copied.items()},"elapsed_s":bundle.elapsed_s,"input":{"prompt":request.prompt,"images":[str(p) for p in request.images],"video":str(request.video) if request.video else None},"geometry_mode":request.geometry_mode,"texture_mode":request.texture_mode,"target_extent_m":request.target_extent_m}
        (run/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
        return SceneAssetBundle(run,copied,bundle.engine,bundle.model,bundle.elapsed_s,bundle.metadata)


__all__=["SceneStudio"]
