"""Production object studio: modern generation -> repair/scale -> optional FEM."""
from __future__ import annotations
import json, shutil, uuid
from pathlib import Path
from typing import Protocol

from ..geometry import export_triangle_mesh, load_triangle_mesh
from .external import GenerativeObjectRequest,ObjectAssetBundle
from .meshing import tetrahedralize_surface
from .model_hub import object_workspace
from .repair import repair_surface_for_fem
from .scaling import apply_physical_scale,compute_physical_scale_transform,transform_native_asset
from .schema import FEMConfig

class GenerativeObjectEngine(Protocol):
    def run(self,request:GenerativeObjectRequest,output_dir:str|Path)->ObjectAssetBundle: ...

class ObjectStudio:
    def __init__(self,workspace:str|Path|None=None)->None:
        self.workspace=object_workspace(workspace); (self.workspace/"runs").mkdir(parents=True,exist_ok=True); self._engines:dict[str,GenerativeObjectEngine]={}
    def register_engine(self,name:str,engine:GenerativeObjectEngine,*,replace:bool=False)->None:
        if name in self._engines and not replace: raise KeyError(name)
        self._engines[name]=engine
    def engines(self)->tuple[str,...]: return tuple(sorted(self._engines))
    def generate(self,request:GenerativeObjectRequest,*,engine:str,repair:bool=True,generate_fem:bool=False,fem_config:FEMConfig|None=None)->ObjectAssetBundle:
        if engine not in self._engines: raise KeyError(f"unknown object engine {engine!r}; registered={sorted(self._engines)}")
        run=self.workspace/"runs"/f"object-{uuid.uuid4().hex}"; bundle=self._engines[engine].run(request,run/"engine"); export=run/"exports"; export.mkdir(parents=True)
        copied={}
        for role,source in bundle.assets.items():
            target=export/f"raw_{role}{source.suffix}"; shutil.copy2(source,target); copied[f"raw_{role}"]=target
        mesh=load_triangle_mesh(bundle.assets["mesh"])
        if repair: mesh=repair_surface_for_fem(mesh)
        transform=None
        if request.target_extent_m is not None:
            transform=compute_physical_scale_transform(mesh,request.target_extent_m); mesh=apply_physical_scale(mesh,transform)
        mesh_path=export/"object_repaired_scaled.ply"; export_triangle_mesh(mesh_path,mesh); copied["mesh"]=mesh_path
        native=bundle.assets.get("pbr") or bundle.assets.get("glb")
        if native is not None:
            native_target=export/("object_pbr"+native.suffix)
            if transform is not None:
                scaled=transform_native_asset(native,native_target,transform)
                if scaled is None: shutil.copy2(native,native_target)
            else: shutil.copy2(native,native_target)
            copied["pbr"]=native_target
        if generate_fem:
            volume=tetrahedralize_surface(mesh,fem_config)
            npz=export/"object_fem.npz"
            import numpy as np
            np.savez_compressed(npz,nodes=volume.nodes,tetrahedra=volume.tetrahedra,cell_regions=volume.cell_regions); copied["fem_npz"]=npz
        manifest={"format":1,"workflow":"zynnova.zynform.studio.generate","engine":engine,"model":request.model,"assets":{k:str(v) for k,v in copied.items()},"repair":repair,"generate_fem":generate_fem,"target_extent_m":request.target_extent_m,"elapsed_s":bundle.elapsed_s}
        (run/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        return ObjectAssetBundle(run,copied,bundle.engine,bundle.model,bundle.elapsed_s,bundle.metadata)

__all__=["ObjectStudio"]
