"""External model storage for object generators."""
from __future__ import annotations
import os,re
from pathlib import Path

def object_workspace(root:str|Path|None=None)->Path:
    base=Path(root or os.environ.get("ZYNNOVA_OBJECT_WORKSPACE") or os.environ.get("ZYNNOVA_WORKSPACE","~/.zynnova")).expanduser().resolve(); target=base if base.name=="zynform" else base/"zynform"; target.mkdir(parents=True,exist_ok=True); return target

def download_object_model(model_id:str,workspace:str|Path|None=None,*,revision:str|None=None,token:str|bool|None=None)->Path:
    try: from huggingface_hub import snapshot_download
    except ImportError as exc: raise RuntimeError("install zynnova[object-models]") from exc
    root=object_workspace(workspace)/"models"; root.mkdir(parents=True,exist_ok=True); target=root/re.sub(r"[^A-Za-z0-9_.-]+","--",model_id)
    snapshot_download(repo_id=model_id,revision=revision,local_dir=target,token=token); return target

__all__=["download_object_model","object_workspace"]
