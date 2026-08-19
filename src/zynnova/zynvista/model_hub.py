"""External model/repository workspace for ZynVista."""
from __future__ import annotations
import os, re
from pathlib import Path


def scene_workspace(root: str|Path|None=None)->Path:
    base=Path(root or os.environ.get("ZYNNOVA_SCENE_WORKSPACE") or os.environ.get("ZYNNOVA_WORKSPACE","~/.zynnova")).expanduser().resolve()
    target=base if base.name=="zynvista" else base/"zynvista"; target.mkdir(parents=True,exist_ok=True); return target


def download_scene_model(model_id: str, workspace: str|Path|None=None, *, revision: str|None=None, token: str|bool|None=None)->Path:
    try: from huggingface_hub import snapshot_download
    except ImportError as exc: raise RuntimeError("install zynnova[scene-models]") from exc
    root=scene_workspace(workspace)/"models"; root.mkdir(parents=True,exist_ok=True)
    path=root/re.sub(r"[^A-Za-z0-9_.-]+","--",model_id)
    snapshot_download(repo_id=model_id,revision=revision,local_dir=path,token=token)
    return path

__all__=["download_scene_model","scene_workspace"]
