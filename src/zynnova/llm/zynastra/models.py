"""External model download and registry helpers."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .workspace import Workspace


@dataclass(frozen=True, slots=True)
class LocalModel:
    model_id: str
    path: Path
    revision: str | None = None


def download_model(
    model_id: str,
    workspace: Workspace,
    *,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    token: str | bool | None = None,
) -> LocalModel:
    """Download a Hugging Face snapshot into ``workspace/models``, never package data."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("install zynnova[llm-local] for model downloads") from exc
    workspace.ensure()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "--", model_id)
    target = workspace.models / safe
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=target,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        token=token,
    )
    meta = {"model_id": model_id, "revision": revision, "path": str(target)}
    (target / "zynnova_model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return LocalModel(model_id, target, revision)


__all__ = ["LocalModel", "download_model"]
