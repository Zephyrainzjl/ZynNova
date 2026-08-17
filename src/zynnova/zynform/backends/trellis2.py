"""Official TRELLIS.2 Python-API adapter executed in an isolated environment."""

from __future__ import annotations

import sys
from pathlib import Path

from ...core import Availability, run_process
from ...core.backend import executable_availability, module_availability
from ..schema import ObjectConfig, ObjectRequest
from ..types import ObjectBackendOutput
from .base import ObjectBackend


class Trellis2Backend(ObjectBackend):
    name = "trellis2"

    def __init__(
        self,
        *,
        python_executable: str = sys.executable,
        repository: str | Path | None = None,
        model_id: str = "microsoft/TRELLIS.2-4B",
        decimation_target: int = 1_000_000,
        texture_size: int = 4096,
        seed: int = 0,
        attention_backend: str | None = None,
        timeout_s: float | None = None,
        **_: object,
    ) -> None:
        self.python_executable = str(python_executable)
        self.repository = None if repository is None else Path(repository).expanduser().resolve()
        self.model_id = str(model_id)
        self.decimation_target = int(decimation_target)
        self.texture_size = int(texture_size)
        self.seed = int(seed)
        self.attention_backend = attention_backend
        self.timeout_s = timeout_s

    def availability(self) -> Availability:
        status = executable_availability(self.python_executable)
        if not status.available and not Path(self.python_executable).is_file():
            return status
        if self.repository is not None:
            if not (self.repository / "trellis2").is_dir():
                return Availability(False, f"invalid TRELLIS.2 checkout: {self.repository}")
        else:
            module_status = module_availability("trellis2")
            if not module_status.available:
                return Availability(
                    False,
                    "TRELLIS.2 is neither importable nor supplied as backend_options.repository",
                )
        return Availability(
            True,
            details={"model_id": self.model_id, "repository": None if self.repository is None else str(self.repository)},
        )

    def run(
        self,
        request: ObjectRequest,
        config: ObjectConfig,
        work_directory: Path,
    ) -> ObjectBackendOutput:
        work_directory.mkdir(parents=True, exist_ok=True)
        helper = work_directory / "run_trellis2.py"
        helper.write_text(_helper_source(), encoding="utf-8")
        output = work_directory / "trellis2.glb"
        argv = [
            self.python_executable,
            str(helper),
            "--image",
            str(request.image.resolve()),
            "--output",
            str(output),
            "--model",
            request.model_id or self.model_id,
            "--decimation-target",
            str(self.decimation_target),
            "--texture-size",
            str(self.texture_size),
            "--seed",
            str(self.seed),
        ]
        env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        if self.attention_backend:
            env["ATTN_BACKEND"] = self.attention_backend
        result = run_process(
            argv,
            cwd=self.repository,
            env=env,
            timeout_s=self.timeout_s,
        )
        if not output.is_file():
            raise RuntimeError(f"TRELLIS.2 did not create {output}")
        return ObjectBackendOutput(
            backend=self.name,
            native_mesh=output,
            metadata={
                "elapsed_s": result.elapsed_s,
                "stdout_tail": result.stdout.splitlines()[-20:],
                "model_id": request.model_id or self.model_id,
                "pbr": True,
            },
        )


def _helper_source() -> str:
    return """from __future__ import annotations
import argparse
import os
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

p=argparse.ArgumentParser()
p.add_argument('--image', required=True)
p.add_argument('--output', required=True)
p.add_argument('--model', required=True)
p.add_argument('--decimation-target', type=int, default=1000000)
p.add_argument('--texture-size', type=int, default=4096)
p.add_argument('--seed', type=int, default=0)
a=p.parse_args()
torch.manual_seed(a.seed)
pipeline=Trellis2ImageTo3DPipeline.from_pretrained(a.model)
pipeline.cuda()
mesh=pipeline.run(Image.open(a.image).convert('RGBA'))[0]
mesh.simplify(16777216)
glb=o_voxel.postprocess.to_glb(
    vertices=mesh.vertices,
    faces=mesh.faces,
    attr_volume=mesh.attrs,
    coords=mesh.coords,
    attr_layout=mesh.layout,
    voxel_size=mesh.voxel_size,
    aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
    decimation_target=a.decimation_target,
    texture_size=a.texture_size,
    remesh=True,
    remesh_band=1,
    remesh_project=0,
    verbose=True,
)
glb.export(a.output, extension_webp=True)
"""


__all__ = ["Trellis2Backend"]
