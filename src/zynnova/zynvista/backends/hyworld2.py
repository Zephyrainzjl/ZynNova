"""Official HY-World 2.0 reconstruction and five-stage generation adapters."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ...core import Availability, require_known_license, run_process
from ...core.backend import executable_availability
from ..schema import SceneConfig, SceneRequest
from ..types import SceneBackendOutput
from .base import SceneBackend


def _stage_input(request: SceneRequest, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    if request.video is not None:
        copied = target / request.video.name
        shutil.copy2(request.video, copied)
        return copied
    for index, source in enumerate(request.images):
        suffix = source.suffix.lower() or ".png"
        shutil.copy2(source, target / f"image_{index:04d}{suffix}")
    return target


def _collect_native(root: Path) -> dict[str, Path]:
    patterns = {
        "point_cloud": ("points.ply", "**/points.ply"),
        "gaussians": ("gaussians.ply", "**/gaussians.ply"),
        "camera_params": ("camera_params.json", "**/camera_params.json"),
        "timing": ("pipeline_timing.json", "**/pipeline_timing.json"),
        "spz": ("*.spz", "**/*.spz"),
        "mesh": (
            "*mesh*.glb",
            "*mesh*.obj",
            "*mesh*.ply",
            "**/*mesh*.glb",
            "**/*mesh*.obj",
            "**/*mesh*.ply",
        ),
    }
    assets: dict[str, Path] = {}
    for role, role_patterns in patterns.items():
        for pattern in role_patterns:
            candidates = sorted(path for path in root.glob(pattern) if path.is_file())
            if candidates:
                assets[role] = candidates[-1]
                break
    sparse = next((path for path in root.glob("**/sparse/0") if path.is_dir()), None)
    if sparse is not None:
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            item = sparse / name
            if item.is_file():
                assets[f"colmap_{name.split('.')[0]}"] = item
    return assets


class HYWorld2ReconstructionBackend(SceneBackend):
    """Execute WorldMirror 2.0 in its own Python environment."""

    name = "hy-world-2-reconstruct"

    def __init__(
        self,
        *,
        python_executable: str = sys.executable,
        repository: str | Path | None = None,
        gpu_count: int = 1,
        enable_bf16: bool = True,
        fsdp_cpu_offload: bool = False,
        accept_license: bool = False,
        timeout_s: float | None = None,
        **_: object,
    ) -> None:
        self.python_executable = str(python_executable)
        self.repository = None if repository is None else Path(repository).expanduser().resolve()
        self.gpu_count = int(gpu_count)
        self.enable_bf16 = bool(enable_bf16)
        self.fsdp_cpu_offload = bool(fsdp_cpu_offload)
        self.accept_license = bool(accept_license)
        self.timeout_s = timeout_s

    def availability(self) -> Availability:
        status = executable_availability(self.python_executable)
        if not status.available and not Path(self.python_executable).is_file():
            return status
        if self.repository is None:
            return Availability(
                False,
                "repository is required for HY-World 2.0; pass backend_options={'repository': ...}",
            )
        if not (self.repository / "hyworld2").is_dir():
            return Availability(False, f"HY-World repository is invalid: {self.repository}")
        if self.gpu_count < 1:
            return Availability(False, "gpu_count must be positive")
        try:
            require_known_license("hy-world-2", explicit=self.accept_license)
        except Exception as exc:
            return Availability(False, str(exc))
        return Availability(
            True,
            details={"gpu_count": self.gpu_count, "repository": str(self.repository)},
        )

    def run(
        self,
        request: SceneRequest,
        config: SceneConfig,
        work_directory: Path,
    ) -> SceneBackendOutput:
        require_known_license("hy-world-2", explicit=self.accept_license)
        if self.repository is None:  # guarded by registry, retained for direct construction
            raise ValueError("repository is required for HY-World 2.0 reconstruction")
        input_path = _stage_input(request, work_directory / "inputs")
        output = work_directory / "worldmirror"
        output.mkdir(parents=True, exist_ok=True)
        if self.gpu_count == 1:
            argv = [self.python_executable]
        else:
            argv = ["torchrun", f"--nproc_per_node={self.gpu_count}"]
        argv += [
            "-m",
            "hyworld2.worldrecon.pipeline",
            "--input_path",
            str(input_path),
            "--strict_output_path",
            str(output),
            "--no_interactive",
            "--save_colmap",
            "--apply_confidence_mask",
            "--confidence_percentile",
            str(config.confidence_percentile),
            "--compress_pts_max_points",
            str(config.maximum_points),
            "--compress_pts_voxel_size",
            str(config.fusion_voxel_size_m),
        ]
        if request.prior_camera is not None:
            argv += ["--prior_cam_path", str(request.prior_camera)]
        if request.prior_depth_directory is not None:
            argv += ["--prior_depth_path", str(request.prior_depth_directory)]
        if self.gpu_count > 1:
            argv.append("--use_fsdp")
            if self.enable_bf16:
                argv.append("--enable_bf16")
            if self.fsdp_cpu_offload:
                argv.append("--fsdp_cpu_offload")
        result = run_process(argv, cwd=self.repository, timeout_s=self.timeout_s)
        assets = _collect_native(output)
        if not assets:
            raise RuntimeError(
                f"WorldMirror completed but produced no recognized files in {output}"
            )
        return SceneBackendOutput(
            backend=self.name,
            native_assets=assets,
            metadata={
                "stdout_tail": result.stdout.splitlines()[-30:],
                "elapsed_s": result.elapsed_s,
                "gpu_count": self.gpu_count,
            },
        )


class HYWorld2GenerationBackend(SceneBackend):
    """Map ZynVista generation to HY-World 2.0's official five-stage workflow."""

    name = "hy-world-2-generate"

    def __init__(
        self,
        *,
        repository: str | Path | None = None,
        python_executable: str = sys.executable,
        gpu_count: int = 4,
        llm_addr: str = "127.0.0.1",
        llm_port: int = 8000,
        llm_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        panorama_backend: str = "hunyuanimage3",
        max_steps: int | None = None,
        accept_license: bool = False,
        timeout_s: float | None = None,
        **_: object,
    ) -> None:
        self.repository = None if repository is None else Path(repository).expanduser().resolve()
        self.python_executable = str(python_executable)
        self.gpu_count = int(gpu_count)
        self.llm_addr = str(llm_addr)
        self.llm_port = int(llm_port)
        self.llm_name = str(llm_name)
        self.panorama_backend = str(panorama_backend)
        self.max_steps = max_steps
        self.accept_license = bool(accept_license)
        self.timeout_s = timeout_s

    def availability(self) -> Availability:
        if self.repository is None:
            return Availability(
                False,
                "repository is required for HY-World 2.0; pass backend_options={'repository': ...}",
            )
        worldgen = self.repository / "hyworld2" / "worldgen"
        if not worldgen.is_dir():
            return Availability(False, f"worldgen directory not found: {worldgen}")
        required = ["traj_generate.py", "traj_render.py", "video_gen.py", "gen_gs_data.py"]
        missing = [name for name in required if not (worldgen / name).is_file()]
        if missing:
            return Availability(False, f"HY-World checkout is missing: {', '.join(missing)}")
        if self.gpu_count < 1:
            return Availability(False, "gpu_count must be positive")
        try:
            require_known_license("hy-world-2", explicit=self.accept_license)
        except Exception as exc:
            return Availability(False, str(exc))
        return Availability(
            True,
            details={
                "gpu_count": self.gpu_count,
                "recommended_gpu_count": 4,
                "llm_endpoint": f"{self.llm_addr}:{self.llm_port}",
            },
        )

    def run(
        self,
        request: SceneRequest,
        config: SceneConfig,
        work_directory: Path,
    ) -> SceneBackendOutput:
        require_known_license("hy-world-2", explicit=self.accept_license)
        if self.repository is None:  # guarded by registry, retained for direct construction
            raise ValueError("repository is required for HY-World 2.0 generation")
        worldgen = self.repository / "hyworld2" / "worldgen"
        target = work_directory / "hyworld_scene"
        result_directory = work_directory / "hyworld_result"
        target.mkdir(parents=True, exist_ok=True)
        result_directory.mkdir(parents=True, exist_ok=True)
        self._prepare_panorama(request, target, worldgen)
        common_llm = [
            "--llm_addr",
            self.llm_addr,
            "--llm_port",
            str(self.llm_port),
            "--llm_name",
            self.llm_name,
        ]
        commands: list[list[str]] = [
            [
                self.python_executable,
                "traj_generate.py",
                "--target_path",
                str(target),
                *common_llm,
                "--apply_nav_traj",
                "--apply_up_route",
                "--apply_recon_iteration",
                "--force_vlm",
            ],
            [
                "torchrun",
                "--nproc_per_node",
                str(self.gpu_count),
                "traj_render.py",
                "--target_path",
                str(target),
                *common_llm,
            ],
            [
                "torchrun",
                "--nproc_per_node",
                str(self.gpu_count),
                "video_gen.py",
                "--target_path",
                str(target),
                "--fsdp",
            ],
            [
                "torchrun",
                "--nproc_per_node",
                str(self.gpu_count),
                "gen_gs_data.py",
                "--root_path",
                str(target),
                "--save_normal",
                "--split_sky",
            ],
        ]
        max_steps = self.max_steps or max(1500, int(round(12_000 / self.gpu_count)))
        commands.append(
            [
                self.python_executable,
                "-m",
                "world_gs_trainer",
                "default",
                "--data_dir",
                str(target / "gs_data"),
                "--result_dir",
                str(result_directory),
                "--max_steps",
                str(max_steps),
                "--save_steps",
                str(max_steps),
                "--eval_steps",
                str(max_steps),
                "--ply_steps",
                str(max_steps),
                "--save_ply",
                "--convert_to_spz",
                "--disable_video",
                "--use_scale_regularization",
                "--antialiased",
                "--depth_loss",
                "--normal_loss",
                "--sky_depth_from_pcd",
                "--use_mask_gaussian",
                "--mask_export_stochastic",
                "--no-mask-export-anchor-protection",
                "--use_anchor_protection",
                "--export_mesh",
                "--strategy.refine-start-iter",
                str(max(1, max_steps // 10)),
                "--strategy.refine-stop-iter",
                str(max(2, max_steps // 2)),
                "--strategy.refine-every",
                "100",
                "--strategy.refine-scale2d-stop-iter",
                str(max(2, max_steps // 2)),
                "--strategy.reset-every",
                "99990",
                "--strategy.grow-grad2d",
                "0.0001",
                "--strategy.prune-scale3d",
                "0.1",
            ]
        )
        stage_reports = []
        env = {"CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(self.gpu_count))}
        for stage, command in enumerate(commands, start=1):
            result = run_process(
                command,
                cwd=worldgen,
                env=env,
                timeout_s=self.timeout_s,
            )
            stage_reports.append(
                {
                    "stage": stage,
                    "elapsed_s": result.elapsed_s,
                    "stdout_tail": result.stdout.splitlines()[-10:],
                }
            )
        assets = _collect_native(result_directory) or _collect_native(target)
        if not assets:
            raise RuntimeError("HY-World generation completed but no mesh/PLY/SPZ was found")
        return SceneBackendOutput(
            backend=self.name,
            native_assets=assets,
            metadata={"stages": stage_reports, "max_steps": max_steps},
        )

    def _prepare_panorama(self, request: SceneRequest, target: Path, worldgen: Path) -> None:
        panorama = target / "panorama.png"
        if request.images and request.metadata.get("input_is_panorama", False):
            shutil.copy2(request.images[0], panorama)
            return
        helper = target / "generate_panorama.py"
        helper.write_text(_panorama_helper(), encoding="utf-8")
        argv = [
            self.python_executable,
            str(helper),
            "--repo",
            str(self.repository),
            "--output",
            str(panorama),
            "--prompt",
            request.prompt or "A physically coherent photorealistic world",
            "--backend",
            self.panorama_backend,
        ]
        if request.images:
            argv += ["--image", str(request.images[0])]
        else:
            argv.append("--prompt-only")
        run_process(argv, cwd=worldgen, timeout_s=self.timeout_s)


def _panorama_helper() -> str:
    """Return an isolated official HY-Pano API launcher script."""

    return """from __future__ import annotations
import argparse
import sys
from pathlib import Path
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True)
p.add_argument('--image')
p.add_argument('--output', required=True)
p.add_argument('--prompt', required=True)
p.add_argument('--backend', default='hunyuanimage3')
p.add_argument('--prompt-only', action='store_true')
a = p.parse_args()
repo = Path(a.repo).resolve()
pano_dir = repo / 'hyworld2' / 'panogen'
sys.path.insert(0, str(pano_dir))
if a.backend == 'qwen-image-edit':
    import torch
    from pipeline_with_qwen_image import HunyuanPanoPipeline
    pipeline = HunyuanPanoPipeline.from_pretrained(
        pretrained_model_name_or_path='Qwen/Qwen-Image-Edit-2509',
        lora_path='tencent/HY-World-2.0',
        lora_subfolder='HY-Pano-2.0',
        torch_dtype=torch.bfloat16,
    )
else:
    from pipeline import HunyuanPanoPipeline
    pipeline = HunyuanPanoPipeline.from_pretrained(
        pretrained_model_name_or_path='tencent/HY-World-2.0',
        subfolder='HY-Pano-2.0',
        attn_impl='sdpa',
        moe_impl='eager',
    )
if a.prompt_only:
    seed_path = Path(a.output).with_name('_prompt_seed.png')
    Image.new('RGB', (1024, 512), (127, 127, 127)).save(seed_path)
    image = str(seed_path)
    task = 'auto'
else:
    image = a.image
    task = 'think_recaption'
out = pipeline(
    image,
    prompt=a.prompt,
    seed=0,
    height=960,
    width=1952,
    bot_task=task,
)
out.save(a.output)
"""


__all__ = ["HYWorld2GenerationBackend", "HYWorld2ReconstructionBackend"]
