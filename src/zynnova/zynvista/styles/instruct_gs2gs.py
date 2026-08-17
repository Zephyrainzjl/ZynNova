"""Official Instruct-GS2GS/Nerfstudio scene-editing adapter."""

from __future__ import annotations

import shutil
from pathlib import Path

from ...core import Availability, ConfigurationError, run_process
from ..schema import SceneConfig
from ..types import SceneBackendOutput
from .base import SceneStyleBackend


class InstructGS2GSStyle(SceneStyleBackend):
    """Edit an existing Gaussian scene through the official Nerfstudio plugin.

    The adapter deliberately runs the plugin in its own environment through CLI
    contracts. It does not import Nerfstudio into the main ZynNova environment.
    """

    name = "instruct-gs2gs"

    def __init__(
        self,
        *,
        data_directory: str | Path | None = None,
        load_directory: str | Path | None = None,
        ns_train: str = "ns-train",
        ns_export: str = "ns-export",
        guidance_scale: float = 12.5,
        image_guidance_scale: float = 1.5,
        max_num_iterations: int = 7_500,
        ip2p_device: str | None = None,
        timeout_s: float | None = None,
        **_: object,
    ) -> None:
        self.data_directory = (
            None if data_directory is None else Path(data_directory).expanduser().resolve()
        )
        self.load_directory = (
            None if load_directory is None else Path(load_directory).expanduser().resolve()
        )
        self.ns_train = str(ns_train)
        self.ns_export = str(ns_export)
        self.guidance_scale = float(guidance_scale)
        self.image_guidance_scale = float(image_guidance_scale)
        self.max_num_iterations = int(max_num_iterations)
        self.ip2p_device = None if ip2p_device is None else str(ip2p_device)
        self.timeout_s = timeout_s
        if self.guidance_scale <= 0.0 or self.image_guidance_scale <= 0.0:
            raise ConfigurationError("guidance scales must be positive")
        if self.max_num_iterations < 1:
            raise ConfigurationError("max_num_iterations must be positive")

    def availability(self) -> Availability:
        if self.data_directory is None or not self.data_directory.is_dir():
            return Availability(
                False,
                "data_directory containing the original Nerfstudio dataset is required",
            )
        if self.load_directory is None or not self.load_directory.is_dir():
            return Availability(
                False,
                "load_directory containing the trained splatfacto nerfstudio_models is required",
            )
        train = shutil.which(self.ns_train) if not Path(self.ns_train).is_file() else self.ns_train
        export = shutil.which(self.ns_export) if not Path(self.ns_export).is_file() else self.ns_export
        if train is None:
            return Availability(False, f"ns-train executable not found: {self.ns_train}")
        if export is None:
            return Availability(False, f"ns-export executable not found: {self.ns_export}")
        return Availability(
            True,
            details={
                "data_directory": str(self.data_directory),
                "load_directory": str(self.load_directory),
            },
        )

    def apply(
        self,
        output: SceneBackendOutput,
        config: SceneConfig,
        work_directory: Path,
    ) -> SceneBackendOutput:
        self.availability().require(self.name)
        prompt = (config.style_prompt or "").strip()
        if not prompt:
            raise ConfigurationError("Instruct-GS2GS requires SceneConfig.style_prompt")
        assert self.data_directory is not None
        assert self.load_directory is not None
        training_root = work_directory / "training"
        export_root = work_directory / "gaussian_export"
        training_root.mkdir(parents=True, exist_ok=True)
        argv = [
            self.ns_train,
            "igs2gs",
            "--data",
            str(self.data_directory),
            "--load-dir",
            str(self.load_directory),
            "--output-dir",
            str(training_root),
            "--max-num-iterations",
            str(self.max_num_iterations),
            "--pipeline.prompt",
            prompt,
            "--pipeline.guidance-scale",
            str(self.guidance_scale),
            "--pipeline.image-guidance-scale",
            str(self.image_guidance_scale),
        ]
        if self.ip2p_device:
            argv.extend(["--pipeline.ip2p-device", self.ip2p_device])
        train_result = run_process(argv, timeout_s=self.timeout_s)
        configs = sorted(
            training_root.rglob("config.yml"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not configs:
            configs = sorted(
                training_root.rglob("config.yaml"),
                key=lambda path: path.stat().st_mtime_ns,
            )
        if not configs:
            raise RuntimeError(f"Instruct-GS2GS produced no config file under {training_root}")
        trained_config = configs[-1]
        export_result = run_process(
            [
                self.ns_export,
                "gaussian-splat",
                "--load-config",
                str(trained_config),
                "--output-dir",
                str(export_root),
            ],
            timeout_s=self.timeout_s,
        )
        splats = sorted(export_root.rglob("*.ply"), key=lambda path: path.stat().st_mtime_ns)
        if not splats:
            raise RuntimeError(f"Nerfstudio exported no Gaussian PLY under {export_root}")
        assets = dict(output.native_assets)
        assets["styled_gaussian_splat"] = splats[-1]
        assets["styled_nerfstudio_config"] = trained_config
        return SceneBackendOutput(
            backend=output.backend,
            dense_views=output.dense_views,
            point_cloud=output.point_cloud,
            mesh=output.mesh,
            scene=output.scene,
            native_assets=assets,
            metadata={
                **output.metadata,
                "style_backend": self.name,
                "style_prompt": prompt,
                "style_iterations": self.max_num_iterations,
                "style_train_elapsed_s": train_result.elapsed_s,
                "style_export_elapsed_s": export_result.elapsed_s,
            },
        )


__all__ = ["InstructGS2GSStyle"]
