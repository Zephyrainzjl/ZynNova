"""Adapter for the official MeanVC2 end-to-end and streaming entry points."""

from __future__ import annotations

import sys
from pathlib import Path

from ...core import Availability, dump_json, run_process, sha256_file
from ...core.backend import executable_availability
from ..policy import enforce_consent_record
from ..schema import ConsentRecord, VoiceConfig, VoiceMode, VoiceRequest
from ..types import VoiceBackendOutput
from .base import VoiceBackend


class MeanVC2Backend(VoiceBackend):
    name = "meanvc2"

    def __init__(
        self,
        *,
        repository: str | Path | None = None,
        python_executable: str = sys.executable,
        model: str = "120ms",
        steps: int = 3,
        timeout_s: float | None = None,
        **_: object,
    ) -> None:
        self.repository = None if repository is None else Path(repository).expanduser().resolve()
        self.python_executable = str(python_executable)
        if model not in {"40ms", "120ms"}:
            raise ValueError("MeanVC2 model must be '40ms' or '120ms'")
        if steps < 1:
            raise ValueError("steps must be positive")
        self.model = model
        self.steps = int(steps)
        self.timeout_s = timeout_s

    def availability(self) -> Availability:
        if self.repository is None:
            return Availability(
                False,
                "repository is required; pass backend_options={'repository': ...}",
            )
        status = executable_availability(self.python_executable)
        if not status.available and not Path(self.python_executable).is_file():
            return status
        script = self.repository / "src" / "infer" / "infer_e2e.py"
        if not script.is_file():
            return Availability(False, f"MeanVC2 infer_e2e.py not found under {self.repository}")
        if not (self.repository / "ckpts").is_dir():
            return Availability(
                False,
                "MeanVC2 checkpoints are absent; run initialization.py --task all",
            )
        return Availability(
            True,
            details={"repository": str(self.repository), "model": self.model},
        )

    def run(
        self,
        request: VoiceRequest,
        config: VoiceConfig,
        work_directory: Path,
    ) -> VoiceBackendOutput:
        if request.mode is VoiceMode.REALTIME:
            raise ValueError("use launch_meanvc2_realtime() for microphone streaming")
        if self.repository is None:  # guarded by registry, retained for direct construction
            raise ValueError("repository is required for MeanVC2")
        work_directory.mkdir(parents=True, exist_ok=True)
        output = work_directory / "meanvc2.wav"
        script = self.repository / "src" / "infer" / "infer_e2e.py"
        model = "40ms" if request.mode is VoiceMode.STREAMING_FILE else self.model
        result = run_process(
            [
                self.python_executable,
                str(script),
                "--model",
                model,
                "--source-wav",
                str(request.source_audio.resolve()),
                "--target-wav",
                str(request.target_reference.resolve()),
                "--output-wav",
                str(output),
                "--steps",
                str(self.steps),
            ],
            cwd=self.repository,
            timeout_s=self.timeout_s,
        )
        if not output.is_file():
            raise RuntimeError(f"MeanVC2 did not create {output}")
        return VoiceBackendOutput(
            backend=self.name,
            audio=output,
            elapsed_s=result.elapsed_s,
            metadata={
                "model": model,
                "steps": self.steps,
                "stdout_tail": result.stdout.splitlines()[-20:],
                "stderr_tail": result.stderr.splitlines()[-20:],
            },
        )


def launch_meanvc2_realtime(
    repository: str | Path,
    *,
    consent: ConsentRecord,
    target_reference: str | Path,
    python_executable: str = sys.executable,
    model: str = "40ms",
    input_audio: str | Path | None = None,
    output_audio: str | Path | None = None,
    audit_directory: str | Path = "zynnova_runs/zynvox_realtime",
    timeout_s: float | None = None,
) -> object:
    """Invoke MeanVC2 runtime only after binding the run to authorization evidence.

    ``target_reference`` is intentionally required even though the upstream realtime
    application may manage its own reference internally: the ZynVox audit boundary
    must bind every interactive launch to the voice authorization the user asserts.
    """

    policy = enforce_consent_record(consent)
    target = Path(target_reference).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    root = Path(repository).expanduser().resolve()
    script = root / "runtime" / "run_rt.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    if model not in {"40ms", "120ms"}:
        raise ValueError("model must be '40ms' or '120ms'")
    audit_root = Path(audit_directory).expanduser().resolve()
    audit_root.mkdir(parents=True, exist_ok=True)
    receipt = audit_root / f"meanvc2_{consent.record_id}.authorization.json"
    dump_json(
        receipt,
        {
            "schema": "zynnova.realtime-authorization/1.0",
            "backend": "meanvc2",
            "consent": {
                "record_id": consent.record_id,
                "basis": consent.basis.value,
                "purpose": consent.purpose,
                "recorded_at": consent.recorded_at,
                "evidence_sha256": None
                if consent.evidence is None
                else sha256_file(consent.evidence),
            },
            "target_reference_sha256": sha256_file(target),
            "policy": {
                "authorized": policy.authorized,
                "evidence_required": policy.evidence_required,
                "evidence_present": policy.evidence_present,
            },
            "mode": "realtime" if input_audio is None else "file",
            "model": model,
        },
    )
    argv = [python_executable, str(script)]
    if input_audio is None:
        argv.extend(["--mode", "realtime", "--model", model])
    else:
        source = Path(input_audio).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if output_audio is None:
            raise ValueError("output_audio is required in file mode")
        argv.extend(
            [
                "--mode",
                "file",
                "--input",
                str(source),
                "--output",
                str(Path(output_audio).expanduser().resolve()),
                "--model",
                model,
            ]
        )
    return run_process(argv, cwd=root / "runtime", timeout_s=timeout_s)


__all__ = ["MeanVC2Backend", "launch_meanvc2_realtime"]
