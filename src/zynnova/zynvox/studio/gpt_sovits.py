"""Managed local GPT-SoVITS API-v2 engine behind the ZynVox Studio API.

GPT-SoVITS remains an external checkout.  ZynNova owns the public API and process
lifecycle, while this adapter translates the stable Studio request to the upstream
local `/tts` contract.
"""
from __future__ import annotations

import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..policy import enforce_consent_record
from .types import GenerationRequest, GenerationResult, VoiceProfile


@dataclass(frozen=True, slots=True)
class GPTSoVITSLocalConfig:
    root: Path
    python: str = "python"
    host: str = "127.0.0.1"
    port: int = 9880
    tts_config: str = "GPT_SoVITS/configs/tts_infer.yaml"
    gpt_weights: Path | None = None
    sovits_weights: Path | None = None
    startup_timeout_s: float = 180.0

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        if not (root / "api_v2.py").is_file():
            raise FileNotFoundError(root / "api_v2.py")
        object.__setattr__(self, "root", root)
        if self.gpt_weights is not None:
            object.__setattr__(self, "gpt_weights", Path(self.gpt_weights).expanduser().resolve())
        if self.sovits_weights is not None:
            object.__setattr__(self, "sovits_weights", Path(self.sovits_weights).expanduser().resolve())


class GPTSoVITSLocalEngine:
    """Start/stop and use an external GPT-SoVITS `api_v2.py` process locally."""

    name = "gpt-sovits-local"

    def __init__(self, config: GPTSoVITSLocalConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._weights_initialized = False

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    def _port_open(self) -> bool:
        try:
            with socket.create_connection((self.config.host, self.config.port), timeout=0.5):
                return True
        except OSError:
            return False

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None and self._port_open():
            return
        if self._port_open():
            # Reuse an already-running local API. This is useful when the user starts
            # GPT-SoVITS in a dedicated environment/container.
            self._process = None
            self._initialize_weights()
            return
        command = [
            self.config.python,
            "api_v2.py",
            "-a",
            self.config.host,
            "-p",
            str(self.config.port),
            "-c",
            self.config.tts_config,
        ]
        self._process = subprocess.Popen(
            command,
            cwd=self.config.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + self.config.startup_timeout_s
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                tail = ""
                if self._process.stdout is not None:
                    tail = self._process.stdout.read()[-5000:]
                raise RuntimeError(f"GPT-SoVITS exited during startup: {tail}")
            if self._port_open():
                self._initialize_weights()
                return
            time.sleep(0.25)
        self.stop()
        raise TimeoutError(f"GPT-SoVITS did not open {self.config.host}:{self.config.port}")

    def _initialize_weights(self) -> None:
        if self._weights_initialized:
            return
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("install zynnova[voice-studio] for the managed GPT-SoVITS engine") from exc
        with httpx.Client(base_url=self.base_url, timeout=120.0) as client:
            if self.config.gpt_weights is not None:
                response = client.get("/set_gpt_weights", params={"weights_path": str(self.config.gpt_weights)})
                response.raise_for_status()
            if self.config.sovits_weights is not None:
                response = client.get("/set_sovits_weights", params={"weights_path": str(self.config.sovits_weights)})
                response.raise_for_status()
        self._weights_initialized = True

    def stop(self) -> None:
        process = self._process
        self._process = None
        self._weights_initialized = False
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def synthesize(self, request: GenerationRequest, profile: VoiceProfile, output: Path) -> GenerationResult:
        enforce_consent_record(profile.consent)
        self.start()
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("install zynnova[voice-studio]") from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        language = request.language.lower()
        if language == "auto":
            language = profile.language.lower()
        if language == "auto":
            raise ValueError("GPT-SoVITS local inference requires an explicit text language")
        prompt_language = str(request.extra.get("prompt_lang") or profile.language).lower()
        if prompt_language == "auto":
            prompt_language = language
        streaming_mode = request.extra.get("streaming_mode", 1 if request.streaming else 0)
        payload = {
            "text": request.text,
            "text_lang": language,
            "ref_audio_path": str(profile.reference_audio),
            "aux_ref_audio_paths": list(request.extra.get("aux_ref_audio_paths", [])),
            "prompt_text": profile.reference_text or "",
            "prompt_lang": prompt_language,
            "top_k": request.top_k,
            "top_p": request.top_p,
            "temperature": request.temperature,
            "text_split_method": request.split_method if request.split_method != "auto" else "cut5",
            "batch_size": request.batch_size,
            "batch_threshold": float(request.extra.get("batch_threshold", 0.75)),
            "split_bucket": bool(request.extra.get("split_bucket", True)),
            "speed_factor": request.speed,
            "fragment_interval": request.fragment_interval,
            "seed": request.seed,
            "media_type": "wav",
            "streaming_mode": streaming_mode,
            "parallel_infer": request.parallel_infer,
            "repetition_penalty": request.repetition_penalty,
            "sample_steps": int(request.extra.get("sample_steps", 32)),
            "super_sampling": bool(request.extra.get("super_sampling", False)),
            "overlap_length": int(request.extra.get("overlap_length", 2)),
            "min_chunk_length": int(request.extra.get("min_chunk_length", 16)),
        }
        started = time.perf_counter()
        with httpx.Client(base_url=self.base_url, timeout=None) as client:
            with client.stream("POST", "/tts", json=payload) as response:
                response.raise_for_status()
                with output.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
        elapsed = time.perf_counter() - started
        if output.stat().st_size == 0:
            raise RuntimeError("GPT-SoVITS returned an empty audio stream")
        return GenerationResult(
            audio=output,
            engine=self.name,
            model=request.model or profile.model,
            elapsed_s=elapsed,
            metadata={"upstream": "GPT-SoVITS api_v2", "streaming_mode": streaming_mode},
        )

    def convert(self, source: Path, profile: VoiceProfile, output: Path, **options: object) -> GenerationResult:
        raise NotImplementedError(
            "GPTSoVITSLocalEngine is the TTS engine; use an existing ZynVox VC backend "
            "or a composite/custom VoiceEngine for voice conversion"
        )

    def __enter__(self) -> "GPTSoVITSLocalEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


__all__ = ["GPTSoVITSLocalConfig", "GPTSoVITSLocalEngine"]
