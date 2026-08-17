"""HTTP adapter for the official GPT-SoVITS ``api_v2.py`` service."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from ...core import Availability, ConfigurationError
from ..tts_schema import TTSConfig, TTSRequest
from ..tts_types import TTSBackendOutput
from .base import TTSBackend


class GPTSoVITSAPIBackend(TTSBackend):
    """Use GPT-SoVITS as a measured comparison backend, not an embedded dependency."""

    name = "gpt-sovits-api"

    def __init__(
        self,
        *,
        endpoint: str = "http://127.0.0.1:9880/tts",
        allow_remote_backend: bool = False,
        timeout_s: float = 600.0,
        prompt_language: str | None = None,
        text_language: str | None = None,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        text_split_method: str = "cut5",
        **_: object,
    ) -> None:
        self.endpoint = endpoint.strip()
        self.allow_remote_backend = bool(allow_remote_backend)
        self.timeout_s = float(timeout_s)
        self.prompt_language = prompt_language
        self.text_language = text_language
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        self.text_split_method = str(text_split_method)
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigurationError("GPT-SoVITS endpoint must be an HTTP(S) URL")

    def availability(self) -> Availability:
        if not self.allow_remote_backend:
            return Availability(
                False,
                "set allow_remote_backend=True after starting the official api_v2.py service",
            )
        return Availability(True, details={"endpoint": self.endpoint})

    def run(
        self,
        request: TTSRequest,
        config: TTSConfig,
        work_directory: Path,
    ) -> TTSBackendOutput:
        self.availability().require(self.name)
        if not request.reference_transcript:
            raise ConfigurationError(
                "GPT-SoVITS requires reference_transcript for zero-shot API inference"
            )
        work_directory.mkdir(parents=True, exist_ok=True)
        output = work_directory / "gpt_sovits.wav"
        language = request.language.casefold()
        if language == "auto":
            language = "auto"
        payload = {
            "text": request.text,
            "text_lang": self.text_language or language,
            "ref_audio_path": str(request.target_reference.resolve()),
            "prompt_lang": self.prompt_language or language,
            "prompt_text": request.reference_transcript,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "text_split_method": self.text_split_method,
            "speed_factor": 1.0 / request.duration_factor,
            "media_type": "wav",
            "streaming_mode": request.streaming,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_s) as response:
                audio = response.read()
                content_type = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"GPT-SoVITS API returned HTTP {exc.code}: {detail[:1000]}"
            ) from exc
        elapsed = time.perf_counter() - started
        if not audio:
            raise RuntimeError("GPT-SoVITS API returned an empty response")
        output.write_bytes(audio)
        return TTSBackendOutput(
            backend=self.name,
            audio=output,
            elapsed_s=elapsed,
            metadata={"endpoint": self.endpoint, "content_type": content_type},
        )


__all__ = ["GPTSoVITSAPIBackend"]
