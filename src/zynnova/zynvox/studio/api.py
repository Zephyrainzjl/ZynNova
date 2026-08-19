"""FastAPI surface owned by ZynNova, independent of the selected acoustic engine."""
# IMPORTANT: do not enable postponed annotations in this module.
# The request Pydantic models are intentionally local to create_app();
# FastAPI must receive the concrete model classes when routes are registered.

from pathlib import Path
from typing import Any

from ..schema import ConsentBasis, ConsentRecord
from .engine import VoiceEngineProfile
from .preprocess import DatasetPrepareConfig, prepare_dataset
from .studio import ZynVoxStudio
from .training import TrainingConfig, train_voice_model
from .types import GenerationRequest


def create_app(studio: ZynVoxStudio | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("install zynnova[voice-studio]") from exc

    studio = studio or ZynVoxStudio()
    app = FastAPI(title="ZynVox Studio API", version="1.0")

    class EnrollBody(BaseModel):
        voice_id: str
        reference_audio: str
        reference_text: str | None = None
        language: str = "auto"
        model: str | None = None
        consent_confirmed: bool = True
        consent_basis: str = "self"
        consent_purpose: str = "authorized speech synthesis"
        consent_evidence: str | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)

    class SpeechBody(BaseModel):
        input: str
        voice: str
        model: str | None = None
        language: str = "auto"
        speed: float = 1.0
        seed: int = -1
        top_k: int = 15
        top_p: float = 1.0
        temperature: float = 1.0
        repetition_penalty: float = 1.35
        batch_size: int = 1
        split_method: str = "auto"
        parallel_infer: bool = True
        stream: bool = False
        extra: dict[str, Any] = Field(default_factory=dict)

    class VCBody(BaseModel):
        source_audio: str
        voice: str
        output_name: str = "converted"
        options: dict[str, Any] = Field(default_factory=dict)

    class DatasetBody(BaseModel):
        dataset_name: str
        input_audio: list[str]
        language: str = "auto"
        sample_rate: int = 32000
        min_segment_s: float = 1.0
        max_segment_s: float = 15.0
        silence_db: float = -42.0
        transcribe: bool = True
        whisper_model: str = "large-v3"

    class TrainingBody(BaseModel):
        dataset_manifest: str
        run_name: str
        engine_name: str
        engine_root: str
        engine_python: str | None = None
        stages: list[str] = ["prepare-text", "ssl-features", "semantic", "acoustic"]
        stage_commands: dict[str, list[str]] = Field(default_factory=dict)
        batch_size: int = 4
        epochs_semantic: int = 15
        epochs_acoustic: int = 8
        precision: str = "bf16"
        device: str = "cuda"
        extra: dict[str, Any] = Field(default_factory=dict)

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "service": "zynvox-studio", "engine": studio.engine.name}

    @app.get("/v1/voices")
    def voices() -> dict[str, Any]:
        return {"data": [{"id": value, "object": "voice"} for value in studio.list_voices()]}

    @app.post("/v1/voices/enroll")
    def enroll(body: EnrollBody) -> dict[str, Any]:
        try:
            consent = ConsentRecord(
                confirmed=body.consent_confirmed,
                basis=ConsentBasis(body.consent_basis),
                purpose=body.consent_purpose,
                evidence=Path(body.consent_evidence) if body.consent_evidence else None,
            )
            profile = studio.enroll_voice(
                body.voice_id,
                body.reference_audio,
                consent,
                reference_text=body.reference_text,
                language=body.language,
                model=body.model,
                metadata=body.metadata,
            )
            return {"id": profile.voice_id, "object": "voice", "reference_audio": str(profile.reference_audio)}
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        values = []
        for path in sorted(studio.workspace.models.iterdir()):
            if path.is_dir() or path.is_file():
                values.append({"id": path.name, "object": "model", "path": str(path)})
        return {"data": values}

    @app.post("/v1/audio/speech")
    def speech(body: SpeechBody):
        try:
            result = studio.synthesize(
                GenerationRequest(
                    text=body.input,
                    voice_id=body.voice,
                    model=body.model,
                    language=body.language,
                    speed=body.speed,
                    seed=body.seed,
                    top_k=body.top_k,
                    top_p=body.top_p,
                    temperature=body.temperature,
                    repetition_penalty=body.repetition_penalty,
                    batch_size=body.batch_size,
                    split_method=body.split_method,
                    streaming=body.stream,
                    parallel_infer=body.parallel_infer,
                    extra=body.extra,
                )
            )
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        if body.stream:
            def iterator():
                with result.audio.open("rb") as handle:
                    while chunk := handle.read(64 * 1024):
                        yield chunk
            return StreamingResponse(iterator(), media_type="audio/wav", headers={"X-ZynVox-Engine": result.engine})
        return FileResponse(result.audio, media_type="audio/wav", filename=result.audio.name)

    @app.post("/v1/audio/voice-conversion")
    def convert(body: VCBody):
        try:
            result = studio.voice_convert(body.source_audio, body.voice, output_name=body.output_name, **body.options)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        return FileResponse(result.audio, media_type="audio/wav", filename=result.audio.name)

    @app.post("/v1/datasets/prepare")
    def dataset_prepare(body: DatasetBody) -> dict[str, Any]:
        try:
            manifest = prepare_dataset(
                DatasetPrepareConfig(
                    dataset_name=body.dataset_name,
                    input_audio=tuple(Path(p) for p in body.input_audio),
                    language=body.language,
                    sample_rate=body.sample_rate,
                    min_segment_s=body.min_segment_s,
                    max_segment_s=body.max_segment_s,
                    silence_db=body.silence_db,
                    transcribe=body.transcribe,
                    whisper_model=body.whisper_model,
                ),
                studio.workspace,
            )
            return {"dataset": body.dataset_name, "manifest": str(manifest)}
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/v1/training/run")
    def training(body: TrainingBody) -> dict[str, Any]:
        try:
            profile = VoiceEngineProfile(
                name=body.engine_name,
                root=Path(body.engine_root),
                python=body.engine_python or __import__("sys").executable,
            )
            result = train_voice_model(
                TrainingConfig(
                    dataset_manifest=Path(body.dataset_manifest),
                    run_name=body.run_name,
                    stages=tuple(body.stages),
                    stage_commands={key: tuple(value) for key, value in body.stage_commands.items()},
                    batch_size=body.batch_size,
                    epochs_semantic=body.epochs_semantic,
                    epochs_acoustic=body.epochs_acoustic,
                    precision=body.precision,
                    device=body.device,
                    extra=body.extra,
                ),
                profile,
                studio.workspace,
            )
            return {
                "run_directory": str(result.run_directory),
                "model_directory": str(result.model_directory),
                "elapsed_s": result.elapsed_s,
                "logs": {key: str(value) for key, value in result.stage_logs.items()},
            }
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

    return app


__all__ = ["create_app"]