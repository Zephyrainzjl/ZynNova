"""Optional Gradio UI backed by the validated ZynVox pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.serialization import to_jsonable
from .pipeline import run_voice_conversion
from .registry import VOICE_BACKENDS
from .schema import ConsentBasis, ConsentRecord, VoiceConfig, VoiceMode, VoiceRequest
from .tts_pipeline import run_speech_synthesis
from .tts_registry import TTS_BACKENDS
from .tts_schema import TTSConfig, TTSRequest


def _report(result: object) -> str:
    return json.dumps(
        {
            "run_directory": str(result.run_directory),
            "manifest": str(result.manifest_path),
            "provenance": None
            if result.provenance_path is None
            else str(result.provenance_path),
            "metrics": to_jsonable(result.metrics),
        },
        indent=2,
        ensure_ascii=False,
    )


def create_ui() -> object:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "ZynVox UI requires gradio; install zynnova[zynnova-ui]"
        ) from exc

    def convert(
        source: str,
        target: str,
        backend: str,
        mode: str,
        repository: str,
        python_executable: str,
        consent_basis: str,
        purpose: str,
        consent_confirmed: bool,
        options_json: str,
    ) -> tuple[str, str]:
        options: dict[str, Any] = json.loads(options_json or "{}")
        if repository.strip():
            options["repository"] = repository.strip()
        if python_executable.strip():
            options["python_executable"] = python_executable.strip()
        result = run_voice_conversion(
            VoiceRequest(
                source_audio=Path(source),
                target_reference=Path(target),
                backend=backend,
                mode=VoiceMode(mode),
                consent=ConsentRecord(
                    confirmed=consent_confirmed,
                    basis=ConsentBasis(consent_basis),
                    purpose=purpose,
                ),
            ),
            VoiceConfig(backend_options=options),
        )
        return str(result.output_audio), _report(result)

    def synthesize(
        text: str,
        target: str,
        backend: str,
        language: str,
        reference_transcript: str,
        emotion_reference: str | None,
        emotion_text: str,
        emotion_alpha: float,
        duration_factor: float,
        style_instruction: str,
        streaming: bool,
        repository: str,
        model_directory: str,
        python_executable: str,
        consent_basis: str,
        purpose: str,
        consent_confirmed: bool,
        options_json: str,
    ) -> tuple[str, str]:
        options: dict[str, Any] = json.loads(options_json or "{}")
        if repository.strip():
            options["repository"] = repository.strip()
        if model_directory.strip():
            options["model_directory"] = model_directory.strip()
        if python_executable.strip():
            options["python_executable"] = python_executable.strip()
        result = run_speech_synthesis(
            TTSRequest(
                text=text,
                target_reference=Path(target),
                backend=backend,
                language=language,
                reference_transcript=reference_transcript or None,
                emotion_reference=None
                if not emotion_reference
                else Path(emotion_reference),
                emotion_text=emotion_text or None,
                emotion_alpha=float(emotion_alpha),
                duration_factor=float(duration_factor),
                style_instruction=style_instruction or None,
                streaming=streaming,
                consent=ConsentRecord(
                    confirmed=consent_confirmed,
                    basis=ConsentBasis(consent_basis),
                    purpose=purpose,
                ),
            ),
            TTSConfig(backend_options=options),
        )
        return str(result.output_audio), _report(result)

    with gr.Blocks(title="ZynVox") as interface:
        gr.Markdown(
            "# ZynVox\nAuthorized speech-to-speech conversion and zero-shot speech "
            "synthesis with measured performance, explicit consent, and provenance."
        )
        with gr.Tabs():
            with gr.Tab("Voice conversion"):
                with gr.Row():
                    vc_source = gr.Audio(label="Source speech", type="filepath")
                    vc_target = gr.Audio(label="Authorized target reference", type="filepath")
                with gr.Row():
                    vc_backend = gr.Dropdown(
                        choices=["meanvc2", "xvc", "external-voice-contract"],
                        value="meanvc2",
                        label="Backend",
                    )
                    vc_mode = gr.Dropdown(
                        choices=[VoiceMode.OFFLINE.value, VoiceMode.STREAMING_FILE.value],
                        value=VoiceMode.OFFLINE.value,
                        label="Mode",
                    )
                vc_repository = gr.Textbox(label="Official backend repository path")
                vc_python = gr.Textbox(label="Isolated environment Python")
                vc_options = gr.Textbox(label="Additional backend options (JSON)", value="{}")
                with gr.Row():
                    vc_basis = gr.Dropdown(
                        choices=[item.value for item in ConsentBasis],
                        value=ConsentBasis.SELF.value,
                        label="Authorization basis",
                    )
                    vc_purpose = gr.Textbox(
                        label="Authorized purpose", value="research evaluation"
                    )
                vc_consent = gr.Checkbox(label="I confirm ownership or explicit permission/license.")
                vc_submit = gr.Button("Convert")
                vc_output = gr.Audio(label="Converted output", type="filepath")
                vc_report = gr.Code(label="Run report", language="json")
                vc_submit.click(
                    convert,
                    inputs=[
                        vc_source,
                        vc_target,
                        vc_backend,
                        vc_mode,
                        vc_repository,
                        vc_python,
                        vc_basis,
                        vc_purpose,
                        vc_consent,
                        vc_options,
                    ],
                    outputs=[vc_output, vc_report],
                )

            with gr.Tab("Zero-shot TTS"):
                tts_text = gr.Textbox(label="Text", lines=4)
                with gr.Row():
                    tts_target = gr.Audio(label="Authorized voice reference", type="filepath")
                    tts_emotion_reference = gr.Audio(
                        label="Optional emotion reference", type="filepath"
                    )
                with gr.Row():
                    tts_backend = gr.Dropdown(
                        choices=[
                            "cosyvoice-3",
                            "indextts-2.5",
                            "gpt-sovits-api",
                            "external-tts-contract",
                        ],
                        value="cosyvoice-3",
                        label="Backend",
                    )
                    tts_language = gr.Dropdown(
                        choices=["AUTO", "ZH", "EN", "JA", "ES", "AR", "KO", "DE", "FR", "IT", "RU"],
                        value="AUTO",
                        label="Language",
                    )
                tts_transcript = gr.Textbox(label="Reference transcript (recommended/required by some backends)")
                tts_emotion_text = gr.Textbox(label="Optional emotion description")
                tts_style = gr.Textbox(label="Optional style/accent instruction")
                with gr.Row():
                    tts_emotion_alpha = gr.Slider(0.0, 1.0, value=1.0, label="Emotion strength")
                    tts_duration = gr.Slider(0.5, 2.0, value=1.0, label="Duration factor")
                    tts_streaming = gr.Checkbox(label="Streaming backend mode")
                tts_repository = gr.Textbox(label="Official backend repository path")
                tts_model = gr.Textbox(label="Model/checkpoint directory")
                tts_python = gr.Textbox(label="Isolated environment Python")
                tts_options = gr.Textbox(label="Additional backend options (JSON)", value="{}")
                with gr.Row():
                    tts_basis = gr.Dropdown(
                        choices=[item.value for item in ConsentBasis],
                        value=ConsentBasis.SELF.value,
                        label="Authorization basis",
                    )
                    tts_purpose = gr.Textbox(
                        label="Authorized purpose", value="research evaluation"
                    )
                tts_consent = gr.Checkbox(label="I confirm ownership or explicit permission/license.")
                tts_submit = gr.Button("Synthesize")
                tts_output = gr.Audio(label="Synthesized output", type="filepath")
                tts_report = gr.Code(label="Run report", language="json")
                tts_submit.click(
                    synthesize,
                    inputs=[
                        tts_text,
                        tts_target,
                        tts_backend,
                        tts_language,
                        tts_transcript,
                        tts_emotion_reference,
                        tts_emotion_text,
                        tts_emotion_alpha,
                        tts_duration,
                        tts_style,
                        tts_streaming,
                        tts_repository,
                        tts_model,
                        tts_python,
                        tts_basis,
                        tts_purpose,
                        tts_consent,
                        tts_options,
                    ],
                    outputs=[tts_output, tts_report],
                )

        with gr.Accordion("Backend diagnostics", open=False):
            gr.JSON(value={"voice_conversion": VOICE_BACKENDS.status(), "tts": TTS_BACKENDS.status()})
    return interface


def launch_ui(**kwargs: object) -> object:
    interface = create_ui()
    return interface.launch(**kwargs)


__all__ = ["create_ui", "launch_ui"]
