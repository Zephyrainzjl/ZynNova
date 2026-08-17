from __future__ import annotations

from pathlib import Path

from zynnova.zynvox import (
    ConsentBasis,
    ConsentRecord,
    TTSConfig,
    TTSRequest,
    run_speech_synthesis,
)

request = TTSRequest(
    text="This sample is generated for an authorized benchmark.",
    target_reference=Path("inputs/authorized_target.wav"),
    reference_transcript="Reference transcript used by prompt-conditioned systems.",
    backend="cosyvoice-3",
    language="EN",
    style_instruction="calm, clear, neutral delivery",
    streaming=True,
    consent=ConsentRecord(
        confirmed=True,
        basis=ConsentBasis.DIRECT_AUTHORIZATION,
        purpose="authorized speech synthesis benchmark",
        evidence=Path("inputs/consent_record.pdf"),
    ),
)
result = run_speech_synthesis(
    request,
    TTSConfig(
        output_directory="zynnova_runs/zynvox_tts_example",
        backend_options={
            "repository": "external/zynnova/cosyvoice",
            "model_directory": "external/models/Fun-CosyVoice3-0.5B",
            "python_executable": "external/envs/cosyvoice/bin/python",
        },
    ),
)
print(result.output_audio)
print(result.benchmark_path)
