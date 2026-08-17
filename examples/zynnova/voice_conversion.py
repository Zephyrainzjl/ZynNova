from __future__ import annotations

from pathlib import Path

from zynnova.zynvox import (
    ConsentBasis,
    ConsentRecord,
    VoiceConfig,
    VoiceMode,
    VoiceRequest,
    run_voice_conversion,
)

request = VoiceRequest(
    source_audio=Path("inputs/source.wav"),
    target_reference=Path("inputs/authorized_target.wav"),
    backend="meanvc2",
    mode=VoiceMode.OFFLINE,
    consent=ConsentRecord(
        confirmed=True,
        basis=ConsentBasis.DIRECT_AUTHORIZATION,
        purpose="authorized research evaluation",
        evidence=Path("inputs/consent_record.pdf"),
    ),
)
result = run_voice_conversion(
    request,
    VoiceConfig(
        output_directory="zynnova_runs/zynvox_example",
        backend_options={
            "repository": "external/zynnova/meanvc2",
            "python_executable": "external/envs/meanvc2/bin/python",
            "device": "cuda",
        },
    ),
)
print(result.output_audio)
print(result.benchmark_path)
