"""Minimal external-environment runner for the official CosyVoice 3 API."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    repository = Path(args.repository).resolve()
    sys.path.insert(0, str(repository))
    matcha = repository / "third_party" / "Matcha-TTS"
    if matcha.is_dir():
        sys.path.insert(0, str(matcha))

    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import AutoModel

    model = AutoModel(model_dir=str(Path(args.model_dir).resolve()))
    text = request["text"]
    reference = request["target_reference"]
    transcript = request.get("reference_transcript")
    instruction = request.get("style_instruction")
    stream = bool(request.get("streaming", False))
    system_prefix = "You are a helpful assistant.<|endofprompt|>"

    if instruction:
        generator = model.inference_instruct2(
            text,
            system_prefix + instruction + "<|endofprompt|>",
            reference,
            stream=stream,
        )
        method = "inference_instruct2"
    elif transcript:
        generator = model.inference_zero_shot(
            text,
            system_prefix + transcript,
            reference,
            stream=stream,
        )
        method = "inference_zero_shot"
    else:
        generator = model.inference_cross_lingual(
            system_prefix + text,
            reference,
            stream=stream,
        )
        method = "inference_cross_lingual"

    start = time.perf_counter()
    first_packet_ms = None
    chunks = []
    for item in generator:
        if first_packet_ms is None:
            first_packet_ms = (time.perf_counter() - start) * 1000.0
        chunk = item["tts_speech"]
        if chunk.ndim == 1:
            chunk = chunk.unsqueeze(0)
        chunks.append(chunk.detach().cpu())
    if not chunks:
        raise RuntimeError("CosyVoice 3 returned no audio chunks")
    audio = torch.cat(chunks, dim=-1)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output), audio, int(model.sample_rate))
    Path(args.metadata).write_text(
        json.dumps(
            {
                "method": method,
                "first_packet_latency_ms": first_packet_ms,
                "chunks": len(chunks),
                "sample_rate": int(model.sample_rate),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
