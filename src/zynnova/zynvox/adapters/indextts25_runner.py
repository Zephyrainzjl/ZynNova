"""Minimal external-environment runner for the official IndexTTS-2.5 API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    repository = Path(args.repository).resolve()
    sys.path.insert(0, str(repository))
    from indextts.infer_v2_5 import IndexTTS2

    use_qwen_emo = bool(request.get("emotion_text"))
    tts = IndexTTS2(
        cfg_path=str(Path(args.config).resolve()),
        model_dir=str(Path(args.model_dir).resolve()),
        use_bf16=bool(args.bf16),
        use_qwen_emo=use_qwen_emo,
    )
    kwargs = {
        "spk_audio_prompt": request["target_reference"],
        "text": request["text"],
        "output_path": str(Path(args.output).resolve()),
        "verbose": True,
        "emo_alpha": float(request.get("emotion_alpha", 1.0)),
        "duration_factor": float(request.get("duration_factor", 1.0)),
    }
    language = str(request.get("language", "AUTO")).upper()
    if language != "AUTO":
        kwargs["lang"] = language
    if request.get("emotion_reference"):
        kwargs["emo_audio_prompt"] = request["emotion_reference"]
    if request.get("emotion_vector") is not None:
        kwargs["emo_vector"] = request["emotion_vector"]
        kwargs["use_random"] = False
    if request.get("emotion_text"):
        kwargs["use_emo_text"] = True
        kwargs["emo_text"] = request["emotion_text"]
        kwargs["use_random"] = False
    tts.infer(**kwargs)
    output = Path(args.output)
    if not output.is_file():
        raise RuntimeError(f"IndexTTS-2.5 did not create {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
