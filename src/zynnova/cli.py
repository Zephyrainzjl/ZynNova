"""Command-line entry point for reproducible ZynNova workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__, backend_status
from .core.serialization import dump_json, to_jsonable
from .zynform import FEMConfig, FEMMethod, ObjectConfig, ObjectRequest, run_object
from .zynmorph import GenerationConfig, MicrostructureCondition, run_zynmorph
from .zynvista import SceneConfig, SceneMode, SceneRequest, run_scene
from .zynvox import (
    ConsentBasis,
    ConsentRecord,
    VoiceConfig,
    VoiceMode,
    VoiceRequest,
    TTSConfig,
    TTSRequest,
    run_speech_synthesis,
    run_voice_conversion,
)


def _json_object(value: str) -> dict[str, Any]:
    candidate = Path(value)
    if candidate.is_file():
        data = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        data = json.loads(value)
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("value must decode to a JSON object")
    return data


def _shape(value: str) -> tuple[int, int, int]:
    try:
        parts = tuple(int(item) for item in value.lower().replace("x", ",").split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be Z,Y,X or ZxYxX (for example 64x64x64)") from exc
    if len(parts) != 3 or min(parts) < 2:
        raise argparse.ArgumentTypeError("shape must contain three integers >= 2")
    return parts  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zynnova")
    parser.add_argument("--version", action="version", version=f"ZynNova {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="inspect optional backend availability")
    status.add_argument("--output", type=Path)

    morph = subparsers.add_parser("morph", help="generate and mesh a battery microstructure")
    morph.add_argument("--shape", type=_shape, default=(32, 32, 32))
    morph.add_argument(
        "--phase-fractions",
        type=_json_object,
        required=True,
        help='JSON such as \'{"1":0.45,"2":0.35,"5":0.20}\'',
    )
    morph.add_argument("--conditions", type=_json_object, default={})
    morph.add_argument("--backend", default="auto")
    morph.add_argument("--output", default="zynnova_runs/zynmorph")
    morph.add_argument("--backend-options", type=_json_object, default={})

    scene = subparsers.add_parser("scene", help="reconstruct or generate a 3D scene")
    scene.add_argument("--image", type=Path, action="append", default=[])
    scene.add_argument("--video", type=Path)
    scene.add_argument("--prompt")
    scene.add_argument("--mode", choices=[item.value for item in SceneMode], default="reconstruct")
    scene.add_argument("--backend", default="auto")
    scene.add_argument("--output", default="zynnova_runs/zynvista")
    scene.add_argument("--formats", default="ply,obj,glb")
    scene.add_argument("--backend-options", type=_json_object, default={})

    object_parser = subparsers.add_parser("object", help="generate and FEM-mesh one 3D object")
    object_parser.add_argument("--image", type=Path, required=True)
    object_parser.add_argument("--prompt")
    object_parser.add_argument("--backend", default="auto")
    object_parser.add_argument("--output", default="zynnova_runs/zynform")
    object_parser.add_argument("--formats", default="glb,obj,ply,stl")
    object_parser.add_argument("--fem-method", choices=[item.value for item in FEMMethod], default="auto")
    object_parser.add_argument("--no-fem", action="store_true")
    object_parser.add_argument("--backend-options", type=_json_object, default={})

    voice = subparsers.add_parser("voice", help="run an authorized voice conversion")
    voice.add_argument("--source", type=Path, required=True)
    voice.add_argument("--target", type=Path, required=True)
    voice.add_argument("--backend", default="auto")
    voice.add_argument(
        "--mode",
        choices=[VoiceMode.OFFLINE.value, VoiceMode.STREAMING_FILE.value],
        default=VoiceMode.OFFLINE.value,
    )
    voice.add_argument("--basis", choices=[item.value for item in ConsentBasis], required=True)
    voice.add_argument("--purpose", required=True)
    voice.add_argument(
        "--confirm-consent",
        action="store_true",
        help="required: confirm ownership, authorization, or valid license",
    )
    voice.add_argument("--repository", type=Path)
    voice.add_argument("--python-executable")
    voice.add_argument("--output", default="zynnova_runs/zynvox")
    voice.add_argument("--backend-options", type=_json_object, default={})

    tts = subparsers.add_parser("tts", help="run authorized zero-shot speech synthesis")
    tts.add_argument("--text", required=True)
    tts.add_argument("--target", type=Path, required=True)
    tts.add_argument("--backend", default="auto")
    tts.add_argument("--language", default="AUTO")
    tts.add_argument("--reference-transcript")
    tts.add_argument("--emotion-reference", type=Path)
    tts.add_argument("--emotion-text")
    tts.add_argument("--emotion-alpha", type=float, default=1.0)
    tts.add_argument("--duration-factor", type=float, default=1.0)
    tts.add_argument("--style-instruction")
    tts.add_argument("--streaming", action="store_true")
    tts.add_argument("--basis", choices=[item.value for item in ConsentBasis], required=True)
    tts.add_argument("--purpose", required=True)
    tts.add_argument("--confirm-consent", action="store_true")
    tts.add_argument("--repository", type=Path)
    tts.add_argument("--model-directory", type=Path)
    tts.add_argument("--python-executable")
    tts.add_argument("--output", default="zynnova_runs/zynvox_tts")
    tts.add_argument("--backend-options", type=_json_object, default={})

    ui = subparsers.add_parser("voice-ui", help="launch the optional ZynVox Gradio UI")
    ui.add_argument("--share", action="store_true")
    ui.add_argument("--server-name", default="127.0.0.1")
    ui.add_argument("--server-port", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        report = backend_status()
        if args.output is not None:
            dump_json(args.output, report)
        print(json.dumps(to_jsonable(report), indent=2, ensure_ascii=False))
        return 0
    if args.command == "morph":
        conditions = dict(args.conditions)
        condition = MicrostructureCondition(
            shape=args.shape,
            phase_fractions={int(key): value for key, value in args.phase_fractions.items()},
            **conditions,
        )
        result = run_zynmorph(
            condition,
            GenerationConfig(backend=args.backend, output_directory=args.output),
            backend_options=args.backend_options,
        )
        print(result.directory)
        return 0
    if args.command == "scene":
        request = SceneRequest(
            images=tuple(args.image),
            video=args.video,
            prompt=args.prompt,
            mode=SceneMode(args.mode),
            backend=args.backend,
        )
        result = run_scene(
            request,
            SceneConfig(
                output_directory=args.output,
                export_formats=tuple(item.strip() for item in args.formats.split(",") if item.strip()),
                backend_options=args.backend_options,
            ),
        )
        print(result.run_directory)
        return 0
    if args.command == "object":
        result = run_object(
            ObjectRequest(image=args.image, prompt=args.prompt, backend=args.backend),
            ObjectConfig(
                output_directory=args.output,
                export_formats=tuple(item.strip() for item in args.formats.split(",") if item.strip()),
                generate_fem=not args.no_fem,
                fem=FEMConfig(method=FEMMethod(args.fem_method)),
                backend_options=args.backend_options,
            ),
        )
        print(result.run_directory)
        return 0
    if args.command == "voice":
        options = dict(args.backend_options)
        if args.repository is not None:
            options["repository"] = str(args.repository)
        if args.python_executable:
            options["python_executable"] = args.python_executable
        request = VoiceRequest(
            source_audio=args.source,
            target_reference=args.target,
            backend=args.backend,
            mode=VoiceMode(args.mode),
            consent=ConsentRecord(
                confirmed=args.confirm_consent,
                basis=ConsentBasis(args.basis),
                purpose=args.purpose,
            ),
        )
        result = run_voice_conversion(
            request,
            VoiceConfig(output_directory=args.output, backend_options=options),
        )
        print(result.output_audio)
        return 0
    if args.command == "tts":
        options = dict(args.backend_options)
        if args.repository is not None:
            options["repository"] = str(args.repository)
        if args.model_directory is not None:
            options["model_directory"] = str(args.model_directory)
        if args.python_executable:
            options["python_executable"] = args.python_executable
        result = run_speech_synthesis(
            TTSRequest(
                text=args.text,
                target_reference=args.target,
                backend=args.backend,
                language=args.language,
                reference_transcript=args.reference_transcript,
                emotion_reference=args.emotion_reference,
                emotion_text=args.emotion_text,
                emotion_alpha=args.emotion_alpha,
                duration_factor=args.duration_factor,
                style_instruction=args.style_instruction,
                streaming=args.streaming,
                consent=ConsentRecord(
                    confirmed=args.confirm_consent,
                    basis=ConsentBasis(args.basis),
                    purpose=args.purpose,
                ),
            ),
            TTSConfig(output_directory=args.output, backend_options=options),
        )
        print(result.output_audio)
        return 0
    if args.command == "voice-ui":
        from .zynvox.ui import launch_ui

        kwargs: dict[str, object] = {
            "share": args.share,
            "server_name": args.server_name,
        }
        if args.server_port is not None:
            kwargs["server_port"] = args.server_port
        launch_ui(**kwargs)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
