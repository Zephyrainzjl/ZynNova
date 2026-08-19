"""Command-line entry point for model download, fine-tuning, chat and API serving."""
from __future__ import annotations

import argparse
import asyncio
import os

from .config import AgentConfig, ProviderConfig
from .finetune import SFTConfig, finetune_lora
from .models import download_model
from .runtime import Agent
from .workspace import Workspace


def _provider(args) -> ProviderConfig:
    return ProviderConfig(
        kind=args.provider, model=args.model, base_url=args.base_url,
        api_key_env=args.api_key_env, api_style=args.api_style,
        reasoning_effort=getattr(args,"reasoning_effort",None),
    )


def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(prog="zynnova-llm", description="ZynAstra agent framework")
    p.add_argument("--workspace", default=os.environ.get("ZYNNOVA_WORKSPACE","~/.zynnova"))
    sub=p.add_subparsers(dest="command", required=True)
    chat=sub.add_parser("chat")
    chat.add_argument("prompt")
    for q in (chat,):
        q.add_argument("--provider", default="openai-compatible")
        q.add_argument("--model", default="gpt-5.6")
        q.add_argument("--base-url", default="https://api.openai.com/v1")
        q.add_argument("--api-key-env", default="OPENAI_API_KEY")
        q.add_argument("--api-style", default="auto")
        q.add_argument("--reasoning-effort", default=None)
    dl=sub.add_parser("download-model"); dl.add_argument("model_id"); dl.add_argument("--revision")
    ft=sub.add_parser("finetune"); ft.add_argument("base_model"); ft.add_argument("dataset"); ft.add_argument("--text-field",default="text"); ft.add_argument("--output-name",default="adapter"); ft.add_argument("--qlora-4bit",action="store_true")
    serve=sub.add_parser("serve")
    serve.add_argument("--provider", default="openai-compatible"); serve.add_argument("--model",default="gpt-5.6"); serve.add_argument("--base-url",default="https://api.openai.com/v1"); serve.add_argument("--api-key-env",default="OPENAI_API_KEY"); serve.add_argument("--api-style",default="auto"); serve.add_argument("--host",default="127.0.0.1"); serve.add_argument("--port",type=int,default=8765)
    return p


async def _chat(args) -> None:
    agent=Agent.create(_provider(args), Workspace(args.workspace), config=AgentConfig())
    try:
        result=await agent.run(args.prompt); print(result.text)
    finally: await agent.aclose()


def main() -> int:
    args=build_parser().parse_args(); ws=Workspace(args.workspace).ensure()
    if args.command=="chat": asyncio.run(_chat(args)); return 0
    if args.command=="download-model": print(download_model(args.model_id,ws,revision=args.revision).path); return 0
    if args.command=="finetune": print(finetune_lora(args.base_model,SFTConfig(args.dataset,text_field=args.text_field,output_name=args.output_name,qlora_4bit=args.qlora_4bit),ws)); return 0
    if args.command=="serve":
        try: import uvicorn
        except ImportError as exc: raise RuntimeError("install zynnova[llm-server]") from exc
        from .server import create_app
        uvicorn.run(create_app(_provider(args),str(ws.root)),host=args.host,port=args.port); return 0
    return 2


if __name__ == "__main__": raise SystemExit(main())
