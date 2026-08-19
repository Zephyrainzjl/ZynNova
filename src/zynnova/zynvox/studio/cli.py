from __future__ import annotations
import argparse
from .api import create_app
from .studio import ZynVoxStudio
from .types import GenerationRequest


def main()->int:
    p=argparse.ArgumentParser(prog="zynvox-studio"); p.add_argument("--workspace",default=None)
    sub=p.add_subparsers(dest="cmd",required=True)
    s=sub.add_parser("speak"); s.add_argument("text"); s.add_argument("--voice",required=True); s.add_argument("--language",default="auto")
    api=sub.add_parser("serve"); api.add_argument("--host",default="127.0.0.1"); api.add_argument("--port",type=int,default=8770)
    ui=sub.add_parser("ui"); ui.add_argument("--host",default="127.0.0.1"); ui.add_argument("--port",type=int,default=7861)
    a=p.parse_args(); studio=ZynVoxStudio(a.workspace)
    if a.cmd=="speak": print(studio.synthesize(GenerationRequest(a.text,a.voice,language=a.language)).audio); return 0
    if a.cmd=="serve":
        try: import uvicorn
        except ImportError as exc: raise RuntimeError("install zynnova[voice-studio]") from exc
        uvicorn.run(create_app(studio),host=a.host,port=a.port); return 0
    if a.cmd=="ui":
        from .ui import build_ui; build_ui(studio).launch(server_name=a.host,server_port=a.port); return 0
    return 2


if __name__=="__main__": raise SystemExit(main())
