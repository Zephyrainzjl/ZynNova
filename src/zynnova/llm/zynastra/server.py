"""Optional FastAPI service for ZynAstra."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .config import AgentConfig, ProviderConfig
from .runtime import Agent


def create_app(provider: ProviderConfig, workspace: str | None = None, *, agent_config: AgentConfig | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError("install zynnova[llm-server] to serve ZynAstra") from exc

    app = FastAPI(title="ZynAstra API", version="1.0")
    agent = Agent.create(provider, workspace, config=agent_config)

    class RunBody(BaseModel):
        prompt: str
        session_id: str | None = None
        reset: bool = False

    @app.on_event("shutdown")
    async def _shutdown() -> None: await agent.aclose()

    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "framework": "zynastra", "provider": provider.kind, "model": provider.model}

    @app.get("/v1/tools")
    async def tools() -> list[dict[str, Any]]:
        await agent.start(); return [spec.as_openai() for spec in agent.tools.specs()]

    @app.post("/v1/agent/run")
    async def run(body: RunBody) -> dict[str, Any]:
        try: result = await agent.run(body.prompt, session_id=body.session_id, reset=body.reset)
        except Exception as exc: raise HTTPException(500, str(exc)) from exc
        return {"text":result.text,"steps":result.steps,"session_id":result.session_id,"usage":dict(result.usage)}

    return app


__all__ = ["create_app"]
