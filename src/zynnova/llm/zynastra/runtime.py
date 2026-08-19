"""Tool-using ZynAstra agent loop."""
from __future__ import annotations

import json
import uuid
from collections import Counter
from typing import Mapping

from .config import AgentConfig, ProviderConfig
from .mcp import MCPConnection
from .memory import SessionStore
from .providers import ModelProvider, create_provider
from .skills import SkillManager
from .tools import ToolRegistry, install_zynnova_tools
from .types import AgentResult, Message
from .workspace import Workspace


class Agent:
    def __init__(
        self,
        provider: ModelProvider,
        workspace: Workspace,
        *,
        config: AgentConfig | None = None,
        tools: ToolRegistry | None = None,
        skills: SkillManager | None = None,
    ) -> None:
        self.provider = provider
        self.workspace = workspace.ensure()
        self.config = config or AgentConfig()
        self.tools = tools or ToolRegistry()
        self.skills = skills or SkillManager((self.workspace.skills,))
        self.memory = SessionStore(self.workspace.memory / "sessions.sqlite3")
        self._mcp: list[MCPConnection] = []
        self._started = False
        if self.config.enable_zynnova_tools:
            install_zynnova_tools(self.tools, self.workspace, self.config.allowed_zynnova_roots)

    @classmethod
    def create(
        cls,
        provider_config: ProviderConfig,
        workspace: str | Workspace | None = None,
        *,
        config: AgentConfig | None = None,
    ) -> "Agent":
        ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
        return cls(create_provider(provider_config), ws, config=config)

    async def start(self) -> None:
        if self._started: return
        if self.config.auto_load_skills: self.skills.discover()
        for server in self.config.mcp_servers:
            connection = await MCPConnection.connect(server)
            await connection.install_tools(self.tools)
            self._mcp.append(connection)
        self._started = True

    async def run(
        self, prompt: str | list[dict[str, object]], *, session_id: str | None = None, reset: bool = False
    ) -> AgentResult:
        await self.start()
        session = session_id or uuid.uuid4().hex
        history = [] if reset or not self.config.session_memory else self.memory.load(session)
        if not history:
            system = self.config.system_prompt + (self.skills.prompt_fragment() if self.config.auto_load_skills else "")
            history.append(Message("system", system))
        history.append(Message("user", prompt))
        usage: Counter[str] = Counter()
        for step in range(1, self.config.max_steps + 1):
            response = await self.provider.complete(history, self.tools.specs())
            usage.update(response.usage)
            assistant = Message("assistant", response.text or None, tool_calls=response.tool_calls)
            history.append(assistant)
            if not response.tool_calls:
                if self.config.session_memory: self.memory.save(session, history)
                return AgentResult(response.text, tuple(history), step, session, dict(usage))
            for call in response.tool_calls:
                try:
                    value = await self.tools.execute(call.name, call.arguments)
                    content = json.dumps(value, ensure_ascii=False, default=str)
                except Exception as exc:
                    content = json.dumps({"error":type(exc).__name__,"message":str(exc)}, ensure_ascii=False)
                if len(content) > self.config.max_tool_output_chars:
                    content = content[:self.config.max_tool_output_chars] + "…<truncated>"
                history.append(Message("tool", content, name=call.name, tool_call_id=call.id))
        if self.config.session_memory: self.memory.save(session, history)
        raise RuntimeError(f"agent exceeded max_steps={self.config.max_steps} without a final answer")

    async def run_many(
        self, prompts: list[str], *, max_concurrency: int = 4
    ) -> tuple[AgentResult, ...]:
        """Run independent sessions concurrently for ensemble/delegated workloads."""
        import asyncio
        semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        async def one(value: str) -> AgentResult:
            async with semaphore:
                return await self.run(value)
        return tuple(await asyncio.gather(*(one(value) for value in prompts)))

    async def aclose(self) -> None:
        for connection in reversed(self._mcp):
            await connection.aclose()
        self._mcp.clear()
        await self.provider.aclose()
        self._started = False

    async def __aenter__(self) -> "Agent":
        await self.start(); return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


__all__ = ["Agent"]
