"""MCP client bridge with lazy imports so MCP stays optional."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from ..config import MCPServerConfig
from ..tools.registry import ToolRegistry


@dataclass(slots=True)
class MCPConnection:
    config: MCPServerConfig
    stack: contextlib.AsyncExitStack
    session: Any

    @classmethod
    async def connect(cls, config: MCPServerConfig) -> "MCPConnection":
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError("install zynnova[llm-mcp] to connect MCP servers") from exc
        stack = contextlib.AsyncExitStack()
        try:
            if config.transport == "stdio":
                if not config.command: raise ValueError(f"MCP stdio server {config.name!r} requires command")
                params = StdioServerParameters(command=config.command, args=list(config.args), env=dict(config.env) or None)
                read, write = await stack.enter_async_context(stdio_client(params))
            elif config.transport in {"streamable-http", "http"}:
                if not config.url: raise ValueError(f"MCP HTTP server {config.name!r} requires url")
                try:
                    from mcp.client.streamable_http import streamablehttp_client
                except ImportError as exc:
                    raise RuntimeError("installed MCP SDK does not expose streamable HTTP transport") from exc
                transport = await stack.enter_async_context(streamablehttp_client(config.url))
                read, write = transport[0], transport[1]
            else:
                raise ValueError(f"unsupported MCP transport: {config.transport}")
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            return cls(config, stack, session)
        except Exception:
            await stack.aclose(); raise

    async def install_tools(self, registry: ToolRegistry) -> tuple[str, ...]:
        result = await self.session.list_tools()
        names=[]
        for tool in result.tools:
            local_name = f"mcp__{self.config.name}__{tool.name}".replace("-", "_")
            remote_name = tool.name
            async def invoke(_remote_name=remote_name, **kwargs: Any):
                response = await self.session.call_tool(_remote_name, kwargs)
                payload=[]
                for item in getattr(response, "content", []) or []:
                    payload.append(getattr(item, "text", None) or str(item))
                return {"is_error": bool(getattr(response,"isError",False)), "content": payload}
            registry.add(
                local_name, invoke, description=f"MCP {self.config.name}: {tool.description or tool.name}",
                parameters=tool.inputSchema or {"type":"object"},
            )
            names.append(local_name)
        return tuple(names)

    async def aclose(self) -> None:
        await self.stack.aclose()


__all__ = ["MCPConnection"]
