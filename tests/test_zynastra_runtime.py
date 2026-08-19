from __future__ import annotations

from pathlib import Path

import pytest

from zynnova.llm.zynastra import Agent, AgentConfig, Workspace
from zynnova.llm.zynastra.types import ProviderResponse, ToolCall


class FakeProvider:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=()):
        self.calls += 1
        if self.calls == 1:
            return ProviderResponse(
                tool_calls=(ToolCall("call-1", "workspace_path", {"category": "artifacts"}),)
            )
        assert messages[-1].role == "tool"
        assert "artifacts" in str(messages[-1].content)
        return ProviderResponse(text="done", usage={"total_tokens": 7})

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_agent_tool_loop_and_memory(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "external").ensure()
    provider = FakeProvider()
    agent = Agent(provider, workspace, config=AgentConfig(max_steps=4))
    result = await agent.run("find my artifact directory", session_id="test-session")
    assert result.text == "done"
    assert result.steps == 2
    assert result.usage["total_tokens"] == 7
    saved = agent.memory.load("test-session")
    assert [item.role for item in saved][-2:] == ["tool", "assistant"]
    await agent.aclose()


def test_workspace_is_external_and_created(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "workspace").ensure()
    assert ws.models.is_dir()
    assert ws.finetunes.is_dir()
    assert ws.skills.is_dir()
