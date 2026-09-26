from typing import Any

from mandri.core.types.agents import Agent, AgentCapabilities
from mandri.runtime.control.agents.base import TaskStop, UnsupportedAgentOperation


class ClaudeAgentControl:
    can_create = False

    def __init__(self, stop_task: TaskStop) -> None:
        self._stop_task = stop_task

    def capabilities(self, agent: Agent) -> AgentCapabilities:
        return AgentCapabilities(stop=agent.task_id is not None)

    async def create(self, content: str, title: str | None) -> dict[str, Any]:
        raise UnsupportedAgentOperation("Claude does not expose direct agent creation")

    async def message(self, agent: Agent, content: str) -> None:
        raise UnsupportedAgentOperation("Claude does not expose direct agent messages")

    async def stop(self, agent: Agent) -> bool:
        if agent.task_id is None:
            raise UnsupportedAgentOperation("The agent has no controllable task identity")
        return await self._stop_task(agent.task_id)
