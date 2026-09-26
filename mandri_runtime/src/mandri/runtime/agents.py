import dataclasses
from typing import cast

from mandri.core.clock import system_now_ms
from mandri.core.ids import PageToken, SessionId
from mandri.core.ports.transcripts import TranscriptPage
from mandri.core.types.agents import Agent, AgentCapabilities, AgentState
from mandri.runtime.control.agents.base import AgentControl, UnsupportedAgentOperation
from mandri.runtime.service import RuntimeService
from mandri.sessions.agents.relationships import agent_id
from mandri.sessions.agents.service import AgentHistory
from mandri.sessions.service import SessionsService


class AgentService:
    def __init__(
        self, history: AgentHistory, runtime: RuntimeService, sessions: SessionsService
    ) -> None:
        self.history_store = history
        self._runtime = runtime
        self._sessions = sessions

    def _control(self, parent_session_id: str) -> AgentControl | None:
        parent = self._runtime.control_for_session(parent_session_id)
        return cast(AgentControl | None, getattr(parent, "agents", None))

    async def list(
        self, parent_session_id: str | None = None
    ) -> tuple[list[Agent], dict[str, dict[str, bool]]]:
        agents = await self.history_store.list(parent_session_id)
        output = []
        capabilities = {}
        for session in await self._sessions.list_sessions():
            if parent_session_id is not None and str(session.id) != parent_session_id:
                continue
            control = self._control(str(session.id))
            capabilities[str(session.id)] = {"create": control.can_create if control else False}
        for agent in agents:
            control = self._control(agent.parent_session_id)
            output.append(
                dataclasses.replace(
                    agent,
                    capabilities=control.capabilities(agent) if control else AgentCapabilities(),
                )
            )
        return output, capabilities

    async def history(self, identity: str, cursor: str | None, limit: int) -> TranscriptPage:
        return await self.history_store.history(
            identity, PageToken(cursor) if cursor else None, limit
        )

    async def create(self, parent_id: str, content: str, title: str | None) -> Agent:
        parent = await self._sessions.get_session(SessionId(parent_id))
        control = self._control(parent_id)
        if control is None or not control.can_create:
            raise UnsupportedAgentOperation("This harness does not support direct child creation")
        result = await control.create(content, title)
        native_id = str(result["id"])
        now = int(system_now_ms())
        agent = Agent(
            id=agent_id(parent_id, native_id),
            parent_session_id=parent_id,
            harness=parent.harness,
            native_id=native_id,
            title=str(result.get("title") or title or "Agent"),
            state=AgentState.RUNNING,
            created_at=now,
            updated_at=now,
        )
        await self.history_store.save(agent)
        return dataclasses.replace(agent, capabilities=control.capabilities(agent))

    async def message(self, identity: str, content: str) -> None:
        agent, control = await self._require_control(identity, "message")
        await control.message(agent, content)
        await self.history_store.set_state(identity, AgentState.RUNNING)

    async def stop(self, identity: str) -> bool:
        agent, control = await self._require_control(identity, "stop")
        stopped = await control.stop(agent)
        if stopped:
            await self.history_store.set_state(identity, AgentState.STOPPED)
        return stopped

    async def _require_control(self, identity: str, operation: str) -> tuple[Agent, AgentControl]:
        agent = await self.history_store.get(identity)
        control = self._control(agent.parent_session_id)
        if control is None or not getattr(control.capabilities(agent), operation):
            raise UnsupportedAgentOperation("The native agent control channel is unavailable")
        return agent, control
