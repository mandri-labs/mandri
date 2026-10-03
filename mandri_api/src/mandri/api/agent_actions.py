from typing import Any

from mandri.core.protocol.agents import (
    AgentCreateParams,
    AgentHistoryParams,
    AgentListParams,
    AgentMessageParams,
    AgentStopParams,
    AgentView,
)
from mandri.core.protocol.errors import ProtocolError, ProtocolErrorCode
from mandri.core.protocol.registry import ActionRegistry
from mandri.core.types.sessions import SessionError
from mandri.core.work_content import contains_completed_content
from mandri.runtime.agents import AgentService
from mandri.runtime.control.agents.base import UnsupportedAgentOperation
from mandri.runtime.control.errors import ControlError
from mandri.sessions.service import SessionsService
from mandri.sessions.transcripts.errors import PageTokenInvalidError
from pydantic import BaseModel


def register_agent_actions(
    registry: ActionRegistry, agents: AgentService, sessions: SessionsService | None = None
) -> None:
    async def handle(params: BaseModel) -> dict[str, Any]:
        try:
            return await _dispatch(agents, params, sessions)
        except UnsupportedAgentOperation as error:
            raise ProtocolError(ProtocolErrorCode.AGENT_UNSUPPORTED, str(error)) from error
        except ControlError as error:
            raise ProtocolError(ProtocolErrorCode.CONTROL_DELIVERY_FAILED, str(error)) from error
        except PageTokenInvalidError as error:
            raise ProtocolError(
                ProtocolErrorCode.INVALID_PARAMS, "History cursor expired"
            ) from error
        except (SessionError, OSError) as error:
            raise ProtocolError(
                ProtocolErrorCode.HARNESS_STORE_UNAVAILABLE, "Agent history is unavailable"
            ) from error

    for action in ("agent.list", "agent.history", "agent.create", "agent.message", "agent.stop"):
        registry.register(action, handle)


async def _dispatch(
    agents: AgentService, params: BaseModel, sessions: SessionsService | None = None
) -> dict[str, Any]:
    statuses = sessions.statuses if sessions is not None else None
    if statuses is not None:
        await statuses.flush()
    if isinstance(params, AgentListParams):
        async with agents.history_store.cache_lock:
            rows, capabilities = await agents.list(
                str(params.session_id) if params.session_id else None
            )
            classified = await agents.history_store.classified()
        views = []
        for agent in rows:
            view = AgentView.model_validate(agent)
            if statuses is not None:
                agent_target = (
                    f"session:{agent.session_id}" if agent.session_id else f"agent:{agent.id}"
                )
                view.status = statuses.get(agent_target)
            views.append(view.model_dump(mode="json"))
        return {
            "agents": views,
            "parent_capabilities": capabilities,
            "classified_session_ids": classified,
        }
    if isinstance(params, AgentHistoryParams):
        completion_revision = None
        status = None
        target = None
        if statuses is not None and params.cursor is None:
            agent = await agents.history_store.get(params.agent_id)
            target = f"session:{agent.session_id}" if agent.session_id else f"agent:{agent.id}"
            status = statuses.get(target)
            if status.work_state == "idle":
                completion_revision = status.completion_revision
        page = await agents.history(params.agent_id, params.cursor, params.limit)
        if (
            completion_revision is not None
            and status is not None
            and not contains_completed_content(
                agent.harness, list(page.entries), status.completion_content_key
            )
        ):
            completion_revision = None
        return {
            "entries": page.entries,
            "next_cursor": page.next_token,
            "has_more": page.has_more,
            "completion_revision": completion_revision,
            "completion_target": target,
        }
    if isinstance(params, AgentCreateParams):
        agent = await agents.create(str(params.session_id), params.content, params.title)
        return {"agent": AgentView.model_validate(agent).model_dump(mode="json")}
    if isinstance(params, AgentMessageParams):
        await agents.message(params.agent_id, params.content)
        return {"agent_id": params.agent_id, "accepted": True}
    if isinstance(params, AgentStopParams):
        return {"agent_id": params.agent_id, "stopped": await agents.stop(params.agent_id)}
    raise ProtocolError(ProtocolErrorCode.INVALID_PARAMS, "Invalid agent parameters")
