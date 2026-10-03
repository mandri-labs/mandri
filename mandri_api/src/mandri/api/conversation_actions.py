from typing import Any

from mandri.core.ids import SessionId
from mandri.core.protocol.errors import ProtocolError, ProtocolErrorCode
from mandri.core.protocol.registry import ActionRegistry, ConversationReadParams
from mandri.core.types.conversation_status import StatusRevisionError
from mandri.core.types.sessions import SessionError
from mandri.runtime.agents import AgentService
from mandri.sessions.service import SessionsService
from pydantic import BaseModel


def register_conversation_actions(
    registry: ActionRegistry, sessions: SessionsService, agents: AgentService | None
) -> None:
    async def read(params: BaseModel) -> dict[str, Any]:
        if not isinstance(params, ConversationReadParams) or sessions.statuses is None:
            raise ProtocolError(ProtocolErrorCode.INVALID_PARAMS, "Conversation status unavailable")
        target = params.target
        kind, identifier = target.split(":", 1)
        try:
            if kind == "session":
                await sessions.get_session(SessionId(identifier))
            elif agents is not None:
                agent = await agents.history_store.get(identifier)
                if agent.session_id:
                    await sessions.statuses.link(target, f"session:{agent.session_id}")
                    target = f"session:{agent.session_id}"
            else:
                raise ValueError("Agent status unavailable")
            status = await sessions.statuses.acknowledge(
                target, params.through_revision, params.completion_key
            )
        except (SessionError, StatusRevisionError, ValueError) as error:
            raise ProtocolError(ProtocolErrorCode.INVALID_PARAMS, str(error)) from error
        return status.model_dump(mode="json")

    registry.register("conversation.read", read)
