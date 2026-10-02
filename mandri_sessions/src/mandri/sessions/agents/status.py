from pathlib import Path

from mandri.core.ids import HarnessKind, HarnessSessionId
from mandri.core.ports.transcripts import SessionRef
from mandri.core.types.agents import Agent, AgentState
from mandri.core.types.execution import ExecutionBackend, ProtectionError
from mandri.core.types.sessions import Session, SessionError
from mandri.sessions.agents.docker import contextual_agent
from mandri.sessions.transcripts.resolver import TranscriptResolver
from mandri.sessions.transcripts.status import jsonl_status


def historical_state(
    agent: Agent, readers: TranscriptResolver, parent: Session | None = None
) -> AgentState:
    try:
        if parent is not None and parent.execution_backend is ExecutionBackend.DOCKER:
            agent = contextual_agent(agent, parent)
        if agent.transcript_path:
            path = Path(agent.transcript_path)
            if not path.is_file() or path.stat().st_size == 0:
                return AgentState.UNKNOWN
            busy, _ = jsonl_status(path, agent.harness)
        elif agent.harness is HarnessKind.OPENCODE:
            reader = (
                readers.for_session(parent) if parent is not None else readers.reader(agent.harness)
            )
            status = getattr(reader, "status", None)
            if reader is None or status is None:
                return AgentState.UNKNOWN
            ref = SessionRef(agent.harness, HarnessSessionId(agent.native_id))
            agent_state = getattr(reader, "agent_state", None)
            if agent_state is not None:
                return AgentState(agent_state(ref))
            if not reader.page(ref, None, 1).entries:
                return AgentState.UNKNOWN
            busy, _ = status(ref)
        else:
            return AgentState.UNKNOWN
    except (SessionError, ProtectionError, OSError):
        return AgentState.UNKNOWN
    if busy is None:
        return AgentState.UNKNOWN
    return AgentState.RUNNING if busy else AgentState.COMPLETED
