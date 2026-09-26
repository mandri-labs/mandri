import uuid

from mandri.core.ids import HarnessKind
from mandri.core.types.agents import Agent, NativeAgent
from mandri.core.types.sessions import Session

_NAMESPACE = uuid.UUID("2b913cd0-d05c-42a6-9fd2-6eb9a9d29097")


def agent_id(parent_session_id: str, native_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{parent_session_id}:{native_id}"))


def resolve_agents(native_agents: list[NativeAgent], sessions: list[Session]) -> list[Agent]:
    session_map = {(session.harness, str(session.native_id)): session for session in sessions}
    native_map: dict[tuple[HarnessKind, str], NativeAgent] = {
        (agent.harness, agent.native_id): agent
        for agent in native_agents
        if agent.harness is not HarnessKind.CLAUDE
    }
    result = []
    seen: set[str] = set()
    for native in native_agents:
        root = _root(native, native_map, session_map)
        if root is None:
            continue
        if native.harness is HarnessKind.CLAUDE and not _valid_claude_chain(native, native_agents):
            continue
        root_id = str(root.id)
        identity = agent_id(root_id, native.native_id)
        if identity in seen:
            continue
        seen.add(identity)
        parent_native = native.parent_agent_native_id
        if native.harness is not HarnessKind.CLAUDE and native.parent_native_id != root.native_id:
            parent_native = native.parent_native_id
        child = session_map.get((native.harness, native.native_id))
        result.append(
            Agent(
                id=identity,
                parent_session_id=root_id,
                parent_agent_id=agent_id(root_id, parent_native) if parent_native else None,
                session_id=str(child.id)
                if child and native.harness is not HarnessKind.CLAUDE
                else None,
                harness=native.harness,
                native_id=native.native_id,
                title=native.title,
                state=native.state,
                created_at=native.created_at,
                updated_at=native.updated_at,
                delegation_id=native.delegation_id,
                task_id=native.task_id,
                transcript_path=native.transcript_path,
            )
        )
    return result


def _valid_claude_chain(native: NativeAgent, agents: list[NativeAgent]) -> bool:
    children = {
        agent.native_id: agent
        for agent in agents
        if agent.harness is HarnessKind.CLAUDE and agent.parent_native_id == native.parent_native_id
    }
    visited = {native.native_id}
    parent = native.parent_agent_native_id
    while parent:
        if parent in visited or parent not in children:
            return False
        visited.add(parent)
        parent = children[parent].parent_agent_native_id
    return True


def _root(
    native: NativeAgent,
    native_map: dict[tuple[HarnessKind, str], NativeAgent],
    sessions: dict[tuple[HarnessKind, str], Session],
) -> Session | None:
    parent = native.parent_native_id
    visited = {native.native_id}
    while native.harness is not HarnessKind.CLAUDE:
        if parent in visited:
            return None
        visited.add(parent)
        ancestor = native_map.get((native.harness, parent))
        if ancestor is None:
            break
        parent = ancestor.parent_native_id
    return sessions.get((native.harness, parent))
