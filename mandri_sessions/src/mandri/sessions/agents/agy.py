from pathlib import Path

from mandri.core.ids import HarnessKind
from mandri.core.ports.agents import AgentDiscoveryResult
from mandri.core.types.agents import AgentState, NativeAgent
from mandri.core.types.sessions import Session
from mandri.sessions.agy_store import (
    agy_metadata,
    agy_roots,
    agy_summary_rows,
    epoch_ms,
    transcript_path,
)
from mandri.sessions.errors import DatabaseAccessError


class AgyAgentDiscovery:
    def __init__(self, root: Path, profiles_root: Path | None = None) -> None:
        self.root = root
        self.profiles_root = profiles_root

    def discover(self, sessions: list[Session]) -> AgentDiscoveryResult:
        agents = []
        classified: set[str] = set()
        complete = True
        metadata = agy_metadata(self.root, self.profiles_root)
        classified.update(
            native_id for native_id, row in metadata.items() if row.get("is_mandri_root") is True
        )
        for root in agy_roots(self.root, self.profiles_root):
            try:
                classified.update(str(row["conversation_id"]) for row in agy_summary_rows(root))
            except DatabaseAccessError:
                complete = False
        known = {
            str(session.native_id) for session in sessions if session.harness is HarnessKind.AGY
        }
        complete = complete and known <= classified
        for native_id, row in metadata.items():
            parent = row.get("parent_conversation_id")
            if not isinstance(parent, str) or not parent or parent == native_id:
                continue
            path = transcript_path(self.root, native_id)
            state = AgentState.UNKNOWN
            if row.get("killed"):
                state = AgentState.STOPPED
            elif row.get("not_fully_idle"):
                state = AgentState.RUNNING
            agents.append(
                NativeAgent(
                    harness=HarnessKind.AGY,
                    native_id=native_id,
                    parent_native_id=parent,
                    title=str(row.get("title") or row.get("agent_name") or "Antigravity agent"),
                    created_at=epoch_ms(row.get("created_at")),
                    updated_at=epoch_ms(row.get("last_modified_time")),
                    transcript_path=str(path) if path else None,
                    state=state,
                )
            )
        return AgentDiscoveryResult(HarnessKind.AGY, agents, frozenset(classified), complete)
