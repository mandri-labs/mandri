import json
from pathlib import Path
from typing import Any

from mandri.core.ids import HarnessKind
from mandri.core.ports.agents import AgentDiscoveryResult
from mandri.core.types.agents import NativeAgent
from mandri.core.types.sessions import Session


class ClaudeAgentDiscovery:
    def __init__(self, projects_root: Path) -> None:
        self._projects = projects_root

    def discover(self, sessions: list[Session]) -> AgentDiscoveryResult:
        result = []
        for session in sessions:
            if session.harness is not HarnessKind.CLAUDE or session.native_id is None:
                continue
            for parent in self._projects.glob(f"*/{session.native_id}"):
                directory = parent / "subagents"
                for path in directory.rglob("agent-*.jsonl"):
                    agent_id = path.stem.removeprefix("agent-")
                    metadata = _read_metadata(path.with_suffix(".meta.json"))
                    stat = path.stat()
                    result.append(
                        NativeAgent(
                            HarnessKind.CLAUDE,
                            agent_id,
                            str(session.native_id),
                            _text(metadata, "description")
                            or _text(metadata, "name")
                            or "Claude agent",
                            int(stat.st_ctime * 1000),
                            int(stat.st_mtime * 1000),
                            parent_agent_native_id=_text(metadata, "parentAgentId"),
                            delegation_id=_text(metadata, "toolUseId"),
                            task_id=_text(metadata, "taskId"),
                            transcript_path=str(path),
                        )
                    )
        return AgentDiscoveryResult(HarnessKind.CLAUDE, result)


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _text(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) and value else None
