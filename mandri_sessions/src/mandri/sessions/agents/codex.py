import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from mandri.core.ids import HarnessKind
from mandri.core.ports.agents import AgentDiscoveryResult
from mandri.core.types.agents import NativeAgent
from mandri.core.types.sessions import Session
from mandri.sessions.errors import DatabaseAccessError


def _spawn_field(source: object, field: str) -> str | None:
    if isinstance(source, str):
        try:
            source = json.loads(source)
        except ValueError:
            return None
    if not isinstance(source, dict):
        return None
    agent = source.get("subagent", source.get("sub_agent", source.get("subAgent")))
    spawn = agent.get("thread_spawn") if isinstance(agent, dict) else None
    parent = spawn.get(field) if isinstance(spawn, dict) else None
    return parent if isinstance(parent, str) and parent else None


def parent_of(source: object) -> str | None:
    return _spawn_field(source, "parent_thread_id")


class CodexAgentDiscovery:
    def __init__(self, codex_home: Path) -> None:
        self._home = codex_home

    def discover(self, sessions: list[Session]) -> AgentDiscoveryResult:
        if not any(session.harness is HarnessKind.CODEX for session in sessions):
            return AgentDiscoveryResult(HarnessKind.CODEX, [], frozenset())
        result, found = self._from_database()
        classified = set(found)
        complete = True
        for path in (self._home / "sessions").rglob("rollout-*.jsonl"):
            if path.stem[-36:] in found:
                continue
            try:
                metadata = _metadata(path)
                stat = path.stat()
            except (DatabaseAccessError, OSError):
                complete = False
                continue
            native_id = metadata.get("id")
            if not isinstance(native_id, str):
                complete = False
                continue
            classified.add(native_id)
            parent = parent_of(metadata.get("source"))
            if parent is None or native_id in found:
                continue
            result.append(
                NativeAgent(
                    HarnessKind.CODEX,
                    native_id,
                    parent,
                    str(metadata.get("agent_nickname") or "Codex agent"),
                    int(stat.st_ctime * 1000),
                    int(stat.st_mtime * 1000),
                    transcript_path=str(path),
                    task_id=_spawn_field(metadata.get("source"), "agent_path"),
                )
            )
            found.add(native_id)
        return AgentDiscoveryResult(HarnessKind.CODEX, result, frozenset(classified), complete)

    def _from_database(self) -> tuple[list[NativeAgent], set[str]]:
        path = self._home / "state_5.sqlite"
        if not path.is_file():
            return [], set()
        try:
            with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
                db.row_factory = sqlite3.Row
                columns = {row[1] for row in db.execute("PRAGMA table_info(threads)")}
                if "source" not in columns:
                    return [], set()
                rows = db.execute("SELECT * FROM threads").fetchall()
        except sqlite3.Error as error:
            raise DatabaseAccessError("Cannot read Codex agent relationships") from error
        agents = []
        for row in rows:
            values = dict(row)
            parent = parent_of(values.get("source"))
            if parent is None or values.get("archived"):
                continue
            agents.append(
                NativeAgent(
                    HarnessKind.CODEX,
                    str(values["id"]),
                    parent,
                    str(values.get("agent_nickname") or values.get("title") or "Codex agent"),
                    int(values.get("created_at_ms") or (values.get("created_at") or 0) * 1000),
                    int(values.get("updated_at_ms") or (values.get("updated_at") or 0) * 1000),
                    transcript_path=values.get("rollout_path"),
                    task_id=_spawn_field(values.get("source"), "agent_path"),
                )
            )
        return agents, {str(row["id"]) for row in rows}


def _metadata(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            line = handle.readline(256 * 1024)
        record = json.loads(line)
    except (OSError, ValueError) as error:
        raise DatabaseAccessError("Cannot read Codex agent metadata") from error
    payload = record.get("payload") if isinstance(record, dict) else None
    return payload if isinstance(payload, dict) and record.get("type") == "session_meta" else {}
