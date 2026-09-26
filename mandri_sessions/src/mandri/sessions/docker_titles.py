from pathlib import Path

from claude_agent_sdk._internal.sessions import (
    _parse_session_info_from_lite,
    _read_session_lite,
)
from mandri.core.ids import HarnessKind, SessionTitle
from mandri.core.types.sessions import Session
from mandri.sessions.adapters.agy_sessions import AgySessionsAdapter
from mandri.sessions.adapters.codex_fetch_sessions import CodexFetchSessions
from mandri.sessions.adapters.opencode_fetch_sessions import OpencodeSqliteFetchSessionsAdapter
from mandri.sessions.adapters.pi_sessions import PiSessionsAdapter
from mandri.sessions.agents.docker import _check_tree
from mandri.sessions.agents.docker_agy import check_discovery_state
from mandri.sessions.execution_context import DockerSessionContext
from mandri.sessions.transcripts.docker_readers import DockerClaudeReader, DockerPiReader


def _database(context: DockerSessionContext, path: Path) -> Path:
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = path.with_name(path.name + suffix)
        if candidate.exists() or candidate.is_symlink():
            context.state_path(candidate)
    return path


def docker_title(session: Session) -> SessionTitle | None:
    context = DockerSessionContext.from_session(session)
    if session.harness is HarnessKind.CLAUDE:
        reader = DockerClaudeReader(context)
        lite = _read_session_lite(reader._resolve(context.reference(session)))
        info = _parse_session_info_from_lite(str(session.native_id), lite) if lite else None
        title = (info.custom_title or info.summary) if info else None
        return SessionTitle(title) if title else None
    if session.harness is HarnessKind.CODEX:
        home = context.state_root / ".codex"
        _check_tree(context, home / "sessions")
        database = _database(context, home / "state_5.sqlite")
        rows = CodexFetchSessions(database, home / "sessions").fetch()
    elif session.harness is HarnessKind.OPENCODE:
        database = _database(context, context.state_root / ".local/share/opencode/opencode.db")
        rows = OpencodeSqliteFetchSessionsAdapter(database).fetch()
    elif session.harness is HarnessKind.PI:
        rows = PiSessionsAdapter(store=DockerPiReader(context).store).fetch()
    else:
        home = context.state_root / ".gemini"
        check_discovery_state(context, home)
        _check_tree(context, home / "antigravity-cli/conversations")
        rows = AgySessionsAdapter(home).fetch()
    return next((row.native_title for row in rows if row.native_id == session.native_id), None)
