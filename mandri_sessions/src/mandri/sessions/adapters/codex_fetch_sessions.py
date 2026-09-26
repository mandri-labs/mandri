"""Codex session fetch: SQLite state DB read with rollout JSONL scan fallback."""

import datetime
import json
import re
import sqlite3
from contextlib import closing
from datetime import UTC
from pathlib import Path
from typing import Any

from mandri.core.ids import (
    EpochMs,
    HarnessKind,
    HarnessSessionId,
    ProjectPath,
    SessionId,
    SessionState,
    SessionTitle,
)
from mandri.core.ports.sessions import FetchSessionsPort
from mandri.core.types.sessions import Session
from mandri.sessions.agents.codex import parent_of
from mandri.sessions.codex_source import is_internal_source
from mandri.sessions.errors import DatabaseAccessError, SchemaDriftError

REQUIRED_THREAD_COLUMNS = frozenset(
    {
        "id",
        "cwd",
        "model",
        "title",
        "first_user_message",
        "archived",
        "archived_at",
        "created_at_ms",
        "updated_at_ms",
    }
)

WINDOWS_EXTENDED_PREFIX = "\\\\?\\"
TITLE_MAX_CHARS = 120
ROLLOUT_SCAN_LINE_CAP = 50
SQLITE_TIMEOUT_S = 2.0

_ROLLOUT_NAME = re.compile(
    r"^rollout-(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})-([0-9a-fA-F-]{36})\.jsonl$"
)


def _normalize_title(raw: str | None) -> str | None:
    if not raw:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    first_line = stripped.splitlines()[0].strip()
    return first_line[:TITLE_MAX_CHARS] or None


def _clean_project_path(raw: str | None) -> str:
    text = raw or ""
    if text.startswith(WINDOWS_EXTENDED_PREFIX):
        return text[len(WINDOWS_EXTENDED_PREFIX) :]
    return text


def _build_session(
    native_id: str,
    project_path: str,
    created_at_ms: int,
    updated_at_ms: int,
    title: str | None,
    model: str | None,
    parent_native_id: str | None = None,
) -> Session:
    return Session(
        id=SessionId(native_id),
        harness=HarnessKind.CODEX,
        native_id=HarnessSessionId(native_id),
        native_title=SessionTitle(title) if title else None,
        title_overlay=None,
        project_path=ProjectPath(_clean_project_path(project_path)),
        created_at=EpochMs(created_at_ms),
        updated_at=EpochMs(updated_at_ms),
        state=SessionState.DISCOVERED,
        model=model,
        gateway_route_id=None,
        deleted=False,
        last_synced_at=EpochMs(0),
        parent_native_id=HarnessSessionId(parent_native_id) if parent_native_id else None,
    )


class CodexStateDbFetchAdapter(FetchSessionsPort):
    """Read-only ThreadMetadata listing from the codex state DB."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)

    def fetch(self) -> list[Session]:
        try:
            uri = f"{self._db_path.resolve().as_uri()}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True, timeout=SQLITE_TIMEOUT_S)) as connection:
                self._check_schema(connection)
                columns = {row[1] for row in connection.execute("PRAGMA table_info(threads)")}
                title_column = (
                    "COALESCE(NULLIF(trim(name), ''), title)" if "name" in columns else "title"
                )
                source_column = "source" if "source" in columns else "NULL"
                rows = connection.execute(
                    f"SELECT id, cwd, model, {title_column}, first_user_message, "
                    f"created_at_ms, updated_at_ms, {source_column} FROM threads "
                    "WHERE archived = 0 OR archived IS NULL "
                    "ORDER BY updated_at_ms DESC, id DESC"
                ).fetchall()
        except sqlite3.OperationalError as error:
            raise DatabaseAccessError(f"cannot read codex state db: {error}") from error
        return [self._row_to_session(row) for row in rows if not is_internal_source(row[-1])]

    @staticmethod
    def _check_schema(connection: sqlite3.Connection) -> None:
        columns = connection.execute("PRAGMA table_info(threads)").fetchall()
        names = {str(row[1]) for row in columns}
        missing = REQUIRED_THREAD_COLUMNS - names
        if missing:
            raise SchemaDriftError(f"codex state db missing columns: {sorted(missing)}")

    @staticmethod
    def _row_to_session(row: tuple[Any, ...]) -> Session:
        native_id, cwd, model, title, first_user_message, created_ms, updated_ms, source = row
        title_text = _normalize_title(title) or _normalize_title(first_user_message)
        return _build_session(
            native_id=str(native_id),
            project_path=str(cwd or ""),
            created_at_ms=int(created_ms or 0),
            updated_at_ms=int(updated_ms or 0),
            title=title_text,
            model=str(model) if model else None,
            parent_native_id=parent_of(source),
        )


class CodexRolloutScanAdapter(FetchSessionsPort):
    """List sessions by scanning rollout-*.jsonl filenames and session_meta heads."""

    def __init__(self, sessions_dir: Path) -> None:
        self._sessions_dir = Path(sessions_dir)

    def fetch(self) -> list[Session]:
        if not self._sessions_dir.is_dir():
            return []
        sessions_by_id: dict[str, Session] = {}
        for path in self._sessions_dir.rglob("rollout-*.jsonl"):
            match = _ROLLOUT_NAME.match(path.name)
            if match is None:
                continue
            started_ms = _filename_epoch_ms(match.group(1))
            native_id = match.group(2)
            head = _read_rollout_head(path, native_id)
            if head is None:
                continue
            cwd, user_text, parent_native_id = head
            sessions_by_id.setdefault(
                native_id,
                _build_session(
                    native_id=native_id,
                    project_path=cwd,
                    created_at_ms=started_ms,
                    updated_at_ms=started_ms,
                    title=_normalize_title(user_text),
                    model=None,
                    parent_native_id=parent_native_id,
                ),
            )
        return sorted(
            sessions_by_id.values(),
            key=lambda session: (session.updated_at, str(session.native_id)),
            reverse=True,
        )


class CodexFetchSessions(FetchSessionsPort):
    """Composite fetch port: state DB first, rollout scan on drift or access failure."""

    def __init__(self, state_db_path: Path, rollouts_dir: Path) -> None:
        self._primary = CodexStateDbFetchAdapter(state_db_path)
        self._fallback = CodexRolloutScanAdapter(rollouts_dir)

    def fetch(self) -> list[Session]:
        try:
            return self._primary.fetch()
        except (SchemaDriftError, DatabaseAccessError):
            return self._fallback.fetch()


def _filename_epoch_ms(stamp: str) -> int:
    moment = datetime.datetime.strptime(stamp, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=UTC)
    return int(moment.timestamp() * 1000)


def _read_rollout_head(path: Path, expected_id: str) -> tuple[str, str, str | None] | None:
    with path.open(encoding="utf-8") as handle:
        first_line = handle.readline()
        if not first_line:
            return None
        try:
            record = json.loads(first_line)
        except json.JSONDecodeError:
            return None
        payload = record.get("payload") if isinstance(record, dict) else None
        if (
            not isinstance(record, dict)
            or record.get("type") != "session_meta"
            or not isinstance(payload, dict)
            or payload.get("id") != expected_id
            or is_internal_source(payload.get("source"))
        ):
            return None
        cwd = payload.get("cwd")
        session_cwd = cwd if isinstance(cwd, str) else ""
        for index, line in enumerate(handle):
            if index + 1 >= ROLLOUT_SCAN_LINE_CAP:
                break
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict) or item.get("type") != "response_item":
                continue
            user_text = _user_message_text(item.get("payload"))
            if user_text is not None:
                return session_cwd, user_text, parent_of(payload.get("source"))
        return session_cwd, "", parent_of(payload.get("source"))


def _user_message_text(payload: Any) -> str | None:
    if not isinstance(payload, dict) or payload.get("role") != "user":
        return None
    content = payload.get("content")
    if not isinstance(content, list):
        return None
    parts = [str(item["text"]) for item in content if isinstance(item, dict) and "text" in item]
    text = " ".join(parts).strip()
    if not text or text.startswith("<"):
        return None
    return text
