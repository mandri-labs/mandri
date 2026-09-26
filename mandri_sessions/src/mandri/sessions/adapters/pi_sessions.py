import json
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from mandri.core.ids import (
    EpochMs,
    HarnessKind,
    HarnessSessionId,
    ProjectPath,
    SessionId,
    SessionState,
    SessionTitle,
)
from mandri.core.ports.sessions import (
    CheckSessionExistsPort,
    DeleteSessionPort,
    FetchSessionsPort,
    RenameSessionPort,
)
from mandri.core.types.availability import SessionOwner
from mandri.core.types.model_selection import ModelSource
from mandri.core.types.sessions import Session
from mandri.sessions.errors import SessionDeleteError, SessionNotFoundError, SessionRenameError
from mandri.sessions.ownership.service import inspect_owner
from mandri.sessions.pi_store import PiSessionInfo, PiSessionStore


class PiSessionsAdapter(
    FetchSessionsPort, RenameSessionPort, DeleteSessionPort, CheckSessionExistsPort
):
    def __init__(self, sessions_root: Path | None = None, *, store: PiSessionStore | None = None):
        self.store = store or PiSessionStore(sessions_root)

    def fetch(self) -> list[Session]:
        infos = self.store.fetch()
        by_path = {str(info.path.resolve()): info.native_id for info in infos}
        synced = EpochMs(int(time.time() * 1000))
        sessions = []
        for info in infos:
            title = info.name or info.first_message
            parent = (
                by_path.get(str(Path(info.parent_path).resolve())) if info.parent_path else None
            )
            sessions.append(
                Session(
                    id=SessionId(info.native_id),
                    harness=HarnessKind.PI,
                    native_id=HarnessSessionId(info.native_id),
                    native_title=SessionTitle(title) if title else None,
                    title_overlay=None,
                    project_path=ProjectPath(info.cwd),
                    created_at=EpochMs(info.created_at),
                    updated_at=EpochMs(info.updated_at),
                    state=SessionState.DISCOVERED,
                    model=info.model,
                    gateway_route_id=None,
                    deleted=False,
                    last_synced_at=synced,
                    reasoning_effort=info.thinking_level,
                    model_source=ModelSource.NATIVE,
                    parent_native_id=HarnessSessionId(parent) if parent else None,
                )
            )
        return sessions

    def exists(self, session_id: SessionId) -> bool:
        return self.store.resolve(str(session_id)) is not None

    def _session(self, session_id: SessionId) -> PiSessionInfo:
        for info in self.store.fetch():
            if info.native_id == str(session_id):
                return info
        raise SessionNotFoundError(f"Pi session not found: {session_id}")

    def rename(self, session_id: SessionId, title: SessionTitle) -> None:
        info = self._session(session_id)
        ownership = inspect_owner(HarnessKind.PI, info.native_id, info.cwd)
        if ownership.owner is not SessionOwner.UNOWNED:
            raise SessionRenameError("Pi session may have an active native writer")
        if info.version < 2:
            raise SessionRenameError("Legacy Pi sessions must be migrated by Pi before renaming")
        entry = {
            "type": "session_info",
            "id": uuid4().hex[:8],
            "parentId": info.leaf_id,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "name": " ".join(str(title).replace("\r", "\n").splitlines()).strip(),
        }
        try:
            with info.path.open("r+b") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                if size:
                    handle.seek(-1, 2)
                    final = handle.read(1)
                    if final != b"\n":
                        handle.write(b"\n")
                handle.write(json.dumps(entry, ensure_ascii=False).encode("utf-8") + b"\n")
        except OSError as error:
            raise SessionRenameError(f"Cannot rename Pi session: {session_id}") from error

    def delete(self, session_id: SessionId) -> None:
        info = self._session(session_id)
        ownership = inspect_owner(HarnessKind.PI, info.native_id, info.cwd)
        if ownership.owner is not SessionOwner.UNOWNED:
            raise SessionDeleteError("Pi session may have an active native writer")
        try:
            info.path.unlink()
        except OSError as error:
            raise SessionDeleteError(f"Cannot delete Pi session: {session_id}") from error
