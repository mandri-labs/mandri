import shutil
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from mandri.core.ids import (
    EpochMs,
    HarnessKind,
    HarnessSessionId,
    ProjectPath,
    SessionId,
    SessionState,
    SessionTitle,
)
from mandri.core.ports.sessions import CheckSessionExistsPort, DeleteSessionPort, FetchSessionsPort
from mandri.core.types.availability import SessionOwner
from mandri.core.types.model_selection import ModelSource
from mandri.core.types.sessions import Session
from mandri.sessions.agy_lease import AgyConversationLease
from mandri.sessions.agy_profiles import read_agy_json, validate_agy_id, write_agy_json
from mandri.sessions.agy_store import (
    agy_metadata,
    agy_roots,
    epoch_ms,
    transcript_head,
    transcript_path,
    workspace_path,
)
from mandri.sessions.errors import SessionDeleteError, SessionNotFoundError, SessionRunningError
from mandri.sessions.ownership.service import inspect_owner


class AgySessionsAdapter(FetchSessionsPort, DeleteSessionPort, CheckSessionExistsPort):
    def __init__(self, root: Path, profiles_root: Path | None = None) -> None:
        self.root = root
        self.profiles_root = profiles_root

    def fetch(self) -> list[Session]:
        metadata = agy_metadata(self.root, self.profiles_root)
        inventory: dict[str, Path] = {}
        scanned_conversations: set[Path] = set()
        scanned_brains: set[Path] = set()
        for root in agy_roots(self.root, self.profiles_root):
            store = root / "antigravity-cli"
            conversations = (store / "conversations").resolve()
            if conversations not in scanned_conversations:
                scanned_conversations.add(conversations)
                for path in conversations.glob("*.db"):
                    inventory.setdefault(path.stem, root)
            brain = (store / "brain").resolve()
            if brain in scanned_brains:
                continue
            scanned_brains.add(brain)
            for path in brain.glob("*"):
                try:
                    validate_agy_id(path.name)
                except ValueError:
                    continue
                if path.is_dir() and transcript_path(root, path.name) is not None:
                    inventory.setdefault(path.name, root)
        sessions = []
        for native_id, root in inventory.items():
            try:
                validate_agy_id(native_id)
            except ValueError:
                continue
            row = metadata.get(native_id, {})
            transcript = transcript_path(root, native_id)
            head = transcript_head(transcript)
            database = root / "antigravity-cli/conversations" / f"{native_id}.db"
            try:
                stat = (transcript or database).stat()
            except FileNotFoundError:
                continue
            title = row.get("title") or row.get("preview") or head.get("title")
            updated = max(epoch_ms(row.get("last_modified_time")), int(stat.st_mtime * 1000))
            created = epoch_ms(row.get("created_at") or head.get("created_at")) or updated
            model = row.get("model") or row.get("modelName")
            sessions.append(
                Session(
                    id=SessionId(native_id),
                    harness=HarnessKind.AGY,
                    native_id=HarnessSessionId(native_id),
                    native_title=SessionTitle(str(title)) if title else None,
                    title_overlay=None,
                    project_path=ProjectPath(workspace_path(row)),
                    created_at=EpochMs(created),
                    updated_at=EpochMs(updated),
                    state=SessionState.DISCOVERED,
                    model=str(model) if model else None,
                    gateway_route_id=None,
                    deleted=False,
                    last_synced_at=EpochMs(int(time.time() * 1000)),
                    parent_native_id=HarnessSessionId(str(row["parent_conversation_id"]))
                    if row.get("parent_conversation_id")
                    else None,
                    model_source=ModelSource.GATEWAY
                    if row.get("model_source") == "gateway"
                    else ModelSource.NATIVE,
                )
            )
        return sorted(sessions, key=lambda session: (session.updated_at, session.id), reverse=True)

    def exists(self, session_id: SessionId) -> bool:
        native_id = validate_agy_id(str(session_id))
        return any(
            (root / "antigravity-cli/conversations" / f"{native_id}.db").is_file()
            for root in agy_roots(self.root, self.profiles_root)
        )

    def delete(self, session_id: SessionId) -> None:
        native_id = validate_agy_id(str(session_id))
        session = next((item for item in self.fetch() if item.native_id == native_id), None)
        if session is None:
            raise SessionNotFoundError(f"Antigravity conversation not found: {native_id}")
        lease = AgyConversationLease(self.root, native_id)
        try:
            lease.acquire()
        except SessionRunningError as error:
            raise SessionDeleteError(
                "Antigravity conversation has an active Mandri writer"
            ) from error
        try:
            self._purge(native_id, str(session.project_path))
        finally:
            lease.release()

    def _purge(self, native_id: str, project_path: str) -> None:
        ownership = inspect_owner(HarnessKind.AGY, native_id, project_path)
        if ownership.owner is not SessionOwner.UNOWNED:
            raise SessionDeleteError("Antigravity conversation may have an active native writer")
        try:
            roots = agy_roots(self.root, self.profiles_root)
            for root in roots:
                self._validate_delete(root, native_id)
            for root in roots:
                self._delete_files(root, native_id)
                self._delete_index(root, native_id)
        except (OSError, sqlite3.Error, ValueError) as error:
            raise SessionDeleteError(
                f"Cannot purge Antigravity conversation: {native_id}"
            ) from error

    @staticmethod
    def _validate_delete(root: Path, native_id: str) -> None:
        store = root / "antigravity-cli"
        conversations = (store / "conversations").resolve()
        for suffix in (".db", ".db-wal", ".db-shm"):
            path = conversations / (native_id + suffix)
            if path.is_symlink() or path.resolve().parent != conversations:
                raise ValueError("Antigravity conversation file escapes native storage")
        brain = (store / "brain").resolve()
        directory = brain / native_id
        if directory.is_symlink() or directory.is_junction() or directory.resolve().parent != brain:
            raise ValueError("Antigravity conversation artifacts escape native storage")

    @staticmethod
    def _delete_files(root: Path, native_id: str) -> None:
        store = root / "antigravity-cli"
        for suffix in (".db", ".db-wal", ".db-shm"):
            (store / "conversations" / (native_id + suffix)).unlink(missing_ok=True)
        directory = store / "brain" / native_id
        if directory.exists():
            shutil.rmtree(directory)

    @staticmethod
    def _delete_index(root: Path, native_id: str) -> None:
        store = root / "antigravity-cli"
        database = store / "conversation_summaries.db"
        if database.is_file():
            with closing(sqlite3.connect(database)) as db:
                tables = db.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
                with db:
                    for row in tables:
                        table = '"' + row[0].replace('"', '""') + '"'
                        columns = {item[1] for item in db.execute(f"PRAGMA table_info({table})")}
                        if "conversation_id" in columns:
                            db.execute(
                                f"DELETE FROM {table} WHERE conversation_id = ?", (native_id,)
                            )
        for name in ("conversation_metadata.json", "last_conversations.json"):
            path = store / "cache" / name
            if path.is_file():
                data = read_agy_json(path)
                write_agy_json(
                    path,
                    {
                        key: value
                        for key, value in data.items()
                        if key != native_id and value != native_id
                    },
                )
