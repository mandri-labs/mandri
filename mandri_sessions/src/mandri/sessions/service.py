"""Unified session read-model service on top of the sync engine."""

import asyncio
import json
import sys
import typing
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from mandri.core.clock import system_now_ms
from mandri.core.fs.paths import normalize_fs_path
from mandri.core.ids import (
    HarnessKind,
    HarnessSessionId,
    PageToken,
    ProjectPath,
    RouteId,
    SessionId,
    SessionState,
    SessionTitle,
)
from mandri.core.ports.database import DatabasePort
from mandri.core.ports.executions import ExecutionRepositoryPort
from mandri.core.ports.native_sessions import PiSessionIdentityPort
from mandri.core.ports.privacy import PrivacyScopePort
from mandri.core.ports.session_privacy import SessionPrivacyPort
from mandri.core.types.availability import SessionOwner
from mandri.core.types.conversation_status import WorkDelta
from mandri.core.types.execution import (
    ExecutionBackend,
    PrivacyMode,
    ProtectionError,
    SessionPolicy,
)
from mandri.core.types.execution_generation import ExecutionGeneration
from mandri.core.types.model_selection import ModelSource, model_capabilities
from mandri.core.types.sessions import (
    Session,
    SessionStateError,
    can_transition,
)
from mandri.sessions.activity import SessionActivity
from mandri.sessions.conversation_status import ConversationStatuses
from mandri.sessions.errors import (
    SessionConflictError,
    SessionNotFoundError,
    SessionRunningError,
)
from mandri.sessions.execution_context import transcript_reference
from mandri.sessions.fork_source import CodexForkSource, selected_codex_source
from mandri.sessions.lineage import SessionLineage
from mandri.sessions.owned_state import purge_docker_state
from mandri.sessions.ownership.docker import docker_ownership
from mandri.sessions.ownership.service import NativeOwnership, inspect_owner, release_writer
from mandri.sessions.purge_state import retain_purge_state
from mandri.sessions.session_rows import row_to_session as _row_to_session
from mandri.sessions.sync import HarnessState, SessionsBackend
from mandri.sessions.transcripts import (
    HarnessStoreUnavailableError,
    SessionRef,
    TranscriptPage,
    TranscriptResolver,
)
from mandri.sessions.transcripts.record_download import RecordDownload
from mandri.sessions.worktrees import SessionWorktrees

HISTORY_DEFAULT_LIMIT = 100
HISTORY_MAX_LIMIT = 500
MIN_HISTORY_LIMIT = 1


class SyncEnginePort(Protocol):
    async def sync(self) -> None:
        raise NotImplementedError

    def harness_state(self, kind: HarnessKind) -> HarnessState:
        raise NotImplementedError

    def backend(self, kind: HarnessKind) -> SessionsBackend | None:
        raise NotImplementedError

    def activity_of(self, session_id: SessionId) -> SessionActivity | None:
        raise NotImplementedError


class SessionsService:
    def __init__(
        self,
        db: DatabasePort,
        engine: SyncEnginePort,
        transcripts: TranscriptResolver | None = None,
        executions: ExecutionRepositoryPort | None = None,
        privacy_scopes: PrivacyScopePort | None = None,
        worktrees_dir: Path | None = None,
        pi_identities: PiSessionIdentityPort | None = None,
        statuses: ConversationStatuses | None = None,
        session_privacy: SessionPrivacyPort | None = None,
    ) -> None:
        self.worktrees = SessionWorktrees(
            db, worktrees_dir or Path.home() / ".mandri" / "worktrees"
        )
        self._db = db
        self._engine = engine
        self._transcripts = transcripts
        self._executions = executions
        self._privacy_scopes = privacy_scopes
        self._session_privacy = session_privacy
        self._pi_identities = pi_identities
        self.statuses = statuses
        self._pi_processes: Callable[[], Mapping[int, str | None]] = dict

    def set_pi_processes(self, callback: Callable[[], Mapping[int, str | None]]) -> None:
        self._pi_processes = callback

    async def execution_generation(self, session_id: SessionId) -> ExecutionGeneration | None:
        await self.get_session(session_id)
        return await self._executions.latest(str(session_id)) if self._executions else None

    async def record_execution(
        self, session_id: SessionId, owner: str, context: str
    ) -> ExecutionGeneration:
        await self.get_session(session_id)
        if self._executions is None:
            raise SessionConflictError("Execution storage is unavailable")
        return await self._executions.create(str(session_id), owner, context)

    async def sync(self) -> None:
        await self._engine.sync()

    def harness_state(self, kind: HarnessKind) -> HarnessState:
        return self._engine.harness_state(kind)

    def activity_of(self, session_id: SessionId) -> SessionActivity | None:
        return self._engine.activity_of(session_id)

    async def list_sessions(
        self,
        harness: HarnessKind | None = None,
        state: SessionState | None = None,
        project_path: str | None = None,
    ) -> list[Session]:
        sql = "SELECT * FROM session WHERE deleted = 0"
        params: list[object] = []
        if harness is not None:
            sql += " AND harness = ?"
            params.append(harness.value)
        if state is not None:
            sql += " AND state = ?"
            params.append(state.value)
        if project_path is not None:
            sql += " AND (project_path = ? OR json_extract(worktree, '$.source_path') = ?)"
            params.extend([str(normalize_fs_path(project_path))] * 2)
        sql += " ORDER BY updated_at DESC"
        rows = await self._db.fetch_all(sql, tuple(params))
        return [_row_to_session(row) for row in rows]

    async def get_session(self, session_id: SessionId) -> Session:
        row = await self._db.fetch_one(
            "SELECT * FROM session WHERE id = ? AND deleted = 0", (str(session_id),)
        )
        if row is None:
            raise SessionNotFoundError(f"unknown session {session_id}")
        return _row_to_session(row)

    async def ensure_session_policy(self, session_id: SessionId) -> Session:
        session = await self.get_session(session_id)
        if session.parent_native_id is None and session.parent_session_id is None:
            session.policy.validate(session.model_source)
            if session.privacy_mode is PrivacyMode.SURROGATE and not session.privacy_scope_id:
                raise ProtectionError("privacy_state_unavailable", "Privacy state is unavailable")
            return session
        return _row_to_session(await SessionLineage(self._db).ensure(str(session_id)))

    async def set_session_privacy(self, expected: Session, mode: PrivacyMode) -> Session:
        if expected.model_source is not ModelSource.GATEWAY:
            raise ProtectionError(
                "privacy_native_unsupported", "Pseudonymization requires a gateway model"
            )
        if mode is expected.privacy_mode:
            return expected
        if self._session_privacy is None:
            raise ProtectionError("privacy_state_unavailable", "Privacy storage is unavailable")
        scope_id = expected.privacy_scope_id
        created = False
        if mode is PrivacyMode.SURROGATE:
            if self._privacy_scopes is None:
                raise ProtectionError("privacy_state_unavailable", "Privacy state is unavailable")
            if scope_id:
                await self._privacy_scopes.validate(scope_id)
            else:
                scope_id = await self._privacy_scopes.create(str(expected.project_path))
                created = True
        try:
            await self._session_privacy.set_privacy(expected, mode, scope_id)
        except BaseException:
            if created and self._privacy_scopes is not None and scope_id is not None:
                await asyncio.shield(self._privacy_scopes.delete(scope_id))
            raise
        return await self.get_session(expected.id)

    async def ensure_no_pending_purge(self, session_id: SessionId) -> None:
        row = await self._db.fetch_one(
            "SELECT session_id FROM session_purge WHERE session_id = ?", (str(session_id),)
        )
        if row is not None:
            raise ProtectionError("session_purge_pending", "Session purge must finish before reuse")

    async def history(
        self, session_id: SessionId, cursor: PageToken | None, limit: int, *, recent: bool = False
    ) -> TranscriptPage:
        session = await self.get_session(session_id)
        if session.native_id is None and cursor is None:
            return TranscriptPage(entries=[], next_token=None, has_more=False)
        if self._transcripts is None or session.native_id is None:
            raise HarnessStoreUnavailableError(f"no transcript store for session {session_id}")
        reader = self._transcripts.for_session(session)
        if reader is None:
            raise HarnessStoreUnavailableError(
                f"no transcript store for harness {session.harness.value}"
            )
        bounded = max(MIN_HISTORY_LIMIT, min(limit, HISTORY_MAX_LIMIT))
        ref = transcript_reference(session)
        read: Callable[[SessionRef, PageToken | None, int], TranscriptPage] = (
            getattr(reader, "recent", reader.page) if recent else reader.page
        )
        return await asyncio.to_thread(read, ref, cursor, bounded)

    @asynccontextmanager
    async def codex_fork_source(
        self, session_id: SessionId, target: ExecutionBackend
    ) -> AsyncIterator[CodexForkSource]:
        await self.ensure_no_pending_purge(session_id)
        session = await self.ensure_session_policy(session_id)
        owner = await self.native_ownership(session_id)
        if owner.owner is not SessionOwner.UNOWNED:
            raise ProtectionError("session_writer_conflict", "The native session has a writer")
        reader = self._transcripts.for_session(session) if self._transcripts else None
        context = selected_codex_source(session, target, reader)
        opening = asyncio.create_task(asyncio.to_thread(context.__enter__))
        try:
            source = await asyncio.shield(opening)
        except asyncio.CancelledError:
            try:
                await opening
            except Exception:
                pass
            else:
                await asyncio.to_thread(context.__exit__, None, None, None)
            raise
        try:
            if await self.get_session(session_id) != session:
                raise ProtectionError("session_writer_conflict", "The source session changed")
            yield source
        finally:
            await asyncio.shield(asyncio.to_thread(context.__exit__, *sys.exc_info()))

    async def transcript_revision(self, session_id: SessionId) -> object:
        session = await self.get_session(session_id)
        reader = self._transcripts.for_session(session) if self._transcripts else None
        revision = getattr(reader, "revision", None)
        if revision is None or session.native_id is None:
            raise HarnessStoreUnavailableError("transcript observation unavailable")
        ref = transcript_reference(session)
        version = await asyncio.to_thread(revision, ref)
        return version, await self.native_ownership(session_id)

    async def work_delta(
        self, session_id: SessionId, checkpoint: dict[str, typing.Any] | None
    ) -> WorkDelta:
        session = await self.get_session(session_id)
        reader = self._transcripts.for_session(session) if self._transcripts else None
        observe = getattr(reader, "work_delta", None)
        if observe is None or session.native_id is None:
            raise HarnessStoreUnavailableError("Work observation is unavailable")
        result: WorkDelta = await asyncio.to_thread(
            observe, transcript_reference(session), checkpoint
        )
        return result

    async def external_busy(self, session_id: SessionId) -> bool:
        busy, _ = await self.external_status(session_id)
        if busy is None:
            raise HarnessStoreUnavailableError("external session activity is unknown")
        return busy

    async def external_status(self, session_id: SessionId) -> tuple[bool | None, str | None]:
        session = await self.get_session(session_id)
        reader = self._transcripts.for_session(session) if self._transcripts else None
        inspect = getattr(reader, "status", None)
        if inspect is None or session.native_id is None:
            return None, None
        ref = transcript_reference(session)
        busy, model = await asyncio.to_thread(inspect, ref)
        owner = await self.native_ownership(session_id)
        if owner.owner is SessionOwner.UNOWNED and busy is not None:
            busy = False
        elif owner.owner is SessionOwner.UNKNOWN:
            busy = None
        return busy, model

    async def native_ownership(self, session_id: SessionId) -> NativeOwnership:
        session = await self.get_session(session_id)
        if session.execution_backend is ExecutionBackend.DOCKER:
            return await asyncio.to_thread(docker_ownership, session)
        if session.native_id is None:
            return NativeOwnership(SessionOwner.UNKNOWN, reason="native_identity_unavailable")
        reader = self._transcripts.for_session(session) if self._transcripts else None
        lock_path = getattr(reader, "writer_lock_path", lambda _ref: None)(
            transcript_reference(session)
        )
        if session.harness is HarnessKind.PI:
            return await asyncio.to_thread(
                inspect_owner,
                session.harness,
                str(session.native_id),
                str(session.project_path),
                lock_path,
                self._pi_processes(),
            )
        return await asyncio.to_thread(
            inspect_owner,
            session.harness,
            str(session.native_id),
            str(session.project_path),
            lock_path,
        )

    async def release_native_writer(self, session_id: SessionId) -> None:
        session = await self.get_session(session_id)
        if session.execution_backend is ExecutionBackend.DOCKER:
            raise SessionConflictError("Stop the Docker execution through the runtime")
        if session.native_id is None:
            raise SessionConflictError("Native identity is unavailable")
        reader = self._transcripts.for_session(session) if self._transcripts else None
        lock_path = getattr(reader, "writer_lock_path", lambda _ref: None)(
            transcript_reference(session)
        )
        owner = await self.native_ownership(session_id)
        await asyncio.to_thread(
            release_writer,
            session.harness,
            str(session.native_id),
            str(session.project_path),
            lock_path,
            owner,
        )

    async def history_record(self, session_id: SessionId, reference: PageToken) -> RecordDownload:
        session = await self.get_session(session_id)
        reader = self._transcripts.for_session(session) if self._transcripts else None
        read = getattr(reader, "record", None)
        if read is None or session.native_id is None:
            raise HarnessStoreUnavailableError("transcript record download unavailable")
        ref = transcript_reference(session)
        return typing.cast(RecordDownload, await asyncio.to_thread(read, ref, reference))

    async def set_session_model(
        self,
        session_id: SessionId,
        model: str,
        model_source: ModelSource | None = None,
        *,
        expected: Session | None = None,
    ) -> Session:
        if not model.strip():
            raise SessionConflictError("model must be a non-empty string")
        existing = await self._db.fetch_one(
            "SELECT * FROM session WHERE id = ? AND deleted = 0", (str(session_id),)
        )
        if existing is None:
            raise SessionNotFoundError(f"unknown session {session_id}")
        source = model_source or ModelSource(str(existing.get("model_source", "gateway")))
        _row_to_session(existing).policy.validate(source)
        if source not in model_capabilities(str(existing["harness"])).sources:
            raise SessionConflictError(
                f"Harness {existing['harness']!r} does not support {source.value} models"
            )
        reset_effort = source.value != existing.get("model_source", "gateway") or (
            source is ModelSource.NATIVE and model != existing["model"]
        )
        condition = ""
        selection: tuple[object, ...] = ()
        if expected is not None:
            condition = (
                " AND model IS ? AND model_source = ?"
                " AND gateway_route_id IS ? AND reasoning_effort IS ?"
            )
            selection = (
                expected.model,
                expected.model_source.value,
                expected.gateway_route_id,
                expected.reasoning_effort,
            )
        updated = await self._db.fetch_one(
            "UPDATE session SET model = ?, model_source = ?,"
            " reasoning_effort = CASE WHEN ? THEN NULL ELSE reasoning_effort END,"
            " gateway_route_id = CASE WHEN ? = 'native' THEN NULL ELSE gateway_route_id END"
            " WHERE id = ? AND deleted = 0 AND policy_revision = ?"
            + condition + " RETURNING *",
            (model, source.value, reset_effort, source.value, str(session_id),
             existing["policy_revision"], *selection),
        )
        if updated is None:
            raise SessionConflictError("Session model selection changed or policy changed")
        return _row_to_session(updated)

    async def observe_native_selection(
        self, expected: Session, model: str, effort: str | None
    ) -> bool:
        updated = await self._db.fetch_one(
            "UPDATE session SET model = ?, reasoning_effort = ?"
            " WHERE id = ? AND deleted = 0 AND model_source = 'native'"
            " AND native_id IS ? AND model IS ? AND reasoning_effort IS ? RETURNING id",
            (model, effort, str(expected.id), expected.native_id, expected.model,
             expected.reasoning_effort),
        )
        return updated is not None

    async def set_session_effort(self, session_id: SessionId, effort: str | None) -> Session:
        if effort is not None and not effort.strip():
            raise SessionConflictError("effort must be a non-empty string")
        existing = await self._db.fetch_one(
            "SELECT id FROM session WHERE id = ? AND deleted = 0", (str(session_id),)
        )
        if existing is None:
            raise SessionNotFoundError(f"unknown session {session_id}")
        await self._db.execute(
            "UPDATE session SET reasoning_effort = ? WHERE id = ?", (effort, str(session_id))
        )
        updated = await self._db.fetch_one("SELECT * FROM session WHERE id = ?", (str(session_id),))
        if updated is None:
            raise SessionNotFoundError(f"unknown session {session_id}")
        return _row_to_session(updated)

    async def rename_session(self, session_id: SessionId, title: SessionTitle) -> Session:
        existing = await self._db.fetch_one(
            "SELECT id FROM session WHERE id = ? AND deleted = 0", (str(session_id),)
        )
        if existing is None:
            raise SessionNotFoundError(f"unknown session {session_id}")
        await self._db.execute(
            "UPDATE session SET title_overlay = ? WHERE id = ? AND deleted = 0",
            (str(title), str(session_id)),
        )
        updated = await self._db.fetch_one("SELECT * FROM session WHERE id = ?", (str(session_id),))
        if updated is None:
            raise SessionNotFoundError(f"unknown session {session_id}")
        return _row_to_session(updated)

    async def delete_session(
        self, session_id: SessionId, purge: bool = False, *, discard_worktree: bool = False
    ) -> None:
        async with self.worktrees.lease(str(session_id)):
            await self._delete_session(session_id, purge, discard_worktree=discard_worktree)

    async def _delete_session(
        self, session_id: SessionId, purge: bool, *, discard_worktree: bool
    ) -> None:
        existing = await self._db.fetch_one(
            "SELECT * FROM session WHERE id = ?", (str(session_id),)
        )
        if existing is None or (existing["deleted"] and not purge):
            raise SessionNotFoundError(f"unknown session {session_id}")
        if SessionState(str(existing["state"])) is SessionState.LIVE:
            raise SessionRunningError(f"session {session_id} is running; stop it first")
        session = _row_to_session(existing)
        if session.worktree is not None and session.native_id is not None:
            owner = await self.native_ownership(session_id)
            if owner.owner is not SessionOwner.UNOWNED:
                raise SessionRunningError("Stop the worktree session writer before deletion")
        managed = (
            session.execution_backend is ExecutionBackend.DOCKER
            or session.privacy_mode is PrivacyMode.SURROGATE
        )
        if purge and managed:
            if session.privacy_mode is PrivacyMode.SURROGATE and self._privacy_scopes is None:
                raise ProtectionError(
                    "privacy_unavailable", "Purge requires the daemon privacy service"
                )
            session = await retain_purge_state(self._db, session)
        if session.execution_backend is ExecutionBackend.DOCKER:
            if purge:
                if session.execution_context is not None:
                    await asyncio.to_thread(purge_docker_state, session)
            else:
                owner = await asyncio.to_thread(docker_ownership, session)
                if owner.owner is not SessionOwner.UNOWNED:
                    raise SessionRunningError(
                        "Native state is owned or its writer cannot be confirmed"
                    )
        if session.worktree is not None:
            await self.worktrees.remove(session_id, discard=discard_worktree, clear=False)
        await self._db.execute(
            "UPDATE session SET deleted = 1, worktree = NULL WHERE id = ?", (str(session_id),)
        )
        if not purge:
            return
        if session.execution_backend is ExecutionBackend.HOST:
            backend = self._engine.backend(session.harness)
            if backend is not None and session.native_id is not None:
                native_id = SessionId(str(session.native_id))
                await asyncio.to_thread(backend.delete, native_id)
        await self._purge_privacy(session)
        if managed:
            await self._db.execute(
                "DELETE FROM session_purge WHERE session_id = ?", (str(session.id),)
            )

    async def _purge_privacy(self, session: Session) -> None:
        await self._db.execute(
            "UPDATE session SET privacy_scope_id = NULL, execution_context = NULL,"
            " gateway_route_id = NULL, native_id = NULL WHERE id = ? AND deleted = 1",
            (str(session.id),),
        )
        if session.gateway_route_id is not None:
            await self._db.execute(
                "DELETE FROM gateway_route WHERE id = ? AND NOT EXISTS"
                " (SELECT 1 FROM session WHERE gateway_route_id = ?)",
                (str(session.gateway_route_id), str(session.gateway_route_id)),
            )
        if session.privacy_scope_id is not None and self._privacy_scopes is not None:
            try:
                await self._privacy_scopes.delete(session.privacy_scope_id)
            except ProtectionError as error:
                if error.code != "privacy_scope_in_use":
                    raise

    async def set_session_state(self, session_id: SessionId, state: SessionState) -> None:
        row = await self._db.fetch_one(
            "SELECT state FROM session WHERE id = ? AND deleted = 0", (str(session_id),)
        )
        if row is None:
            return
        current = SessionState(str(row["state"]))
        if current is state:
            return
        if not can_transition(current, state):
            raise SessionStateError(
                f"illegal transition {current.value} -> {state.value} for session {session_id}"
            )
        await self._db.execute(
            "UPDATE session SET state = ?, updated_at = ? WHERE id = ? AND deleted = 0",
            (state.value, int(system_now_ms()), str(session_id)),
        )

    async def set_session_route_id(self, session_id: SessionId, route_id: RouteId) -> Session:
        existing = await self._db.fetch_one(
            "SELECT id FROM session WHERE id = ? AND deleted = 0", (str(session_id),)
        )
        if existing is None:
            raise SessionNotFoundError(f"unknown session {session_id}")
        await self._db.execute(
            "UPDATE session SET gateway_route_id = ?, updated_at = ? WHERE id = ? AND deleted = 0",
            (str(route_id), int(system_now_ms()), str(session_id)),
        )
        updated = await self._db.fetch_one("SELECT * FROM session WHERE id = ?", (str(session_id),))
        if updated is None:
            raise SessionNotFoundError(f"unknown session {session_id}")
        return _row_to_session(updated)

    async def set_session_interaction_mode(
        self, session_id: SessionId, mode: str, applied: str
    ) -> None:
        await self._db.execute(
            "UPDATE session SET interaction_mode = ?, updated_at = ? WHERE id = ? AND deleted = 0",
            (json.dumps({"mode": mode, "applied": applied}), int(system_now_ms()), str(session_id)),
        )

    async def set_execution_context(self, session_id: SessionId, context: str | None) -> None:
        if context is not None and not isinstance(json.loads(context), dict):
            raise SessionStateError("Execution context must be a JSON object")
        row = await self._db.fetch_one(
            "UPDATE session SET execution_context = ?, updated_at = ?"
            " WHERE id = ? AND deleted = 0 RETURNING id",
            (context, int(system_now_ms()), str(session_id)),
        )
        if row is None:
            raise SessionNotFoundError(f"unknown session {session_id}")

    async def create_session(
        self,
        kind: HarnessKind,
        model: str | None = None,
        route_id: RouteId | None = None,
        project_path: ProjectPath | None = None,
        reasoning_effort: str | None = None,
        model_source: ModelSource = ModelSource.GATEWAY,
        execution_backend: ExecutionBackend = ExecutionBackend.HOST,
        privacy_mode: PrivacyMode = PrivacyMode.NONE,
        privacy_scope_id: str | None = None,
        execution_context: str | None = None,
        initial_state: SessionState = SessionState.DISCOVERED,
    ) -> Session:
        SessionPolicy(execution_backend, privacy_mode).validate(model_source)
        if privacy_mode is PrivacyMode.SURROGATE and not privacy_scope_id:
            raise SessionStateError("Protected sessions require a persisted privacy scope")
        session_id = str(uuid.uuid4())
        now = system_now_ms()
        await self._db.execute(
            "INSERT INTO session (id, harness, native_id, native_title, title_overlay,"
            " project_path, created_at, updated_at, state, model, gateway_route_id,"
            " deleted, last_synced_at, reasoning_effort, model_source,"
            " execution_backend, privacy_mode, privacy_scope_id, execution_context)"
            " VALUES (?, ?, NULL, NULL, NULL, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                kind.value,
                str(normalize_fs_path(str(project_path or ""))),
                now,
                now,
                initial_state.value,
                model,
                None if route_id is None else str(route_id),
                now,
                reasoning_effort,
                model_source.value,
                execution_backend.value,
                privacy_mode.value,
                privacy_scope_id,
                execution_context,
            ),
        )
        row = await self._db.fetch_one("SELECT * FROM session WHERE id = ?", (session_id,))
        if row is None:
            raise SessionNotFoundError(f"session {session_id} not persisted")
        return _row_to_session(row)

    async def reveal_native_id(self, session_id: SessionId, native_id: HarnessSessionId) -> Session:
        return await self._write_native_id(session_id, native_id, rotate=False)

    async def rotate_native_id(self, session_id: SessionId, native_id: HarnessSessionId) -> Session:
        return await self._write_native_id(session_id, native_id, rotate=True)

    async def adopt_pi_native_id(
        self,
        session_id: SessionId,
        native_id: HarnessSessionId,
        *,
        ignored_pid: int | None = None,
        check_writer: bool = True,
    ) -> Session:
        session = await self.get_session(session_id)
        if session.harness is not HarnessKind.PI or self._pi_identities is None:
            raise SessionConflictError("Pi session identity adoption is unavailable")
        if session.native_id == native_id:
            return session
        if check_writer and session.execution_backend is ExecutionBackend.HOST:
            owner = await asyncio.to_thread(
                inspect_owner,
                HarnessKind.PI,
                str(native_id),
                str(session.project_path),
                None,
                self._pi_processes(),
                frozenset({ignored_pid}) if ignored_pid is not None else frozenset(),
            )
            if owner.owner is not SessionOwner.UNOWNED:
                raise SessionConflictError("Pi session has an external or unconfirmed writer")
        if not await self._pi_identities.adopt(str(session_id), str(native_id)):
            raise SessionConflictError(f"native id {native_id} already claimed")
        return await self.get_session(session_id)

    async def _write_native_id(
        self, session_id: SessionId, native_id: HarnessSessionId, rotate: bool
    ) -> Session:
        existing = await self._db.fetch_one(
            "SELECT * FROM session WHERE id = ?", (str(session_id),)
        )
        if existing is None:
            raise SessionNotFoundError(f"unknown session {session_id}")
        claimed = await self._db.fetch_one(
            "SELECT id FROM session WHERE native_id = ? AND id != ?",
            (str(native_id), str(session_id)),
        )
        if claimed is not None:
            raise SessionConflictError(f"native id {native_id} already claimed")
        rotation_filter = "" if rotate else " AND native_id IS NULL"
        await self._db.execute(
            f"UPDATE session SET native_id = ? WHERE id = ?{rotation_filter}",
            (str(native_id), str(session_id)),
        )
        updated = await self._db.fetch_one("SELECT * FROM session WHERE id = ?", (str(session_id),))
        if updated is None:
            raise SessionNotFoundError(f"unknown session {session_id}")
        return _row_to_session(updated)
