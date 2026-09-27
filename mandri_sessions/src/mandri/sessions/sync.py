"""Session reconciliation between harness stores and the local read model."""

import asyncio
import dataclasses
import logging
import uuid
from collections.abc import Callable, Mapping
from typing import Protocol

from mandri.core.clock import system_now_ms
from mandri.core.fs.paths import normalize_fs_path
from mandri.core.hub import Hub, Topic
from mandri.core.ids import (
    ClaimEvidence,
    EpochMs,
    HarnessKind,
    ProjectPath,
    SessionId,
    SessionState,
    SessionTitle,
)
from mandri.core.ports.database import DatabasePort
from mandri.core.types.config import SyncConfig
from mandri.core.types.execution import PrivacyMode
from mandri.core.types.model_selection import ModelSource
from mandri.core.types.sessions import Session
from mandri.sessions.activity import (
    ActivityTracker,
    ActivityTransition,
    SessionActivity,
    backoff_interval,
)
from mandri.sessions.claiming import NativeSessionRecord, claim_evidence_matches
from mandri.sessions.lineage import SessionLineage, ordered_sessions
from mandri.sessions.session_rows import row_to_session

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 30
RUNTIMES_TOPIC = Topic("runtimes")
SESSIONS_ALL_TOPIC = Topic("sessions.all")


class SessionsBackend(Protocol):
    def fetch(self) -> list[Session]:
        raise NotImplementedError

    def rename(self, session_id: SessionId, title: SessionTitle) -> None:
        raise NotImplementedError

    def delete(self, session_id: SessionId) -> None:
        raise NotImplementedError

    def exists(self, session_id: SessionId) -> bool:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class HarnessState:
    degraded: bool = False
    last_sync_error: Exception | None = None
    last_sync_at: EpochMs | None = None


@dataclasses.dataclass(frozen=True)
class _PendingClaim:
    session_id: SessionId
    evidence: ClaimEvidence


class SyncEngine:
    def __init__(
        self,
        db: DatabasePort,
        harnesses: Mapping[HarnessKind, SessionsBackend],
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        hub: Hub | None = None,
        sync_config: SyncConfig | None = None,
        docker_title_reader: Callable[[Session], SessionTitle | None] | None = None,
    ) -> None:
        self._db = db
        self._docker_title_reader = docker_title_reader
        self._harnesses = dict(harnesses)
        self._ttl_seconds = ttl_seconds
        self._hub = hub
        self._sync_config = sync_config if sync_config is not None else SyncConfig()
        self._states: dict[HarnessKind, HarnessState] = {
            kind: HarnessState() for kind in self._harnesses
        }
        self._published_degraded: dict[HarnessKind, bool] = {}
        self._activity = ActivityTracker()
        self._next_observation: dict[SessionId, EpochMs] = {}

    async def sync(self) -> None:
        for kind, backend in self._harnesses.items():
            try:
                candidates = await self._deletion_candidates(kind)
                rows = await asyncio.to_thread(backend.fetch)
                await self._reconcile(kind, rows)
                await self._tombstone_missing(rows, candidates)
            except Exception as error:
                logger.warning("Session inventory unavailable for %s: %s", kind.value, error)
                self._states[kind] = HarnessState(
                    degraded=True, last_sync_error=error, last_sync_at=system_now_ms()
                )
                self._publish_runtime_state(kind, degraded=True)
                continue
            self._states[kind] = HarnessState(degraded=False, last_sync_at=system_now_ms())
            self._publish_runtime_state(kind, degraded=False)

        await self._sync_docker_titles()

    async def _sync_docker_titles(self) -> None:
        if self._docker_title_reader is None:
            return
        rows = await self._db.fetch_all(
            "SELECT * FROM session WHERE execution_backend = 'docker' AND deleted = 0"
            " AND native_id IS NOT NULL AND execution_context IS NOT NULL"
        )
        changed = False
        for row in rows:
            session = row_to_session(row)
            try:
                title = await asyncio.to_thread(self._docker_title_reader, session)
            except Exception:
                logger.debug("Docker session title unavailable for %s", session.id, exc_info=True)
                continue
            if not title:
                continue
            updated = await self._db.fetch_one(
                "UPDATE session SET native_title = ?, last_synced_at = ?"
                " WHERE id = ? AND deleted = 0 AND execution_backend = 'docker'"
                " AND native_id = ? AND execution_context = ? AND native_title IS NOT ?"
                " RETURNING id",
                (
                    title,
                    system_now_ms(),
                    str(session.id),
                    str(session.native_id),
                    session.execution_context,
                    title,
                ),
            )
            changed |= updated is not None
        if changed:
            self._publish_sessions_changed()

    def _publish_sessions_changed(self) -> None:
        if self._hub is not None:
            self._hub.publish(
                SESSIONS_ALL_TOPIC,
                {"source": "mandri", "ts": system_now_ms(), "raw": {"type": "sessions_changed"}},
            )

    def harness_state(self, kind: HarnessKind) -> HarnessState:
        return self._states.get(kind, HarnessState())

    def activity_of(self, session_id: SessionId) -> SessionActivity | None:
        return self._activity.snapshot(session_id)

    def backend(self, kind: HarnessKind) -> SessionsBackend | None:
        return self._harnesses.get(kind)

    def stale(self, kind: HarnessKind, ttl_seconds: int | None = None) -> bool:
        state = self._states.get(kind)
        ttl = self._ttl_seconds if ttl_seconds is None else ttl_seconds
        if state is None or state.degraded or state.last_sync_at is None:
            return True
        return system_now_ms() - state.last_sync_at > ttl * 1000

    def _publish_runtime_state(self, kind: HarnessKind, degraded: bool) -> None:
        if self._hub is None:
            return
        if self._published_degraded.get(kind) is degraded:
            return
        self._published_degraded[kind] = degraded
        self._hub.publish(
            RUNTIMES_TOPIC,
            {"harness": kind.value, "installed": True, "degraded": degraded},
        )

    async def _reconcile(self, kind: HarnessKind, rows: list[Session]) -> None:
        pending = await self._pending_claims(kind)
        changed = False
        for row in ordered_sessions(rows):
            row = dataclasses.replace(
                row, project_path=ProjectPath(normalize_fs_path(str(row.project_path)))
            )
            if row.native_id is None:
                continue
            existing = await self._existing_by_native_id(kind, row.native_id)
            row = await SessionLineage(self._db).prepare(row, existing)
            if existing is not None and (not row.project_path or existing.get("worktree")):
                row = dataclasses.replace(
                    row, project_path=ProjectPath(str(existing["project_path"]))
                )
            changed |= existing is None or (
                existing.get("native_title") != row.native_title
                or existing.get("project_path") != row.project_path
                or existing.get("parent_native_id") != row.parent_native_id
                or existing.get("privacy_mode") != row.privacy_mode.value
                or (
                    kind in (HarnessKind.AGY, HarnessKind.PI)
                    and existing.get("state") == SessionState.DISCOVERED.value
                    and row.model is not None
                    and (
                        existing.get("model") != row.model
                        or existing.get("model_source") != row.model_source.value
                        or (
                            kind is HarnessKind.PI
                            and existing.get("reasoning_effort") != row.reasoning_effort
                        )
                    )
                )
            )
            if existing is not None:
                session_id = SessionId(str(existing["id"]))
                await SessionLineage(self._db).persist(existing, row)
                await self._update_existing(existing, row)
            else:
                claimed = await self._claim_verified(kind, pending, row)
                if claimed is None:
                    session_id = await self._insert_new(kind, row)
                else:
                    session_id = claimed
            self._observe_activity(session_id, row)
        if changed:
            self._publish_sessions_changed()

    async def _pending_claims(self, kind: HarnessKind) -> list[_PendingClaim]:
        rows = await self._db.fetch_all(
            "SELECT id, project_path, created_at FROM session"
            " WHERE harness = ? AND native_id IS NULL AND deleted = 0"
            " AND execution_backend = 'host'"
            " ORDER BY created_at ASC",
            (kind.value,),
        )
        window_ms = self._ttl_seconds * 1000
        claims: list[_PendingClaim] = []
        for row in rows:
            created_at = EpochMs(int(row["created_at"]))
            claims.append(
                _PendingClaim(
                    session_id=SessionId(str(row["id"])),
                    evidence=ClaimEvidence(
                        harness=kind,
                        project_path=ProjectPath(str(row["project_path"])),
                        captured_native_id=None,
                        time_window=(
                            EpochMs(created_at - window_ms),
                            EpochMs(created_at + window_ms),
                        ),
                    ),
                )
            )
        return claims

    async def _existing_by_native_id(
        self, kind: HarnessKind, native_id: object
    ) -> dict[str, object] | None:
        return await self._db.fetch_one(
            "SELECT * FROM session WHERE harness = ? AND native_id = ?"
            " AND execution_backend = 'host'",
            (kind.value, str(native_id)),
        )

    async def _claim_verified(
        self, kind: HarnessKind, pending: list[_PendingClaim], row: Session
    ) -> SessionId | None:
        if row.native_id is None or row.parent_native_id is not None:
            return None
        candidate = NativeSessionRecord(
            harness=kind,
            native_id=row.native_id,
            project_path=row.project_path,
            created_at=row.created_at,
        )
        for claim in pending:
            if not claim_evidence_matches(claim.evidence, candidate):
                continue
            pending.remove(claim)
            await self._claim_pending(claim.session_id, row)
            return claim.session_id
        return None

    async def _deletion_candidates(self, kind: HarnessKind) -> list[dict[str, object]]:
        return await self._db.fetch_all(
            "SELECT id, native_id, state, updated_at, last_synced_at FROM session"
            " WHERE harness = ? AND native_id IS NOT NULL AND deleted = 0"
            " AND state != ? AND worktree IS NULL AND execution_backend = 'host'",
            (kind.value, SessionState.LIVE.value),
        )

    async def _tombstone_missing(
        self, rows: list[Session], candidates: list[dict[str, object]]
    ) -> None:
        present = {str(row.native_id) for row in rows if row.native_id is not None}
        changed = False
        for record in candidates:
            if str(record["native_id"]) in present:
                continue
            deleted = await self._db.fetch_one(
                "UPDATE session SET deleted = 1 WHERE id = ? AND native_id = ?"
                " AND state = ? AND updated_at = ? AND last_synced_at = ?"
                " AND deleted = 0 AND worktree IS NULL AND execution_backend = 'host'"
                " RETURNING id",
                (
                    record["id"],
                    record["native_id"],
                    record["state"],
                    record["updated_at"],
                    record["last_synced_at"],
                ),
            )
            changed |= deleted is not None
            if deleted is not None:
                logger.debug(
                    "Session %s missing from native inventory; native_id=%s state=%s",
                    record["id"],
                    record["native_id"],
                    record["state"],
                )
        if changed:
            self._publish_sessions_changed()

    def _observe_activity(self, session_id: SessionId, row: Session) -> None:
        now = system_now_ms()
        due = self._next_observation.get(session_id)
        if due is not None and now < due:
            return
        transition = self._activity.observe(
            session_id, row.updated_at, now, self._sync_config.activity_quiet_seconds
        )
        level = self._activity.backoff_level(session_id)
        interval_ms = backoff_interval(self._sync_config, level) * 1000
        self._next_observation[session_id] = EpochMs(now + interval_ms)
        if transition is not None:
            self._publish_activity(transition, row.updated_at, row.harness)

    def _publish_activity(
        self, transition: ActivityTransition, last_activity_at: EpochMs, harness: HarnessKind
    ) -> None:
        if self._hub is None:
            return
        self._hub.publish(
            SESSIONS_ALL_TOPIC,
            {
                "type": "activity",
                "session_id": str(transition.session_id),
                "harness": harness.value,
                "activity": transition.state.value,
                "last_activity_at": int(last_activity_at),
            },
        )

    async def _insert_new(self, kind: HarnessKind, row: Session) -> SessionId:
        session_id = SessionId(str(uuid.uuid4()))
        source = ModelSource.GATEWAY
        if kind in (HarnessKind.AGY, HarnessKind.PI):
            source = row.model_source
        if row.privacy_mode is PrivacyMode.SURROGATE:
            source = ModelSource.GATEWAY
        elif kind in (HarnessKind.CODEX, HarnessKind.CLAUDE) and (
            row.model is None or "/" not in row.model
        ):
            source = ModelSource.NATIVE
        await self._db.execute(
            "INSERT INTO session (id, harness, native_id, native_title, title_overlay,"
            " project_path, created_at, updated_at, state, model, gateway_route_id,"
            " deleted, last_synced_at, model_source, execution_backend, privacy_mode,"
            " privacy_scope_id, parent_native_id, parent_session_id, reasoning_effort)"
            " VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, NULL, 0, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(session_id),
                kind.value,
                str(row.native_id),
                row.native_title,
                row.project_path,
                row.created_at,
                row.updated_at,
                SessionState.DISCOVERED.value,
                row.model,
                system_now_ms(),
                source.value,
                row.execution_backend.value,
                row.privacy_mode.value,
                row.privacy_scope_id,
                row.parent_native_id,
                row.parent_session_id,
                row.reasoning_effort,
            ),
        )
        return session_id

    async def _update_existing(self, existing: dict[str, object], row: Session) -> None:
        await self._db.execute(
            "UPDATE session SET native_title = ?, project_path = ?,"
            " updated_at = MAX(updated_at, ?), last_synced_at = ? WHERE id = ?",
            (
                row.native_title,
                row.project_path,
                row.updated_at,
                system_now_ms(),
                str(existing["id"]),
            ),
        )
        if row.harness is HarnessKind.AGY and row.model is not None:
            await self._db.execute(
                "UPDATE session SET model = ?, model_source = ? WHERE id = ? AND state = ?"
                " AND privacy_mode = 'none'",
                (
                    row.model,
                    row.model_source.value,
                    str(existing["id"]),
                    SessionState.DISCOVERED.value,
                ),
            )

        if row.harness is HarnessKind.PI and row.model is not None:
            await self._db.execute(
                "UPDATE session SET model = ?, model_source = ?, reasoning_effort = ?"
                " WHERE id = ? AND state = ? AND gateway_route_id IS NULL"
                " AND privacy_mode = 'none'",
                (
                    row.model,
                    row.model_source.value,
                    row.reasoning_effort,
                    str(existing["id"]),
                    SessionState.DISCOVERED.value,
                ),
            )

    async def _claim_pending(self, pending_id: SessionId, row: Session) -> None:
        await self._db.execute(
            "UPDATE session SET native_id = ?, native_title = ?, project_path = ?,"
            " updated_at = MAX(updated_at, ?), last_synced_at = ? WHERE id = ?",
            (
                str(row.native_id),
                row.native_title,
                row.project_path,
                row.updated_at,
                system_now_ms(),
                str(pending_id),
            ),
        )
