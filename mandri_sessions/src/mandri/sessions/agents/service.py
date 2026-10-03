from __future__ import annotations

import asyncio
import builtins
import dataclasses
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from mandri.core.ids import (
    ActivityState,
    FsPath,
    HarnessKind,
    HarnessSessionId,
    PageToken,
    SessionId,
    SessionState,
)
from mandri.core.ports.agents import AgentDiscovery
from mandri.core.ports.transcripts import SessionRef, TranscriptPage
from mandri.core.types.agents import Agent, AgentState
from mandri.core.types.conversation_status import WorkDelta
from mandri.core.types.execution import ExecutionBackend
from mandri.core.types.sessions import Session, SessionError
from mandri.sessions.agents.docker import contextual_agent, discover_docker_agents
from mandri.sessions.agents.relationships import resolve_agents
from mandri.sessions.agents.status import historical_state
from mandri.sessions.agents.store import AgentStore
from mandri.sessions.errors import SessionNotFoundError
from mandri.sessions.execution_context import transcript_reference
from mandri.sessions.service import SessionsService
from mandri.sessions.transcripts.errors import HarnessStoreUnavailableError
from mandri.sessions.transcripts.recent import recent_jsonl
from mandri.sessions.transcripts.resolver import TranscriptResolver
from mandri.sessions.transcripts.work_delta import jsonl_work_delta

logger = logging.getLogger(__name__)


class AgentHistory:
    def __init__(
        self,
        store: AgentStore,
        sessions: SessionsService,
        readers: TranscriptResolver,
        discoveries: list[AgentDiscovery],
    ) -> None:
        self._store = store
        self._sessions = sessions
        self._readers = readers
        self._discoveries = discoveries
        self._refresh_lock = asyncio.Lock()
        self._refreshed_at = 0.0
        self.cache_lock = asyncio.Lock()
        self._epochs: dict[str, int] = {}
        self._priority: str | None = None
        self._retry_at: dict[str, float] = {}
        self.changed = asyncio.Event()

    async def refresh(self, *, force: bool = False) -> None:
        async with self._refresh_lock:
            if not force and time.monotonic() - self._refreshed_at < 3:
                return
            sessions = await self._sessions.list_sessions()
            host_sessions = [
                session
                for session in sessions
                if session.execution_backend is ExecutionBackend.HOST
            ]
            epochs = dict(self._epochs)
            native = []
            coverage: dict[HarnessKind, frozenset[str] | None] = {}
            incomplete: set[HarnessKind] = set()
            for discovery in self._discoveries if host_sessions else []:
                try:
                    result = await asyncio.to_thread(discovery.discover, host_sessions)
                    native.extend(result.agents)
                    coverage[result.harness] = result.classified_native_ids
                    if not result.complete:
                        incomplete.add(result.harness)
                except (SessionError, OSError, ValueError):
                    logger.exception("Agent discovery unavailable for %s", type(discovery).__name__)
                    raise HarnessStoreUnavailableError("Agent discovery is unavailable") from None
            resolved = resolve_agents(native, host_sessions)
            incomplete_roots = {
                str(session.id) for session in host_sessions if session.harness in incomplete
            }
            session_coverage = {
                str(session.id): coverage.get(session.harness) for session in host_sessions
            }
            for session in sessions:
                if session.execution_backend is not ExecutionBackend.DOCKER:
                    continue
                result = await asyncio.to_thread(discover_docker_agents, session)
                resolved.extend(resolve_agents(result.agents, [session]))
                session_coverage[str(session.id)] = result.classified_native_ids
                if not result.complete:
                    incomplete_roots.add(str(session.id))
            async with self.cache_lock:
                for agent in resolved:
                    try:
                        current = await self._store.get(agent.id)
                    except SessionNotFoundError:
                        current = None
                    if current is not None and self._epochs.get(
                        agent.parent_session_id, 0
                    ) != epochs.get(agent.parent_session_id, 0):
                        agent = dataclasses.replace(
                            current, session_id=agent.session_id or current.session_id
                        )
                    elif current is not None:
                        agent = dataclasses.replace(
                            agent,
                            state=current.state,
                            updated_at=max(current.updated_at, agent.updated_at),
                            task_id=agent.task_id or current.task_id,
                            delegation_id=agent.delegation_id or current.delegation_id,
                        )
                    if agent != current:
                        await self._store.upsert(agent)
                for session in sessions:
                    family = [
                        agent for agent in resolved if agent.parent_session_id == str(session.id)
                    ]
                    if str(session.id) in incomplete_roots and not family:
                        continue
                    revision = hashlib.sha256(
                        json.dumps(
                            [
                                session.harness.value,
                                session.native_id,
                                session.updated_at,
                                [
                                    (agent.id, agent.updated_at, agent.transcript_path)
                                    for agent in sorted(family, key=lambda agent: agent.id)
                                ],
                            ]
                        ).encode()
                    ).hexdigest()
                    await self._store.set_revision(str(session.id), revision)
                resolved_ids = {agent.id for agent in resolved}
                parents = {str(session.id): session for session in sessions}
                for previous in await self._store.list():
                    parent = parents.get(previous.parent_session_id)
                    if (
                        previous.parent_session_id not in incomplete_roots
                        and previous.id not in resolved_ids
                        and (parent is None or not self._active(parent))
                        and self._epochs.get(previous.parent_session_id, 0)
                        == epochs.get(previous.parent_session_id, 0)
                    ):
                        await self._store.remove(previous.id)
                native_children = {(agent.harness, agent.native_id) for agent in native}
                linked = {agent.session_id for agent in resolved}
                await self._store.classify(
                    [
                        session
                        for session in sessions
                        if (
                            session.native_id is None
                            or session_coverage.get(str(session.id)) is None
                            or session.native_id in (session_coverage[str(session.id)] or ())
                        )
                        and (
                            session.execution_backend is ExecutionBackend.DOCKER
                            or (session.harness, session.native_id) not in native_children
                            or str(session.id) in linked
                        )
                    ]
                )
            self.changed.set()
            self._refreshed_at = time.monotonic()

    async def list(self, session_id: str | None = None) -> list[Agent]:
        if session_id is not None:
            self._priority = session_id
            self.changed.set()
        sessions = {str(session.id) for session in await self._sessions.list_sessions()}
        return [
            agent
            for agent in await self._store.list(session_id)
            if agent.parent_session_id in sessions
        ]

    async def get(self, agent_id: str) -> Agent:
        agent = await self._store.get(agent_id)
        await self._sessions.get_session(SessionId(agent.parent_session_id))
        return agent

    async def save(self, agent: Agent) -> None:
        await self._sessions.get_session(SessionId(agent.parent_session_id))
        async with self.cache_lock:
            self._epochs[agent.parent_session_id] = self._epochs.get(agent.parent_session_id, 0) + 1
            await self._store.upsert(agent)
            await self._store.invalidate(agent.parent_session_id)
        self.changed.set()

    async def classified(self) -> builtins.list[str]:
        return await self._store.classified()

    def _active(self, session: Session) -> bool:
        activity = self._sessions.activity_of(session.id)
        return session.state is SessionState.LIVE or (
            activity is not None and activity.state is ActivityState.ACTIVE
        )

    async def backfill(self) -> bool:
        pending = await self._store.pending()
        sessions = sorted(
            await self._sessions.list_sessions(), key=lambda row: row.updated_at, reverse=True
        )
        sessions.sort(key=lambda row: str(row.id) != self._priority)
        for session in sessions:
            identity = str(session.id)
            if (
                identity not in pending
                or self._active(session)
                or time.monotonic() < self._retry_at.get(identity, 0)
            ):
                continue
            epoch = self._epochs.get(identity, 0)
            revision = pending[identity]
            staged = []
            for agent in await self._store.list(identity):
                current_session = await self._sessions.get_session(session.id)
                if self._active(current_session) or self._epochs.get(identity, 0) != epoch:
                    return False
                state = await asyncio.to_thread(
                    historical_state, agent, self._readers, current_session
                )
                if state is AgentState.UNKNOWN:
                    self._retry_at[identity] = time.monotonic() + 30
                    return True
                staged.append(dataclasses.replace(agent, state=state))
            async with self.cache_lock:
                current_session = await self._sessions.get_session(session.id)
                if (
                    self._active(current_session)
                    or self._epochs.get(identity, 0) != epoch
                    or (await self._store.pending()).get(identity) != revision
                ):
                    return False
                for agent in staged:
                    await self._store.upsert(agent)
                await self._store.complete(identity, revision)
            if self._priority == identity:
                self._priority = None
            return True
        return False

    async def history(self, agent_id: str, cursor: PageToken | None, limit: int) -> TranscriptPage:
        limit = max(1, min(limit, 500))
        agent = await self.get(agent_id)
        parent = await self._sessions.get_session(SessionId(agent.parent_session_id))
        if parent.execution_backend is ExecutionBackend.DOCKER:
            agent = contextual_agent(agent, parent)
        if agent.session_id is not None:
            return await self._sessions.history(
                SessionId(agent.session_id), cursor, limit, recent=True
            )
        if agent.harness is HarnessKind.CLAUDE:
            if agent.transcript_path is None:
                raise HarnessStoreUnavailableError("Agent transcript is unavailable")
            return await asyncio.to_thread(recent_jsonl, Path(agent.transcript_path), cursor, limit)
        reader = self._readers.for_session(parent)
        if reader is None:
            raise HarnessStoreUnavailableError("Agent transcript reader is unavailable")
        ref = SessionRef(
            agent.harness,
            HarnessSessionId(agent.native_id),
            transcript_reference(parent).project_path,
            FsPath(agent.transcript_path) if agent.transcript_path else None,
        )
        recent = getattr(reader, "recent", reader.page)
        return await asyncio.to_thread(recent, ref, cursor, limit)

    async def set_state(self, agent_id: str, state: AgentState) -> Agent:
        agent = dataclasses.replace(
            await self.get(agent_id), state=state, updated_at=int(time.time() * 1000)
        )
        await self.save(agent)
        return agent

    async def work_delta(self, agent_id: str, checkpoint: dict[str, Any] | None) -> WorkDelta:
        agent = await self.get(agent_id)
        parent = await self._sessions.get_session(SessionId(agent.parent_session_id))
        if parent.execution_backend is ExecutionBackend.DOCKER:
            agent = contextual_agent(agent, parent)
        if agent.session_id:
            return await self._sessions.work_delta(SessionId(agent.session_id), checkpoint)
        ref = SessionRef(
            agent.harness, HarnessSessionId(agent.native_id),
            transcript_reference(parent).project_path,
            FsPath(agent.transcript_path) if agent.transcript_path else None,
        )
        if agent.harness is HarnessKind.CLAUDE and agent.transcript_path:
            return await asyncio.to_thread(
                jsonl_work_delta, Path(agent.transcript_path), ref, checkpoint
            )
        reader = self._readers.for_session(parent)
        observe = getattr(reader, "work_delta", None)
        if observe is None:
            raise HarnessStoreUnavailableError("Agent work observation is unavailable")
        result: WorkDelta = await asyncio.to_thread(observe, ref, checkpoint)
        return result
