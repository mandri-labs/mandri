"""Tests for reasoning effort threading in the runtime service."""

import dataclasses
import sys
import typing
from types import SimpleNamespace

import pytest
from mandri.core.ids import EpochMs, HarnessKind, ProjectPath, RouteId, SessionId, SessionState
from mandri.core.types.sessions import Session
from mandri.gateway.errors.upstream import RouteNotFoundError
from mandri.runtime import service as runtime_service
from mandri.runtime.service import RuntimeService
from mandri.sessions.errors import SessionNotFoundError
from mandri.sessions.service import SessionsService


class StubRoutes:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.effort_calls: list[tuple[str, str | None]] = []
        self.missing: set[str] = set()

    async def create(
        self,
        provider_name: str,
        model_id: str,
        formats: object,
        reasoning_effort: str | None = None,
    ) -> object:
        self.created.append(
            {
                "provider_name": provider_name,
                "model_id": model_id,
                "reasoning_effort": reasoning_effort,
            }
        )
        route_id = f"route-{len(self.created)}"
        if route_id in self.missing:
            raise RouteNotFoundError(f"unknown route {route_id}")
        return SimpleNamespace(id=route_id)

    async def set_reasoning_effort(self, route_id: str, effort: str | None) -> None:
        self.effort_calls.append((route_id, effort))
        if route_id in self.missing:
            raise RouteNotFoundError(f"unknown route {route_id}")


class StubSessions:
    def __init__(self, record: Session) -> None:
        self._record = record
        self.created: list[dict[str, object]] = []
        self.effort_calls: list[tuple[str, str | None]] = []

    async def create_session(
        self,
        kind: HarnessKind,
        model: str | None,
        route_id: RouteId | None,
        project_path: ProjectPath | None,
        reasoning_effort: str | None = None,
    ) -> object:
        self.created.append(
            {
                "kind": kind,
                "model": model,
                "route_id": route_id,
                "reasoning_effort": reasoning_effort,
            }
        )
        return SimpleNamespace(id="runtime-1")

    async def set_session_state(self, session_id: SessionId, state: SessionState) -> None:
        return None

    async def get_session(self, session_id: SessionId) -> Session:
        if str(session_id) != str(self._record.id):
            raise SessionNotFoundError(f"unknown session {session_id}")
        return self._record

    async def set_session_effort(self, session_id: SessionId, effort: str | None) -> Session:
        self.effort_calls.append((str(session_id), effort))
        self._record = dataclasses.replace(self._record, reasoning_effort=effort)
        return self._record


def _session(route_id: str | None = None, effort: str | None = None) -> Session:
    return Session(
        id=SessionId("s1"),
        harness=HarnessKind.CLAUDE,
        native_id=None,
        native_title=None,
        title_overlay=None,
        project_path=ProjectPath("C:/work/proj"),
        created_at=EpochMs(1_000),
        updated_at=EpochMs(2_000),
        state=SessionState.STOPPED,
        model="prov/model",
        gateway_route_id=None if route_id is None else RouteId(route_id),
        deleted=False,
        last_synced_at=EpochMs(2_000),
        reasoning_effort=effort,
    )


async def test_start_session_threads_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = StubRoutes()
    sessions = StubSessions(_session())

    async def fake_spawn(argv: list[str], cwd: object, env: dict[str, str]) -> object:
        return SimpleNamespace()

    monkeypatch.setattr(runtime_service, "spawn", fake_spawn)
    service = RuntimeService(
        harness_commands={"claude": [sys.executable, "-p"]},
        sessions=typing.cast(SessionsService, sessions),
        routes=typing.cast(object, routes),
    )
    runtime_session = await service.start_session(
        "claude", "prov/model", "C:/work/proj", effort="high"
    )
    assert runtime_session.id == "runtime-1"
    assert routes.created == [
        {"provider_name": "prov", "model_id": "model", "reasoning_effort": "high"}
    ]
    assert sessions.created[0]["reasoning_effort"] == "high"


async def test_start_session_effort_defaults_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = StubRoutes()
    sessions = StubSessions(_session())

    async def fake_spawn(argv: list[str], cwd: object, env: dict[str, str]) -> object:
        return SimpleNamespace()

    monkeypatch.setattr(runtime_service, "spawn", fake_spawn)
    service = RuntimeService(
        harness_commands={"claude": [sys.executable, "-p"]},
        sessions=typing.cast(SessionsService, sessions),
        routes=typing.cast(object, routes),
    )
    await service.start_session("claude", "prov/model", "C:/work/proj")
    assert routes.created[0]["reasoning_effort"] is None
    assert sessions.created[0]["reasoning_effort"] is None


async def test_set_session_effort_updates_registry_and_session() -> None:
    routes = StubRoutes()
    sessions = StubSessions(_session(route_id="route-9"))
    service = RuntimeService(
        harness_commands={},
        sessions=typing.cast(SessionsService, sessions),
        routes=typing.cast(object, routes),
    )
    updated = await service.set_session_effort("s1", "high")
    assert routes.effort_calls == [("route-9", "high")]
    assert sessions.effort_calls == [("s1", "high")]
    assert updated.reasoning_effort == "high"


async def test_set_session_effort_survives_missing_route() -> None:
    routes = StubRoutes()
    routes.missing.add("route-9")
    sessions = StubSessions(_session(route_id="route-9"))
    service = RuntimeService(
        harness_commands={},
        sessions=typing.cast(SessionsService, sessions),
        routes=typing.cast(object, routes),
    )
    updated = await service.set_session_effort("s1", "low")
    assert sessions.effort_calls == [("s1", "low")]
    assert updated.reasoning_effort == "low"


async def test_set_session_effort_updates_session_without_route() -> None:
    routes = StubRoutes()
    sessions = StubSessions(_session(route_id=None))
    service = RuntimeService(
        harness_commands={},
        sessions=typing.cast(SessionsService, sessions),
        routes=typing.cast(object, routes),
    )
    updated = await service.set_session_effort("s1", "medium")
    assert routes.effort_calls == []
    assert updated.reasoning_effort == "medium"


async def test_set_session_effort_accepts_none() -> None:
    routes = StubRoutes()
    sessions = StubSessions(_session(route_id="route-9", effort="high"))
    service = RuntimeService(
        harness_commands={},
        sessions=typing.cast(SessionsService, sessions),
        routes=typing.cast(object, routes),
    )
    updated = await service.set_session_effort("s1", None)
    assert routes.effort_calls == [("route-9", None)]
    assert sessions.effort_calls == [("s1", None)]
    assert updated.reasoning_effort is None


async def test_set_session_effort_unknown_session_raises() -> None:
    routes = StubRoutes()
    sessions = StubSessions(_session(route_id="route-9"))
    service = RuntimeService(
        harness_commands={},
        sessions=typing.cast(SessionsService, sessions),
        routes=typing.cast(object, routes),
    )
    with pytest.raises(SessionNotFoundError):
        await service.set_session_effort("nope", "high")
    assert sessions.effort_calls == []
