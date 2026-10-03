"""Effort handling tests for the runtime REST routes."""

import dataclasses
from types import SimpleNamespace
from typing import Any

from mandri.api.deps import GatewayWiring, gateway_wiring, runtime_service, sessions_service
from mandri.core.ids import EpochMs, HarnessKind, ProjectPath, RouteId, SessionId, SessionState
from mandri.core.types.sessions import Session
from mandri.gateway.reasoning_catalog import ReasoningCatalog, ReasoningInfo
from mandri.sessions.errors import SessionNotFoundError

MODEL_ARG = "openrouter/anthropic/claude-4"
SESSION_ID = "8b1f3d2a-4c5e-4f6a-9b0c-1d2e3f4a5b6c"
CATALOG = ReasoningCatalog(
    {
        ("openrouter", "anthropic/claude-4"): ReasoningInfo(
            efforts=["low", "medium", "high"], default_effort="medium"
        )
    }
)


class StubRuntime:
    def __init__(self, record: Session) -> None:
        self.record = record
        self.start_calls: list[dict[str, Any]] = []
        self.effort_calls: list[tuple[str, str | None]] = []

    async def start_session(
        self,
        harness: str,
        model: str,
        cwd: str,
        env_wiring: dict[str, str] | None = None,
        mode: str | None = None,
        effort: str | None = None,
    ) -> SimpleNamespace:
        self.start_calls.append({"harness": harness, "model": model, "effort": effort})
        return SimpleNamespace(
            id=SESSION_ID, harness=harness, route_id="route-1", project_path=str(cwd), mode=mode
        )

    async def set_session_effort(self, session_id: str, effort: str | None) -> Session:
        self.effort_calls.append((session_id, effort))
        return dataclasses.replace(self.record, reasoning_effort=effort)


class StubSessions:
    def __init__(self, record: Session | None) -> None:
        self.record = record
        self.statuses = None

    async def get_session(self, session_id: SessionId) -> Session:
        if self.record is None:
            raise SessionNotFoundError(f"no session {session_id}")
        return self.record

    def activity_of(self, session_id: SessionId) -> None:
        return None


def make_record(model: str | None = MODEL_ARG) -> Session:
    return Session(
        id=SessionId(SESSION_ID),
        harness=HarnessKind.OPENCODE,
        native_id=None,
        native_title=None,
        title_overlay=None,
        project_path=ProjectPath("D:/tmp"),
        created_at=EpochMs(1),
        updated_at=EpochMs(1),
        state=SessionState.LIVE,
        model=model,
        gateway_route_id=RouteId("route-1"),
        deleted=False,
        last_synced_at=EpochMs(1),
    )


def make_client_factory(make_client, record: Session | None):
    runtime = StubRuntime(record if record is not None else make_record())
    sessions = StubSessions(record)
    overrides = {
        gateway_wiring: lambda: GatewayWiring(
            registry=object(), openai=object(), anthropic=object(), reasoning_catalog=CATALOG
        ),
        runtime_service: lambda: runtime,
        sessions_service: lambda: sessions,
    }
    return make_client(overrides), runtime


def test_start_session_passes_effort_to_runtime(make_client) -> None:
    client, runtime = make_client_factory(make_client, None)
    response = client.post(
        "/v1/runtime/sessions",
        json={"harness": "opencode", "model": MODEL_ARG, "cwd": "D:/tmp", "effort": "medium"},
    )
    assert response.status_code == 201
    assert runtime.start_calls == [{"harness": "opencode", "model": MODEL_ARG, "effort": "medium"}]


def test_start_session_normalizes_empty_effort(make_client) -> None:
    client, runtime = make_client_factory(make_client, None)
    response = client.post(
        "/v1/runtime/sessions",
        json={"harness": "opencode", "model": MODEL_ARG, "cwd": "D:/tmp", "effort": ""},
    )
    assert response.status_code == 201
    assert runtime.start_calls[0]["effort"] is None


def test_start_session_rejects_unknown_effort(make_client) -> None:
    client, runtime = make_client_factory(make_client, None)
    response = client.post(
        "/v1/runtime/sessions",
        json={"harness": "opencode", "model": MODEL_ARG, "cwd": "D:/tmp", "effort": "ultra"},
    )
    assert response.status_code == 400
    body = response.json()["error"]
    assert body["code"] == "invalid_effort"
    assert body["detail"]["allowed"] == ["low", "medium", "high"]
    assert runtime.start_calls == []


def test_start_session_without_catalog_accepts_any_effort(make_client) -> None:
    runtime = StubRuntime(make_record())
    client = make_client(
        {
            gateway_wiring: lambda: GatewayWiring(
                registry=object(), openai=object(), anthropic=object(), reasoning_catalog=None
            ),
            runtime_service: lambda: runtime,
            sessions_service: lambda: StubSessions(None),
        }
    )
    response = client.post(
        "/v1/runtime/sessions",
        json={"harness": "opencode", "model": MODEL_ARG, "cwd": "D:/tmp", "effort": "ultra"},
    )
    assert response.status_code == 201
    assert runtime.start_calls[0]["effort"] == "ultra"


def test_start_session_rejects_effort_when_model_explicitly_disables_thinking(
    make_client, monkeypatch
):
    monkeypatch.setattr(
        "mandri_api.tests.test_runtime_effort.CATALOG",
        ReasoningCatalog({("openrouter", "anthropic/claude-4"): ReasoningInfo([])}),
    )
    client, runtime = make_client_factory(make_client, None)
    response = client.post(
        "/v1/runtime/sessions",
        json={"harness": "opencode", "model": MODEL_ARG, "cwd": "D:/tmp", "effort": "on"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["detail"]["allowed"] == []
    assert runtime.start_calls == []


def test_patch_effort_updates_session(make_client) -> None:
    record = make_record()
    client, runtime = make_client_factory(make_client, record)
    response = client.patch(f"/v1/runtime/sessions/{SESSION_ID}/effort", json={"effort": "low"})
    assert response.status_code == 200
    assert runtime.effort_calls == [(SESSION_ID, "low")]
    assert response.json()["id"] == SESSION_ID
    assert response.json()["reasoning_effort"] == "low"


def test_patch_effort_clears_effort_with_null(make_client) -> None:
    record = make_record()
    client, runtime = make_client_factory(make_client, record)
    response = client.patch(f"/v1/runtime/sessions/{SESSION_ID}/effort", json={"effort": None})
    assert response.status_code == 200
    assert runtime.effort_calls == [(SESSION_ID, None)]
    assert response.json()["reasoning_effort"] is None


def test_patch_effort_rejects_unknown_effort(make_client) -> None:
    record = make_record()
    client, runtime = make_client_factory(make_client, record)
    response = client.patch(f"/v1/runtime/sessions/{SESSION_ID}/effort", json={"effort": "ultra"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_effort"
    assert runtime.effort_calls == []


def test_patch_effort_skips_validation_without_model(make_client) -> None:
    record = make_record(model=None)
    client, runtime = make_client_factory(make_client, record)
    response = client.patch(f"/v1/runtime/sessions/{SESSION_ID}/effort", json={"effort": "ultra"})
    assert response.status_code == 200
    assert runtime.effort_calls == [(SESSION_ID, "ultra")]


def test_patch_effort_unknown_session_returns_404(make_client) -> None:
    client, runtime = make_client_factory(make_client, None)
    runtime.effort_calls = []
    sessions_stub = StubSessions(None)
    client = make_client(
        {
            gateway_wiring: lambda: GatewayWiring(
                registry=object(), openai=object(), anthropic=object(), reasoning_catalog=CATALOG
            ),
            runtime_service: lambda: runtime,
            sessions_service: lambda: sessions_stub,
        }
    )
    response = client.patch(
        "/v1/runtime/sessions/9a7b6c5d-4e3f-4a2b-8c1d-0e9f8a7b6c5d/effort", json={"effort": "low"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"
