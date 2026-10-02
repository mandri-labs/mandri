from unittest.mock import AsyncMock

import httpx
import pytest
from mandri.runtime import native_id
from mandri.runtime.control.errors import ControlTransportError


async def test_initialization_is_read_only_until_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if len(calls) == 1:
            raise httpx.ReadTimeout("initializing", request=request)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [], "cursor": {}})
        return httpx.Response(200, json={"data": {"id": "native-child"}})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        native_id.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs),
    )
    assert await native_id.await_opencode_session_id(4096) == "native-child"
    assert calls == ["GET", "GET", "POST"]


async def test_ambiguous_creation_is_not_replayed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [], "cursor": {}})
        raise httpx.ReadTimeout("reply lost after creation", request=request)

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        native_id.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs),
    )
    with pytest.raises(ControlTransportError, match="could not be confirmed"):
        await native_id.await_opencode_session_id(4096)
    assert calls == ["GET", "POST"]


async def test_startup_deadline_prevents_infinite_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = AsyncMock(return_value=None)
    monkeypatch.setattr(native_id, "capture_opencode_session_id", probe)
    monkeypatch.setattr(native_id, "_OPENCODE_STARTUP_SECONDS", 0)
    with pytest.raises(ControlTransportError, match="startup expired"):
        await native_id.await_opencode_session_id(4096)
    probe.assert_not_awaited()


async def test_resume_waits_for_legacy_migration_before_declaring_session_missing(monkeypatch):
    requests = []
    responses = iter(
        [
            httpx.Response(404),
            httpx.Response(200, json={"status": "running", "progress": {"label": "Migrating"}}),
            httpx.Response(200, json={"data": {"id": "legacy"}}),
        ]
    )
    client_type = httpx.AsyncClient

    def respond(request):
        requests.append(request.url.path)
        return next(responses)

    monkeypatch.setattr(
        native_id.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs),
    )
    assert await native_id.verify_opencode_session_id(4096, "legacy") == "legacy"
    assert requests == [
        "/api/session/legacy",
        "/api/experimental/migration/v1",
        "/api/session/legacy",
    ]
