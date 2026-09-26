from types import SimpleNamespace
from unittest.mock import AsyncMock

from mandri.core.ids import HarnessKind, RouteId, SessionId
from mandri.core.types.execution import ExecutionBackend, PrivacyMode
from mandri.core.types.model_selection import ModelSource
from mandri.runtime.service import RuntimeService


async def test_live_worker_keeps_route_but_new_execution_rotates_it():
    record = SimpleNamespace(
        id=SessionId("session"),
        harness=HarnessKind.CODEX,
        model_source=ModelSource.GATEWAY,
        gateway_route_id=RouteId("old"),
        execution_backend=ExecutionBackend.DOCKER,
        privacy_mode=PrivacyMode.SURROGATE,
        privacy_scope_id="scope",
        model="provider/model",
        reasoning_effort=None,
    )
    routes = AsyncMock()
    routes.get.return_value = record
    sessions = AsyncMock()
    runtime = RuntimeService({}, routes=routes, sessions=sessions)
    runtime._bind_route = AsyncMock(return_value="new")
    assert await runtime._resume_route(record, "codex", rotate=False) == "old"
    routes.delete.assert_not_awaited()
    runtime._bind_route.assert_not_awaited()
    assert await runtime._resume_route(record, "codex") == "new"
    routes.delete.assert_awaited_once_with(RouteId("old"))
    sessions.set_session_route_id.assert_awaited_once_with(SessionId("session"), RouteId("new"))
