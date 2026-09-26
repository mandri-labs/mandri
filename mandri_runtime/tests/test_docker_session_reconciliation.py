from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.hub import Hub, Topic
from mandri.runtime.service import RuntimeService


@pytest.mark.parametrize("startup", [False, True])
async def test_reconciliation_repairs_untracked_sessions_and_refreshes_the_session_list(startup):
    repository = SimpleNamespace(
        active=AsyncMock(return_value=[]),
        reconcile_stopped_sessions=AsyncMock(return_value=["orphan"]),
    )
    docker = SimpleNamespace(
        owner="daemon",
        reconcile=AsyncMock(return_value=[]),
        client=SimpleNamespace(run=AsyncMock(return_value="")),
    )
    hub = Hub()
    subscription = hub.subscribe(Topic("sessions.all"))
    runtime = RuntimeService({}, executions=repository, hub=hub)
    runtime._docker = docker
    runtime.registry.mark_live("tracked", SimpleNamespace(returncode=None))
    runtime._session_state("resuming").resuming = True
    runtime._session_state("stopping").stopping = True
    if startup:
        assert await runtime.reconcile_docker() == []
    else:
        assert await runtime.reconcile_and_persist() == ["orphan"]
    repository.reconcile_stopped_sessions.assert_awaited_once_with(
        "daemon", frozenset({"tracked", "resuming", "stopping"})
    )
    assert subscription.queue.get_nowait()["payload"]["raw"] == {"type": "sessions_changed"}
    assert runtime.registry.status("tracked") == "live"
    hub.unsubscribe(subscription)
