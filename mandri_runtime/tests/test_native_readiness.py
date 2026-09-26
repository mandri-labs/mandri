import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.ids import HarnessSessionId
from mandri.core.types.execution import PrivacyMode, ProtectionError
from mandri.runtime.control.errors import ControlTransportError
from mandri.runtime.launch_preparation import PreparedLaunch
from mandri.runtime.native_readiness import require_native_identity
from mandri.runtime.service import RuntimeService


@pytest.mark.parametrize(
    "control", [None, SimpleNamespace(capture_identity=AsyncMock(return_value=None))]
)
async def test_missing_native_identity_never_becomes_ready(control):
    with pytest.raises(ProtectionError) as failure:
        await require_native_identity(control)
    assert failure.value.code == "native_initialization_failed"


async def test_native_initialization_timeout_cancels_pending_request():
    cancelled = asyncio.Event()

    async def pending():
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    with pytest.raises(ProtectionError) as failure:
        await require_native_identity(
            SimpleNamespace(capture_identity=pending), timeout_seconds=0.01
        )
    assert failure.value.code == "native_initialization_timeout"
    assert cancelled.is_set()


async def test_native_transport_failure_is_sanitized():
    control = SimpleNamespace(
        capture_identity=AsyncMock(side_effect=ControlTransportError("private"))
    )
    with pytest.raises(ProtectionError) as failure:
        await require_native_identity(control)
    assert failure.value.code == "native_initialization_failed"
    assert "private" not in str(failure.value)


@pytest.mark.parametrize("privacy_mode", [PrivacyMode.NONE, PrivacyMode.SURROGATE])
async def test_host_start_waits_for_native_thread_before_ready(privacy_mode):
    ready = asyncio.Event()
    entered = asyncio.Event()

    async def capture():
        entered.set()
        await ready.wait()
        return HarnessSessionId("thread-new")

    runtime = RuntimeService({"codex": ["synthetic"]})
    runtime._validate_policy = AsyncMock()
    runtime._launch.prepare = Mock(return_value=PreparedLaunch(["synthetic"], {}, None))
    runtime._create_privacy_scope = AsyncMock(return_value=None)
    runtime._create_session_record = AsyncMock(return_value="session")
    runtime._spawn_execution = AsyncMock(
        return_value=SimpleNamespace(returncode=None, stop=AsyncMock())
    )
    runtime._attach_feed = Mock()
    runtime._attach_liveness = Mock()
    runtime._session_state("session").control = SimpleNamespace(
        capture_identity=capture, aclose=AsyncMock()
    )
    runtime._executions.phase = AsyncMock()
    task = asyncio.create_task(
        runtime.start_session("codex", "provider/model", "/workspace", privacy_mode=privacy_mode)
    )
    await asyncio.wait_for(entered.wait(), 1.0)
    assert not task.done()
    runtime._executions.phase.assert_not_awaited()
    ready.set()
    started = await task
    assert started.native_id == "thread-new"
    runtime._executions.phase.assert_awaited_once()


async def test_failed_host_initialization_stops_spawned_process():
    process = SimpleNamespace(returncode=None, stop=AsyncMock())
    runtime = RuntimeService({"codex": ["synthetic"]})
    runtime._create_session_record = AsyncMock(return_value="session")
    runtime._spawn_execution = AsyncMock(return_value=process)
    runtime._attach_feed = Mock()
    runtime._attach_liveness = Mock()
    runtime._session_state("session").control = SimpleNamespace(
        capture_identity=AsyncMock(side_effect=ControlTransportError("closed")), aclose=AsyncMock()
    )
    with pytest.raises(ProtectionError) as failure:
        await runtime.start_session("codex", "provider/model", "/workspace")
    assert failure.value.code == "native_initialization_failed"
    process.stop.assert_awaited_once()
    assert runtime.registry.status("session") == "stopped"
