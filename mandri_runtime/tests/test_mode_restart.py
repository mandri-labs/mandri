from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.hub import Hub
from mandri.core.ids import HarnessKind, HarnessSessionId, ModeApplication
from mandri.runtime.control.errors import ControlTransportError, ModeRejectedError
from mandri.runtime.errors import SessionNotRunningError
from mandri.runtime.service import RuntimeService
from mandri.runtime.session_feed import session_topic


def setup_restart(harness=HarnessKind.CODEX):
    hub = Hub()
    runtime = RuntimeService({}, hub=hub)
    runtime._sessions = SimpleNamespace(
        get_session=AsyncMock(return_value=SimpleNamespace(harness=harness, interaction_mode=None)),
        set_session_state=AsyncMock(),
        set_session_interaction_mode=AsyncMock(),
    )
    process = SimpleNamespace(returncode=None, stop=AsyncMock(return_value=0))
    runtime.registry.mark_live("session", process, harness.value)
    state = runtime._session_state("session")
    control = SimpleNamespace(
        set_mode=AsyncMock(return_value=ModeApplication.REQUIRES_RESTART),
        interrupt=AsyncMock(return_value=True),
        aclose=AsyncMock(),
    )
    state.control = control
    runtime._capture_identity = AsyncMock(return_value=HarnessSessionId("native"))
    runtime._resume_session = AsyncMock()
    return runtime, state, control, process, hub


@pytest.mark.parametrize(
    "harness,mode",
    [
        (HarnessKind.CODEX, "full-access"),
        (HarnessKind.AGY, "plan"),
        (HarnessKind.PI, "acceptEdits"),
    ],
)
async def test_mode_restart_preserves_subscription_and_passes_mode(harness, mode):
    runtime, state, control, process, hub = setup_restart(harness)
    handle = hub.subscribe(session_topic("session"), since=0)
    assert await runtime.set_session_mode("session", mode) is ModeApplication.RESTARTED
    control.interrupt.assert_awaited_once()
    process.stop.assert_awaited_once()
    runtime._resume_session.assert_awaited_once_with("session", mode=mode)
    runtime._sessions.set_session_interaction_mode.assert_awaited_once_with(
        "session", mode, "restarted"
    )
    assert not state.resuming
    while not handle.queue.empty():
        assert handle.queue.get_nowait() is not None
    hub.publish(session_topic("session"), {"response": "new process"})
    assert (await handle.queue.get())["payload"] == {"response": "new process"}
    hub.unsubscribe(handle)


async def test_failed_restart_does_not_confirm_requested_mode():
    runtime, state, _, _, _ = setup_restart()
    runtime._resume_session.side_effect = ControlTransportError("launch failed")
    with pytest.raises(ControlTransportError, match="launch failed"):
        await runtime.set_session_mode("session", "auto")
    runtime._sessions.set_session_interaction_mode.assert_not_awaited()
    assert not state.resuming


async def test_explicit_stop_during_interrupt_prevents_restart():
    runtime, state, control, process, _ = setup_restart()

    async def interrupted():
        state.stop_revision += 1

    control.interrupt.side_effect = interrupted
    with pytest.raises(SessionNotRunningError):
        await runtime.set_session_mode("session", "auto")
    runtime._resume_session.assert_not_awaited()
    process.stop.assert_not_awaited()
    assert not state.resuming


async def test_permission_refusal_never_restarts_the_harness():
    runtime, _, control, process, _ = setup_restart()
    control.set_mode.side_effect = ModeRejectedError("denied")
    with pytest.raises(ModeRejectedError):
        await runtime.set_session_mode("session", "auto")
    process.stop.assert_not_awaited()
    runtime._resume_session.assert_not_awaited()


async def test_restart_requires_conversation_identity_before_interrupting():
    runtime, _, control, process, _ = setup_restart()
    runtime._capture_identity.return_value = None
    with pytest.raises(ControlTransportError, match="conversation"):
        await runtime.set_session_mode("session", "auto")
    control.interrupt.assert_not_awaited()
    process.stop.assert_not_awaited()


async def test_prompt_is_rejected_during_mode_restart():
    runtime, state, _, _, _ = setup_restart()
    state.resuming = True
    with pytest.raises(ControlTransportError, match="changing state"):
        await runtime.send_session_prompt("session", "hello")
