from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.hub import Hub
from mandri.core.ports.control import PromptOutcome, PromptState
from mandri.runtime.service import RuntimeService
from mandri.runtime.session_feed import session_topic


@pytest.mark.parametrize("harness", ["agy", "codex", "claude", "opencode"])
async def test_model_restart_keeps_subscriber_and_delivers_prompt_once(harness):
    hub = Hub()
    runtime = RuntimeService({}, hub=hub)
    process = SimpleNamespace(returncode=None, stop=AsyncMock(return_value=0))
    runtime.registry.mark_live("session", process, harness)
    state = runtime._session_state("session")
    state.viewed = True
    state.control = SimpleNamespace(aclose=AsyncMock())
    runtime.is_busy = Mock(return_value=False)
    runtime._models.needs_restart = AsyncMock(return_value=True)
    handle = hub.subscribe(session_topic("session"), since=0)
    control = SimpleNamespace(send_prompt=AsyncMock(return_value=PromptOutcome(PromptState.QUEUED)))

    async def resume(session_id, **kwargs):
        runtime.registry.mark_live(session_id, SimpleNamespace(returncode=None), harness)
        state.control = control

    runtime._resume_session = AsyncMock(side_effect=resume)
    outcome = await runtime.send_session_prompt("session", "continue")
    assert outcome.state is PromptState.QUEUED
    control.send_prompt.assert_awaited_once_with("continue")
    assert state.viewed
    while not handle.queue.empty():
        assert handle.queue.get_nowait() is not None
    hub.publish(session_topic("session"), {"response": "delivered"})
    assert (await handle.queue.get())["payload"] == {"response": "delivered"}
    hub.unsubscribe(handle)


async def test_model_restart_failure_releases_transition_guard():
    runtime = RuntimeService({})
    runtime._models.needs_restart = AsyncMock(return_value=True)
    state = runtime._session_state("session")

    async def stop(*args, **kwargs):
        assert state.resuming

    runtime.stop_session = AsyncMock(side_effect=stop)
    runtime._resume_session = AsyncMock(side_effect=RuntimeError("launch failed"))
    with pytest.raises(RuntimeError, match="launch failed"):
        await runtime._apply_pending_model("session")
    assert not state.resuming
