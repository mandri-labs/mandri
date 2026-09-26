from types import SimpleNamespace
from unittest.mock import AsyncMock

from mandri.core.hub import Hub
from mandri.core.ids import HarnessSessionId
from mandri.runtime.service import RuntimeService
from mandri.runtime.session_feed import session_topic


async def test_steering_preserves_viewer_and_excludes_interrupted_process_from_reconciliation():
    hub = Hub()
    runtime = RuntimeService({}, hub=hub)
    old_process = SimpleNamespace(returncode=None, stop=AsyncMock(return_value=1))
    runtime.registry.mark_live("session", old_process, "agy")
    state = runtime._session_state("session")
    state.viewed = True
    handle = hub.subscribe(session_topic("session"), since=0)
    observed = []

    async def interrupt():
        observed.append("interrupt")
        old_process.returncode = 1
        assert state.resuming
        assert await runtime.reconcile_and_persist() == []
        return True

    control = SimpleNamespace(
        capture_identity=AsyncMock(return_value=HarnessSessionId("native")),
        interrupt=interrupt,
        aclose=AsyncMock(),
    )
    state.control = control

    async def resume(session_id):
        observed.append("resume")
        old_process.stop.assert_awaited_once()
        control.aclose.assert_awaited_once()
        assert state.resuming
        assert state.viewed
        runtime.registry.mark_live(session_id, SimpleNamespace(returncode=None), "agy")

    runtime._resume_session = AsyncMock(side_effect=resume)
    await runtime._restart_agy("session")
    assert observed == ["interrupt", "resume"]
    assert not state.resuming and not state.stopping
    assert state.native_id == "native"
    while not handle.queue.empty():
        assert handle.queue.get_nowait() is not None
    hub.publish(session_topic("session"), {"probe": True})
    assert (await handle.queue.get())["payload"] == {"probe": True}
    hub.unsubscribe(handle)
