from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.hub import Hub
from mandri.runtime.service import RuntimeService
from mandri.runtime.session_feed import session_topic


@pytest.mark.parametrize("operation", ["stop", "reconcile", "reconcile_and_persist", "rollback"])
async def test_process_exit_preserves_session_subscription_and_replay(operation):
    hub = Hub()
    runtime = RuntimeService({}, hub=hub)
    process = SimpleNamespace(returncode=None, stop=AsyncMock(return_value=0))
    runtime.registry.mark_live("session", process, "pi")
    topic = session_topic("session")
    hub.publish(topic, {"response": "before exit"})
    handle = hub.subscribe(topic, since=0)
    if operation == "stop":
        await runtime.stop_session("session", restore_native=False)
    elif operation == "rollback":
        await runtime._rollback_session("session")
    else:
        process.returncode = 0
        await getattr(runtime, operation)()

    assert not handle.closed
    if operation != "rollback":
        stopped = handle.queue.get_nowait()
        assert stopped["payload"]["type"] == "session_stopped"
        assert stopped["seq"] == 2
    hub.publish(topic, {"response": "after resume"})
    assert handle.queue.get_nowait()["payload"] == {"response": "after resume"}
    replay = hub.subscribe(topic, since=0)
    assert replay.replay[0]["payload"] == {"response": "before exit"}
    assert [frame["seq"] for frame in replay.replay] == list(range(1, len(replay.replay) + 1))
    hub.unsubscribe(handle)
    hub.unsubscribe(replay)
