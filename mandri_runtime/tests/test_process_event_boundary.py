import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.hub import Hub, Topic
from mandri.runtime.control.codex import CodexControlAdapter
from mandri.runtime.control_hub import HubEventLines
from mandri.runtime.pump import LinePump
from mandri.runtime.service import RuntimeService
from mandri.runtime.session_feed import session_topic


async def test_control_request_ignores_prior_process_response_with_reused_id():
    hub = Hub()
    topic = Topic("session.test")
    hub.publish(topic, {"source": "codex", "raw": {"id": 1, "result": {"process": "old"}}})
    boundary = hub.sequence(topic)
    lines = HubEventLines(hub, topic, "codex", since=boundary)

    def write(data):
        request = json.loads(data)
        hub.publish(
            topic,
            {"source": "codex", "raw": {"id": request["id"], "result": {"process": "new"}}},
        )

    control = CodexControlAdapter(
        LinePump(lines.chunks), SimpleNamespace(write=write, drain=AsyncMock())
    )
    try:
        assert await control.read_account_rate_limits() == {"process": "new"}
        assert hub.subscribe(topic, since=0).replay[0]["payload"]["raw"]["result"] == {
            "process": "old"
        }
    finally:
        await control.aclose()
        lines.close()
        await hub.close_all()


@pytest.mark.parametrize(
    "harness,raw",
    [
        ("codex", {"method": "turn/started", "params": {"turn": {"id": "turn"}}}),
        ("claude", {"type": "assistant", "message": {"content": []}}),
        ("opencode", {"type": "session.status", "properties": {"status": {"type": "busy"}}}),
        ("agy", {"event": "step_update", "step_update": {"state": "ACTIVE"}}),
        ("pi", {"type": "agent_start"}),
    ],
)
async def test_runtime_liveness_replays_only_current_process_events(harness, raw):
    hub = Hub()
    observed = asyncio.Event()
    port = Mock()
    port.observe.side_effect = lambda evidence: observed.set()
    runtime = RuntimeService({}, hub=hub, liveness=port)
    topic = session_topic("session")
    hub.publish(topic, {"source": harness, "raw": raw})
    runtime._session_state("session").feed_start_seq = hub.sequence(topic)
    hub.publish(topic, {"source": harness, "raw": raw})
    runtime._attach_liveness("session", harness)
    try:
        await asyncio.wait_for(observed.wait(), 1)
        port.observe.assert_called_once()
    finally:
        await runtime._events.stop_liveness_adapter("session")
        await hub.close_all()
