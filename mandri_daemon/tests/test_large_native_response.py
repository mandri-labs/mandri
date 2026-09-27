import asyncio
import json
from types import SimpleNamespace

from mandri.core.hub import Hub, Topic
from mandri.core.ids import HarnessKind, HarnessSessionId
from mandri.daemon.serve import build_harness_adapters
from mandri.runtime.adapters import AdapterContext
from mandri.runtime.native_readiness import require_native_identity
from mandri.runtime.process import DEFAULT_LINE_LIMIT
from mandri.runtime.session_feed import SessionFeed


async def test_large_resume_response_reaches_native_control_without_degradation():
    hub = Hub()
    topic = Topic("session.test")
    stdout, stderr = asyncio.StreamReader(), asyncio.StreamReader()
    streams = SimpleNamespace(stdout=stdout, stderr=stderr)
    text = "x" * (DEFAULT_LINE_LIMIT + 1)

    async def write_stdin(data):
        request = json.loads(data)
        if "id" not in request:
            return
        result = (
            {"thread": {"id": "existing-thread", "turns": [{"text": text}]}}
            if request["method"] == "thread/resume"
            else {}
        )
        stdout.feed_data((json.dumps({"id": request["id"], "result": result}) + "\n").encode())

    process = SimpleNamespace(process=streams, write_stdin=write_stdin)
    adapters = build_harness_adapters(
        AdapterContext(
            kind=HarnessKind.CODEX,
            process=process,
            hub=hub,
            topic=topic,
            resume_thread_id=HarnessSessionId("existing-thread"),
        )
    )
    assert adapters is not None
    feed = SessionFeed(hub, "test", HarnessKind.CODEX, streams)
    feed.start()
    try:
        identity = await require_native_identity(adapters.control, timeout_seconds=5)
        assert identity == "existing-thread"
        replay = hub.subscribe(topic, since=0)
        assert len(replay.replay) == 2
        assert all(frame["payload"]["source"] == "codex" for frame in replay.replay)
        assert replay.replay[-1]["payload"]["raw"]["result"]["thread"]["turns"] == [{"text": text}]
    finally:
        await adapters.control.aclose()
        await feed.stop()
        await hub.close_all()
