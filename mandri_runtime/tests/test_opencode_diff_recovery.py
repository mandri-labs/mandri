import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from mandri.core.hub import Hub, Topic
from mandri.core.ids import SessionId
from mandri.runtime.commands import CommandService
from mandri.runtime.control.opencode_diffs import OpencodeDiffRecovery
from mandri.runtime.control.opencode_events import OpencodeEvents
from mandri.runtime.liveness import OpencodeLivenessAdapter, WorkingStateTracker


async def test_slow_diff_does_not_block_compaction_or_the_next_turn():
    requested, release = asyncio.Event(), asyncio.Event()
    queue = asyncio.Queue()

    async def call(method, path, body):
        if "/message/" in path:
            return httpx.Response(
                200, json={"data": {"type": "user", "id": path.rsplit("/", 1)[-1], "text": "Work"}}
            )
        requested.set()
        await release.wait()
        return httpx.Response(200, json={"data": [{"file": "first.py", "patch": "+one"}]})

    tracker = WorkingStateTracker()
    session = SessionId("session")
    tracker.register(session)
    adapter = OpencodeLivenessAdapter(Hub(), Topic("session.session"), tracker, session)
    control = AsyncMock()
    control.list_commands.return_value = [{"id": "compact", "name": "compact", "kind": "command"}]
    control.execute_command.return_value = {"kind": "transcript", "message": "Submitted"}
    commands = CommandService(
        lambda _: control, lambda _: True, lambda _: tracker.working_state(session).busy
    )
    recovery = OpencodeDiffRecovery(call, queue.put_nowait)
    converter = OpencodeEvents(call, recovery.submit)

    async def absorb(kind, **data):
        converted = await converter.convert({"type": kind, "data": {"sessionID": "root", **data}})
        for raw in converted:
            adapter._absorb({"payload": {"source": "opencode", "raw": raw}})
        return converted

    try:
        await absorb("session.execution.started")
        await absorb("session.inbox.delivered", inboxID="first")
        assert tracker.working_state(session).busy
        async with asyncio.timeout(1):
            terminal = await absorb("session.execution.succeeded")
            await requested.wait()
        assert terminal[0]["properties"]["status"] == {"type": "idle"}
        assert not tracker.working_state(session).busy
        assert queue.empty()
        compact = await commands.invoke("session", "invocation", "compact", "")
        await commands._tasks["session"]
        assert compact.state == "succeeded"
        await absorb("session.execution.started")
        await absorb("session.inbox.delivered", inboxID="second")
        assert tracker.working_state(session).busy
        release.set()
        async with asyncio.timeout(1):
            recovered = await queue.get()
        assert recovered["properties"]["turnID"] == "first"
        adapter._absorb({"payload": {"source": "opencode", "raw": recovered}})
        assert tracker.working_state(session).busy
        await absorb("session.execution.succeeded")
        assert not tracker.working_state(session).busy
    finally:
        await recovery.close()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.Response(200, json={"data": []}),
        httpx.Response(200, text="invalid"),
    ],
)
async def test_missing_diffs_do_not_publish_an_empty_snapshot(response):
    publish = AsyncMock()
    recovery = OpencodeDiffRecovery(AsyncMock(return_value=response), publish)
    await recovery._recover("root", "user")
    publish.assert_not_called()


async def test_close_cancels_unfinished_diff_requests():
    requested, cancelled = asyncio.Event(), asyncio.Event()

    async def call(*args):
        requested.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    recovery = OpencodeDiffRecovery(call, lambda _: None)
    recovery.submit("root", "user")
    await requested.wait()
    await recovery.close()
    assert cancelled.is_set()
    assert not recovery._tasks
