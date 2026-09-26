import asyncio
import uuid
from unittest.mock import AsyncMock

import pytest
from mandri.core.types.execution import ProtectionError
from mandri.runtime.start_operations import StartOperations


async def test_duplicate_request_joins_one_execution_and_rejects_changed_input():
    operations = StartOperations[str]()
    identifier = str(uuid.uuid4())
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def create():
        calls.append("created")
        started.set()
        await release.wait()
        return "session"

    first = asyncio.create_task(operations.run(identifier, "same", create))
    await started.wait()
    second = asyncio.create_task(operations.run(identifier, "same", create))
    with pytest.raises(ProtectionError, match="another request"):
        await operations.run(identifier, "changed", create)
    release.set()
    assert await asyncio.gather(first, second) == ["session", "session"]
    assert calls == ["created"]


async def test_cancel_before_request_never_starts_harness():
    operations = StartOperations[str]()
    identifier = str(uuid.uuid4())
    stop = AsyncMock()
    factory = AsyncMock(return_value="session")
    await operations.cancel(identifier, stop)
    with pytest.raises(ProtectionError, match="cancelled"):
        await operations.run(identifier, "request", factory)
    stop.assert_not_awaited()
    factory.assert_not_awaited()


async def test_concurrent_cancellations_wait_until_startup_cleanup_finishes():
    operations = StartOperations[str]()
    identifier = str(uuid.uuid4())
    started = asyncio.Event()
    cleaning = asyncio.Event()
    finish_cleanup = asyncio.Event()

    async def create():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await finish_cleanup.wait()

    running = asyncio.create_task(operations.run(identifier, "same", create))
    await started.wait()
    first = asyncio.create_task(operations.cancel(identifier, AsyncMock()))
    await cleaning.wait()
    second = asyncio.create_task(operations.cancel(identifier, AsyncMock()))
    await asyncio.sleep(0)
    assert not first.done() and not second.done()
    finish_cleanup.set()
    await asyncio.gather(first, second)
    with pytest.raises(ProtectionError, match="cancelled"):
        await running


async def test_late_cancel_stops_created_session_once():
    operations = StartOperations[str]()
    identifier = str(uuid.uuid4())
    assert await operations.run(identifier, "same", AsyncMock(return_value="session")) == "session"
    stop = AsyncMock()
    await operations.cancel(identifier, stop)
    await operations.cancel(identifier, stop)
    stop.assert_awaited_once_with("session")
