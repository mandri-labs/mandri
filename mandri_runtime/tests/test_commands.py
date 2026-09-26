import asyncio
from typing import Any

import pytest
from mandri.core.protocol.errors import ProtocolError
from mandri.runtime.commands import CommandService
from mandri.runtime.control.errors import ControlError, ControlTransportError


class Control:
    commands_cancellable = True

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = [{"id": "native", "name": "native"}]
        self.result: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self.calls: list[tuple[str, str]] = []
        self.interrupts = 0
        self.discovery: asyncio.Event | None = None

    async def list_commands(self) -> list[dict[str, Any]]:
        if self.discovery is not None:
            await self.discovery.wait()
        return self.rows

    async def execute_command(self, command_id: str, arguments: str) -> dict[str, Any]:
        self.calls.append((command_id, arguments))
        return await self.result

    async def interrupt(self) -> bool:
        self.interrupts += 1
        return True


class Harness:
    def __init__(self) -> None:
        self.control: Control | None = Control()
        self.live = True
        self.busy = False
        self.service = CommandService(
            lambda _: self.control, lambda _: self.live, lambda _: self.busy
        )

    async def finish(self) -> None:
        await self.service._tasks["session"]


async def test_idempotency_retains_result_and_rejects_identity_reuse() -> None:
    harness = Harness()
    control = harness.control
    assert control is not None
    first, duplicate = await asyncio.gather(
        harness.service.invoke("session", "one", "native", "arg"),
        harness.service.invoke("session", "one", "native", "arg"),
    )
    assert first is duplicate
    assert first.arguments == "arg"
    control.result.set_result({"kind": "text", "text": "Done"})
    await harness.finish()
    assert first.state == "succeeded"
    assert control.calls == [("native", "arg")]
    assert await harness.service.invoke("session", "one", "native", "arg") is first
    with pytest.raises(ProtocolError, match="identity already used"):
        await harness.service.invoke("session", "one", "native", "different")


@pytest.mark.parametrize("rows", [[], [{"id": "native", "name": "native", "available": False}]])
async def test_catalog_changes_do_not_send_unknown_commands(rows: list[dict[str, Any]]) -> None:
    harness = Harness()
    assert harness.control is not None
    harness.control.rows = rows
    with pytest.raises(ProtocolError, match="no longer available"):
        await harness.service.invoke("session", "one", "native", "")
    assert not harness.control.calls


@pytest.mark.parametrize(
    "error,state",
    [
        (ControlError("Rejected"), "failed"),
        (ControlTransportError("EOF"), "unknown"),
        (TimeoutError(), "unknown"),
        (ValueError("Malformed result"), "unknown"),
    ],
)
async def test_failure_and_uncertainty_have_distinct_retry_rules(
    error: Exception, state: str
) -> None:
    harness = Harness()
    assert harness.control is not None
    record = await harness.service.invoke("session", "one", "native", "")
    harness.control.result.set_exception(error)
    await harness.finish()
    assert record.state == state
    assert harness.service.active("session") is (state == "unknown")
    if state == "unknown":
        with pytest.raises(ProtocolError, match="active operation"):
            await harness.service.invoke("session", "two", "native", "")
        harness.control = Control()
        assert not harness.service.active("session")
        new_record = await harness.service.invoke("session", "two", "native", "")
        harness.control.result.set_result({"kind": "text", "text": "Done"})
        await harness.finish()
        assert new_record.state == "succeeded"


async def test_disconnect_before_execution_still_marks_unknown() -> None:
    harness = Harness()
    record = await harness.service.invoke("session", "one", "native", "")
    harness.service.disconnected("session")
    assert record.state == "unknown"
    assert harness.service.active("session")
    with pytest.raises(asyncio.CancelledError):
        await harness.finish()


async def test_session_changed_during_discovery_prevents_dispatch() -> None:
    harness = Harness()
    old = harness.control
    assert old is not None
    old.discovery = asyncio.Event()
    invocation = asyncio.create_task(harness.service.invoke("session", "one", "native", ""))
    await asyncio.sleep(0)
    harness.control = Control()
    old.discovery.set()
    with pytest.raises(ProtocolError, match="Session changed"):
        await invocation
    assert not old.calls


async def test_busy_and_missing_owner_rejected() -> None:
    harness = Harness()
    harness.busy = True
    with pytest.raises(ProtocolError, match="active operation"):
        await harness.service.invoke("session", "one", "native", "")
    harness.control = None
    with pytest.raises(ProtocolError, match="Resume"):
        await harness.service.invoke("session", "one", "native", "")


async def test_cancel_does_not_interrupt_replacement_process() -> None:
    harness = Harness()
    old = harness.control
    assert old is not None
    record = await harness.service.invoke("session", "one", "native", "")
    harness.control = Control()
    await harness.service.cancel("session", "one")
    assert record.state == "unknown"
    assert harness.control.interrupts == 0
    old.result.set_result({"kind": "text", "text": "Done"})
    await harness.finish()


async def test_busy_changed_during_discovery_prevents_dispatch() -> None:
    harness = Harness()
    control = harness.control
    assert control is not None
    control.discovery = asyncio.Event()
    invocation = asyncio.create_task(harness.service.invoke("session", "one", "native", ""))
    await asyncio.sleep(0)
    harness.busy = True
    control.discovery.set()
    with pytest.raises(ProtocolError, match="became busy"):
        await invocation
    assert not control.calls
