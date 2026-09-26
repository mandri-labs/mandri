"""Tests for the HarnessControl port consumer path in the runtime service."""

import pytest
from mandri.core.ids import ApprovalDecision, HarnessSessionId, ModeApplication
from mandri.core.ports.control import (
    ControlRequest,
    ControlSink,
    HarnessControl,
    PromptOutcome,
    PromptState,
)
from mandri.runtime.control.errors import SteerUnsupportedError
from mandri.runtime.errors import SessionNotRunningError
from mandri.runtime.service import RuntimeService


class FakeHarnessControl:
    def __init__(self) -> None:
        self.modes: list[str] = []
        self.prompts: list[str] = []
        self.interrupts = 0
        self.closed = False

    async def answer_approval(self, native_request_ref: str, decision: ApprovalDecision) -> bool:
        return decision is ApprovalDecision.ALLOW

    async def set_mode(self, mode: str) -> ModeApplication:
        self.modes.append(mode)
        return ModeApplication.MID_SESSION_APPLIED

    async def send_prompt(self, content: str) -> PromptOutcome:
        self.prompts.append(content)
        return PromptOutcome(state=PromptState.QUEUED)

    async def interrupt(self) -> bool:
        self.interrupts += 1
        return True

    async def capture_identity(self) -> HarnessSessionId | None:
        return HarnessSessionId("native-1")

    async def next_native_request(self) -> ControlRequest:
        return ControlRequest(
            native_request_ref="ref",
            tool_name="bash",
            tool_input={},
            tool_use_id="t1",
        )

    async def aclose(self) -> None:
        self.closed = True


class FakeSink:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.drained = False

    def write(self, data: bytes) -> None:
        self.chunks.append(data)

    async def drain(self) -> None:
        self.drained = True


def test_fakes_satisfy_the_control_ports() -> None:
    controls: dict[str, HarnessControl] = {"s1": FakeHarnessControl()}
    sinks: dict[str, ControlSink] = {"s1": FakeSink()}
    assert "s1" in controls and "s1" in sinks


async def test_unknown_session_control_rejected() -> None:
    service = RuntimeService(harness_commands={})
    with pytest.raises(SteerUnsupportedError):
        await service.set_session_mode("missing", "default")


async def test_send_prompt_requires_live_session() -> None:
    service = RuntimeService(harness_commands={})
    with pytest.raises(SessionNotRunningError):
        await service.send_session_prompt("s1", "hi")


async def test_control_delegation_round_trip() -> None:
    service = RuntimeService(harness_commands={})
    control = FakeHarnessControl()
    service._session_state("s1").control = control
    service.registry.mark_live("s1")
    application = await service.set_session_mode("s1", "acceptEdits")
    assert application is ModeApplication.MID_SESSION_APPLIED
    assert control.modes == ["acceptEdits"]
    outcome = await service.send_session_prompt("s1", "hi")
    assert outcome.state is PromptState.QUEUED
    assert control.prompts == ["hi"]
    assert await service.interrupt_session("s1") is True
    assert control.interrupts == 1


async def test_control_sink_writes_and_drains() -> None:
    sink = FakeSink()
    sink.write(b"data")
    await sink.drain()
    assert sink.chunks == [b"data"]
    assert sink.drained
