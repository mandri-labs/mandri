"""Tests for push-based identity reveal from the claude control adapter."""

import asyncio
import dataclasses
import typing
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import mandri.runtime.native_id as native_id
import pytest
from mandri.core.hub import Hub
from mandri.core.ids import ApprovalDecision, HarnessSessionId, ModeApplication, SessionId
from mandri.core.ports.control import (
    ControlRequest,
    HarnessControl,
    PromptOutcome,
    PromptState,
)
from mandri.core.types.execution import ExecutionBackend
from mandri.runtime.adapters import AdapterContext, HarnessAdapters
from mandri.runtime.control.claude import ClaudeControlAdapter
from mandri.runtime.control.errors import ControlTransportError
from mandri.runtime.errors import OpencodeSessionMissingError
from mandri.runtime.process import ManagedProcess
from mandri.runtime.pump import LinePump
from mandri.runtime.service import RuntimeService
from mandri.sessions.service import SessionsService


class FakeByteStream:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()

    def feed(self, line: bytes) -> None:
        self._queue.put_nowait(line)

    async def chunks(self) -> AsyncIterator[bytes]:
        while True:
            yield await self._queue.get()


class FakeSink:
    def write(self, data: bytes) -> None:
        pass

    async def drain(self) -> None:
        return None


class FakeUnrevealedControl:
    def __init__(self) -> None:
        self.closed = False

    async def answer_approval(self, native_request_ref: str, decision: ApprovalDecision) -> bool:
        return decision is ApprovalDecision.ALLOW

    async def set_mode(self, mode: str) -> ModeApplication:
        return ModeApplication.MID_SESSION_APPLIED

    async def send_prompt(self, content: str) -> PromptOutcome:
        return PromptOutcome(state=PromptState.QUEUED)

    async def interrupt(self) -> bool:
        return True

    async def capture_identity(self) -> HarnessSessionId | None:
        return None

    async def next_native_request(self) -> ControlRequest:
        return ControlRequest(
            native_request_ref="ref",
            tool_name="bash",
            tool_input={},
            tool_use_id="t1",
        )

    async def aclose(self) -> None:
        self.closed = True


class StubSessions:
    def __init__(self) -> None:
        self.revealed: list[tuple[str, str]] = []

    async def reveal_native_id(self, session_id: object, native_id: HarnessSessionId) -> None:
        self.revealed.append((str(session_id), str(native_id)))


async def _no_lines() -> AsyncIterator[bytes]:
    return
    yield


def _identity_frame(session_id: str) -> bytes:
    return f'{{"type":"system","subtype":"init","session_id":"{session_id}"}}\n'.encode()


def _claude_adapter(
    stream: FakeByteStream, on_identity: typing.Callable[[HarnessSessionId], None] | None
) -> ClaudeControlAdapter:
    return ClaudeControlAdapter(
        stdout_pump=LinePump(stream.chunks),
        stderr_pump=LinePump(_no_lines),
        stdin=FakeSink(),
        on_identity=on_identity,
    )


async def test_late_identity_frame_pushes_once_to_callback() -> None:
    stream = FakeByteStream()
    received: list[HarnessSessionId] = []
    adapter = _claude_adapter(stream, received.append)
    capture = asyncio.create_task(adapter.capture_identity())
    stream.feed(_identity_frame("claude-1"))
    identity = await asyncio.wait_for(capture, timeout=1.0)
    await adapter.aclose()
    assert identity == HarnessSessionId("claude-1")
    assert received == [HarnessSessionId("claude-1")]


async def test_late_identity_frame_without_callback_still_captures() -> None:
    stream = FakeByteStream()
    adapter = _claude_adapter(stream, None)
    capture = asyncio.create_task(adapter.capture_identity())
    stream.feed(_identity_frame("claude-2"))
    identity = await asyncio.wait_for(capture, timeout=1.0)
    await adapter.aclose()
    assert identity == HarnessSessionId("claude-2")


async def test_service_reveals_pushed_identity() -> None:
    stub = StubSessions()
    sessions = typing.cast(SessionsService, stub)
    captured: dict[str, object] = {}

    def factory(context: AdapterContext) -> HarnessAdapters:
        captured["on_identity"] = context.on_identity
        control: HarnessControl = FakeUnrevealedControl()
        return HarnessAdapters(control=control)

    service = RuntimeService(harness_commands={}, sessions=sessions, hub=Hub(), adapters=factory)
    service.registry.mark_live("s1")
    process = typing.cast(ManagedProcess, object())
    service._attach_control("s1", "claude", process, None)
    on_identity = captured["on_identity"]
    assert callable(on_identity)
    on_identity(HarnessSessionId("native-9"))
    for _ in range(5):
        await asyncio.sleep(0)
    assert stub.revealed == [("s1", "native-9")]


class FakeProbeResponse:
    def __init__(self, status_code: int, payload: typing.Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> typing.Any:
        return self._payload


ProbeOutcome = FakeProbeResponse | httpx.HTTPError


class FakeProbeClient:
    def __init__(self, outcomes: typing.Sequence[ProbeOutcome]) -> None:
        self._outcomes = outcomes
        self._index = 0

    async def __aenter__(self) -> typing.Any:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get(self, url: str) -> FakeProbeResponse:
        outcome = self._outcomes[min(self._index, len(self._outcomes) - 1)]
        self._index += 1
        if isinstance(outcome, httpx.HTTPError):
            raise outcome
        return outcome


def _install_probe_client(
    monkeypatch: pytest.MonkeyPatch, outcomes: typing.Sequence[ProbeOutcome]
) -> None:
    client = FakeProbeClient(outcomes)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: client)


async def test_verify_opencode_session_returns_existing_id_on_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_probe_client(monkeypatch, [FakeProbeResponse(200, {"data": {"id": "ses_existing"}})])
    identity = await native_id.verify_opencode_session_id(4096, "ses_existing")
    assert identity == HarnessSessionId("ses_existing")


async def test_verify_opencode_session_fails_fast_on_missing_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_probe_client(
        monkeypatch, [FakeProbeResponse(404), FakeProbeResponse(200, {"status": "completed"})]
    )
    with pytest.raises(OpencodeSessionMissingError):
        await native_id.verify_opencode_session_id(4096, "ses_gone")


def test_opencode_session_missing_error_is_domain_rooted() -> None:
    from mandri.core.errors import MandriError
    from mandri.runtime.errors.base import RuntimeDomainError

    assert issubclass(OpencodeSessionMissingError, MandriError)
    assert issubclass(OpencodeSessionMissingError, RuntimeDomainError)
    assert not issubclass(OpencodeSessionMissingError, RuntimeError)


async def test_verify_opencode_session_retries_until_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_probe_client(
        monkeypatch,
        [httpx.ConnectError("refused"), FakeProbeResponse(200, {"data": {"id": "ses_retry"}})],
    )
    identity = await native_id.verify_opencode_session_id(4096, "ses_retry")
    assert identity == HarnessSessionId("ses_retry")


async def test_verify_opencode_session_fails_when_process_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_probe_client(monkeypatch, [httpx.ConnectError("refused")])
    with pytest.raises(ControlTransportError):
        await native_id.verify_opencode_session_id(4096, "ses_down", alive=lambda: False)


async def test_explicit_reset_rotates_docker_conversation_without_resume_rejection() -> None:
    sessions = SimpleNamespace(rotate_native_id=AsyncMock())
    runtime = RuntimeService(harness_commands={}, sessions=typing.cast(SessionsService, sessions))
    process = typing.cast(ManagedProcess, object())
    runtime.registry.mark_live("s1", process)
    state = runtime._session_state("s1")
    state.native_id = HarnessSessionId("old")
    state.policy = dataclasses.replace(state.policy, execution_backend=ExecutionBackend.DOCKER)
    callback = runtime._on_conversation_reset("s1", process)
    callback(HarnessSessionId("new"))
    await asyncio.gather(*runtime._control_tasks)
    assert state.native_id == "new"
    sessions.rotate_native_id.assert_awaited_once_with(SessionId("s1"), "new")
    callback(HarnessSessionId("new"))
    await asyncio.gather(*runtime._control_tasks)
    assert sessions.rotate_native_id.await_count == 1


async def test_reset_from_replaced_process_cannot_rotate_new_session() -> None:
    sessions = SimpleNamespace(rotate_native_id=AsyncMock())
    runtime = RuntimeService(harness_commands={}, sessions=typing.cast(SessionsService, sessions))
    old = typing.cast(ManagedProcess, object())
    runtime.registry.mark_live("s1", typing.cast(ManagedProcess, object()))
    await runtime._accept_conversation_reset("s1", HarnessSessionId("old-reset"), old)
    sessions.rotate_native_id.assert_not_awaited()
