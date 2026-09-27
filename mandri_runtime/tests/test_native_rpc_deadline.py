import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.runtime.control.codex import CodexControlAdapter
from mandri.runtime.control.pi import PiControlAdapter


@pytest.mark.parametrize("kind", ["codex", "pi"])
async def test_native_rpc_deadline_includes_stalled_stdin(kind, monkeypatch):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def drain():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    stdin = SimpleNamespace(write=Mock(), drain=drain)
    if kind == "codex":
        control = CodexControlAdapter(Mock(), stdin)
        control._ensure_reader = Mock()
        command = "initialize"
    else:
        control = PiControlAdapter(Mock(), stdin)
        control._ensure_reader = Mock()
        command = "get_state"
    monkeypatch.setattr(
        f"mandri.runtime.control.{kind}.RPC_RESPONSE_TIMEOUT_SECONDS", 0.01, raising=False
    )
    request = asyncio.create_task(control._call(command, {}))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.sleep(0.03)
        assert request.done(), "Native RPC deadline did not include the blocked stdin write"
        with pytest.raises(TimeoutError):
            await request
        assert cancelled.is_set()
        assert not control._pending
    finally:
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.parametrize("command", ["prompt", "compact", "bash"])
async def test_pi_execution_requests_remain_unbounded(command, monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def drain():
        entered.set()
        await release.wait()

    control = PiControlAdapter(Mock(), SimpleNamespace(write=Mock(), drain=drain))
    control._ensure_reader = Mock()
    monkeypatch.setattr(
        "mandri.runtime.control.pi.RPC_RESPONSE_TIMEOUT_SECONDS", 0.01, raising=False
    )
    request = asyncio.create_task(control._call(command, {}))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.sleep(0.03)
        assert not request.done()
        release.set()
        next(iter(control._pending.values())).set_result(
            {"type": "response", "command": command, "success": True, "data": {"done": True}}
        )
        assert await asyncio.wait_for(request, 1) == {"done": True}
        assert not control._pending
    finally:
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)


async def test_codex_handshake_notification_has_existing_rpc_deadline(monkeypatch):
    entered = asyncio.Event()

    async def drain():
        entered.set()
        await asyncio.Event().wait()

    control = CodexControlAdapter(Mock(), SimpleNamespace(write=Mock(), drain=drain))
    control._ensure_reader = Mock()
    control._call = AsyncMock(return_value={"result": {}})
    monkeypatch.setattr("mandri.runtime.control.codex.RPC_RESPONSE_TIMEOUT_SECONDS", 0.01)
    request = asyncio.create_task(control.capture_identity())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.sleep(0.03)
        assert request.done(), "Native initialized notification did not respect the RPC deadline"
        with pytest.raises(TimeoutError):
            await request
        control._call.assert_awaited_once()
        assert control._call.call_args.args[0] == "initialize"
        assert not control._pending
    finally:
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)
