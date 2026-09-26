from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.ids import HarnessSessionId
from mandri.runtime.control.codex import CodexControlAdapter
from mandri.runtime.control.errors import ControlTransportError


async def test_native_fork_preserves_selected_policy_and_defers_automatic_goal_turns():
    params = {
        "model": "selected/model",
        "modelProvider": "mandri",
        "sandbox": "workspace-write",
        "approvalPolicy": "on-request",
        "cwd": "/workspace",
        "runtimeWorkspaceRoots": ["/workspace"],
    }
    control = CodexControlAdapter(
        None,
        None,
        params,
        fork_thread_id=HarnessSessionId("original"),
        fork_path="/home/worker/.codex/selected.jsonl",
    )
    control._handshake = AsyncMock()
    control._call = AsyncMock(return_value={"result": {"thread": {"id": "new-native"}}})
    assert await control.capture_identity() == "new-native"
    control._call.assert_awaited_once_with(
        "thread/fork",
        {
            **params,
            "threadId": "original",
            "path": "/home/worker/.codex/selected.jsonl",
            "deferGoalContinuation": True,
        },
    )
    assert await control.capture_identity() == "new-native"
    assert control._call.await_count == 1


async def test_fork_never_claims_source_native_identity():
    control = CodexControlAdapter(None, None, fork_thread_id=HarnessSessionId("source"))
    control._handshake = AsyncMock()
    control._call = AsyncMock(return_value={"result": {"thread": {"id": "source"}}})
    with pytest.raises(ControlTransportError, match="source conversation"):
        await control.capture_identity()
    assert control._thread_id is None


async def test_experimental_path_capability_is_explicit_for_fork_only():
    control = CodexControlAdapter(None, None, fork_thread_id=HarnessSessionId("source"))
    control._ensure_reader = Mock()
    control._call = AsyncMock(return_value={"result": {}})
    control._send = AsyncMock()
    await control._handshake()
    assert control._call.call_args.args[1]["capabilities"] == {"experimentalApi": True}


def test_resume_and_fork_cannot_be_combined():
    with pytest.raises(ValueError, match="mutually exclusive"):
        CodexControlAdapter(
            None,
            None,
            resume_thread_id=HarnessSessionId("one"),
            fork_thread_id=HarnessSessionId("two"),
        )
